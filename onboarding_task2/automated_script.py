import robosuite as suite
import numpy as np
import json
import os
import robosuite.utils.transform_utils as T

class OraclePolicy:
    def __init__(self, env):
        self.env = env
        # Updated Order: Cereal -> Milk -> Can -> Bread
        self.objects = ["Cereal", "Milk", "Can", "Bread"]
        self.object_bodies = ["Cereal_main", "Milk_main", "Can_main", "Bread_main"]
        self.object_bin_map = {
            "Milk_main": "bin1",
            "Bread_main": "bin1",
            "Cereal_main": "bin2",
            "Can_main": "bin2"
        }
        self.bin_positions = {
            "bin1": np.array([0.1, -0.25, 0.85]),
            "bin2": np.array([0.1, 0.28, 0.85])
        }
        self.current_obj_idx = 0
        self.state = "APPROACH" # APPROACH, ORIENT, DESCEND, GRASP, LIFT, MOVE, RELEASE
        self.hover_height = 1.15
        self.grasp_height = 0.835
        self.lift_height = 1.3
        self.error_threshold_pos = 0.02
        self.error_threshold_ori = 0.05
        self.counter = 0

    def get_object_pos(self, obj_name):
        obj_id = self.env.sim.model.body_name2id(obj_name)
        return self.env.sim.data.body_xpos[obj_id]

    def get_object_quat(self, obj_name):
        obj_id = self.env.sim.model.body_name2id(obj_name)
        return self.env.sim.data.body_xquat[obj_id]

    def get_eef_pos(self):
        return self.env.sim.data.site_xpos[self.env.sim.model.site_name2id("gripper0_right_grip_site")]

    def get_eef_quat(self):
        mat = self.env.sim.data.site_xmat[self.env.sim.model.site_name2id("gripper0_right_grip_site")].reshape(3, 3)
        return T.mat2quat(mat)

    def get_action(self):
        if self.current_obj_idx >= len(self.objects):
            return np.zeros(7) # Done

        target_obj = self.object_bodies[self.current_obj_idx]
        obj_pos = self.get_object_pos(target_obj)
        obj_quat = self.get_object_quat(target_obj)
        eef_pos = self.get_eef_pos()
        eef_quat = self.get_eef_quat()
        
        target_pos = np.zeros(3)
        # Calculate Target Rotation (Strict Yaw Only)
        # 1. Get object Euler angles
        obj_mat = T.quat2mat(obj_quat)
        # Extract yaw from rotation matrix. 
        # For a matrix [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
        # If we assume it's mostly upright, yaw is atan2(yx, xx)
        obj_yaw = np.arctan2(obj_mat[1, 0], obj_mat[0, 0])
        
        # 2. Construct Target Quaternion
        # Base: Top Down (Z down). For Panda, this is often a rotation of 180 deg around X.
        # R_x(180) = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        # This makes Z -> -Z, Y -> -Y.
        # Then, we want to rotate around the NEW Z (which is world -Z) by some angle? 
        # Or just rotate around World Z by obj_yaw?
        # Target = RotZ(obj_yaw) * RotX(180)
        
        # RotZ(yaw) = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        # RotX(180) = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        # Result = [[c, s, 0], [s, -c, 0], [0, 0, -1]]
        
        c = np.cos(obj_yaw)
        s = np.sin(obj_yaw)
        
        target_mat = np.array([
            [c,  s, 0],
            [s, -c, 0],
            [0,  0, -1]
        ])
        
        target_quat = T.mat2quat(target_mat)

        gripper = -1 # Open by default

        if self.state == "APPROACH":
            target_pos = np.array([obj_pos[0], obj_pos[1], self.hover_height])
            if np.linalg.norm(target_pos - eef_pos) < self.error_threshold_pos:
                self.state = "ORIENT"
        
        elif self.state == "ORIENT":
            target_pos = np.array([obj_pos[0], obj_pos[1], self.hover_height])
            # Check orientation error
            d_quat = T.get_orientation_error(target_quat, eef_quat)
             # Relaxed threshold for orientation
            if np.linalg.norm(d_quat) < 0.2: 
                self.state = "DESCEND"

        elif self.state == "DESCEND":
            # Dynamic Grasp Height
            target_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2]])
            if np.linalg.norm(target_pos - eef_pos) < self.error_threshold_pos:
                self.state = "GRASP"
                self.counter = 0

        elif self.state == "GRASP":
            target_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2]])
            gripper = 1
            self.counter += 1
            if self.counter > 25: 
                self.state = "LIFT"

        elif self.state == "LIFT":
            target_pos = np.array([obj_pos[0], obj_pos[1], self.lift_height])
            gripper = 1
            if eef_pos[2] > self.lift_height - 0.05:
                 self.state = "MOVE"

        elif self.state == "MOVE":
            target_bin = self.object_bin_map[target_obj]
            target_pos = self.bin_positions[target_bin]
            gripper = 1
            if np.linalg.norm(target_pos[:2] - eef_pos[:2]) < 0.05:
                self.state = "RELEASE"
                self.counter = 0
        
        elif self.state == "RELEASE":
            target_bin = self.object_bin_map[target_obj]
            target_pos = self.bin_positions[target_bin]
            gripper = -1
            self.counter += 1
            if self.counter > 15:
                self.state = "APPROACH"
                self.current_obj_idx += 1

        # Position Control
        kp = 5.0
        dpos = (target_pos - eef_pos) * kp
        
        # Orientation Control
        # Always maintain target orientation in these phases
        d_quat = T.get_orientation_error(target_quat, eef_quat)
        kp_ori = 1.0 # Restore gain
        d_ori = d_quat * kp_ori
        
        action = np.array([dpos[0], dpos[1], dpos[2], d_ori[0], d_ori[1], d_ori[2], gripper])
        return action

def collect_onboarding_data(num_episodes=20):
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=True,        # Set to False for faster collection
        has_offscreen_renderer=True, 
        use_camera_obs=True,      # Needed for Step 4 Video
        reward_shaping=True,
        control_freq=20,
    )

    storage_dir = "./trajectories/automated"
    os.makedirs(storage_dir, exist_ok=True)

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        trajectory_data = []
        
        oracle = OraclePolicy(env)
        
        while not done:
            action = oracle.get_action()
            
            # Simple termination if all objects are moved (optional, or just run for max steps)
            if oracle.current_obj_idx >= 4:
                break
                
            obs, reward, done, info = env.step(action)
            
            if done: break

            # Record state (using the mapping we discovered)
            # trajectory_data.append(env.sim.data.qpos.flatten().tolist())
            # For simplicity in this task, let's just save metadata at the end. 
            # If we need full trajectory, uncomment above.
            trajectory_data.append(env.sim.data.qpos.flatten().tolist())

        # Step 4: Metadata Calculation
        # Check all objects
        success = True
        objects = ["Milk_main", "Bread_main", "Cereal_main", "Can_main"]
        for obj in objects:
            obj_id = env.sim.model.body_name2id(obj)
            pos = env.sim.data.body_xpos[obj_id]
            # Bin 2 is at [0.1, 0.28, 0.8], size is roughly +/- 0.15 in x/y?
            # Let's just check if it's in the general area and high enough
            if not (0.0 < pos[0] < 0.2 and 0.18 < pos[1] < 0.38 and pos[2] > 0.8):
                success = False
        
        metadata = {
            "episode": ep,
            "success": success,
            "duration": len(trajectory_data),
            "failure_type": "none" if success else "incomplete_task"
        }

        # Save Metadata
        with open(f"{storage_dir}/ep_{ep}_meta.json", "w") as f:
            json.dump(metadata, f)
            
    env.close()

if __name__ == "__main__":
    collect_onboarding_data(1) # Run 1 episode to test
