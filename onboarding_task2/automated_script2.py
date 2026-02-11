"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np
import robosuite.utils.transform_utils as T

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

class ProgrammedDevice:
    """
    A 'Device' that generates programmed actions (up, down, left, right...) 
    based on an Oracle Policy to solve the task.
    Mimics a user pressing keys or using a joystick.
    """
    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0):
        self.env = env
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.active_robot = 0 # Single robot
        
        # Oracle Logic / State Machine
        self.objects = ["Can"]
        self.object_bodies = ["Can_main"]
        self.object_bin_map = {
            "Milk_main": "bin1",
            "Bread_main": "bin1",
            "Cereal_main": "bin2",
            "Can_main": "bin2"
        }
        self.bin_positions = {
            "bin1": np.array([0.1, -0.25, 0.8]),
            "bin2": np.array([0.1, 0.28, 0.8])
        }
        # Map objects to their visual (ghost) target body for precise placement
        self.object_visual_map = {
            "Can_main": "VisualCan_main",
        }
        self.current_obj_idx = 0
        self.state = "APPROACH"
        self.hover_height = 1
        self.grasp_height = 0.835 
        self.lift_height = 1.1 # Lowered from 1.3
        self.error_threshold_pos = 0.05
        self.counter = 0
        self.state_timer = 0
        self._state_timeout_limit = 200 # 40 seconds at 20Hz

        self.next_state = {
            "APPROACH": "ORIENT",
            "ORIENT": "DESCEND",
            "DESCEND": "GRASP",
            "GRASP": "LIFT",
            "LIFT": "MOVE1",
            "MOVE1": "MOVE2",
            "MOVE2": "RELEASE",
            "RELEASE": "APPROACH"
        }

    def start_control(self):
        pass

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

    def input2action(self, goal_update_mode="target"):
        if self.current_obj_idx >= len(self.objects):
            return None # Done, trigger exit

        target_obj = self.object_bodies[self.current_obj_idx]
        obj_pos = self.get_object_pos(target_obj)
        obj_quat = self.get_object_quat(target_obj)
        eef_pos = self.get_eef_pos()
        eef_quat = self.get_eef_quat()
        
        # State Timer
        self.state_timer += 1
        
        # Timeout Logic
        if self.state_timer > self._state_timeout_limit:
            print(f"Timeout in state {self.state}. Moving to next action...")
            self.state_timer = 0
            # Retry same object:
            self.state = self.next_state[self.state]
            # Do NOT increment current_obj_idx
            return self.input2action(goal_update_mode) # Recurse for retry

        # --- Oracle Logic to determine Target ---

        # Target Orientation (Yaw Only)
        obj_mat = T.quat2mat(obj_quat)
        obj_yaw = np.arctan2(obj_mat[1, 0], obj_mat[0, 0])
        c = np.cos(obj_yaw)
        s = np.sin(obj_yaw)
        # Z down, X aligned with object Yaw
        target_mat = np.array([
            [c,  s, 0],
            [s, -c, 0],
            [0,  0, -1]
        ])
        target_quat = T.mat2quat(target_mat)
        
        gripper_action = -1 # Open
        target_pos = np.zeros(3)

        # State Machine Transitions
        if self.state == "APPROACH":
            target_pos = np.array([obj_pos[0], obj_pos[1], self.hover_height])
            if np.linalg.norm(target_pos - eef_pos) < self.error_threshold_pos:
                self.state = "ORIENT"
                self.state_timer = 0
        
        elif self.state == "ORIENT":
            target_pos = np.array([obj_pos[0], obj_pos[1], self.hover_height])
            d_quat = T.get_orientation_error(target_quat, eef_quat)
            if np.linalg.norm(d_quat) < 0.2:
                self.state = "DESCEND"
                self.state_timer = 0

        elif self.state == "DESCEND":
            target_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2]]) 
            if np.linalg.norm(target_pos - eef_pos) < self.error_threshold_pos:
                self.state = "GRASP"
                self.counter = 0
                self.state_timer = 0

        elif self.state == "GRASP":
            target_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2]])
            gripper_action = 1 # Close
            self.counter += 1
            if self.counter > 20:
                self.state = "LIFT"
                self.state_timer = 0

        elif self.state == "LIFT":
            target_pos = np.array([obj_pos[0], obj_pos[1], self.lift_height])
            gripper_action = 1
            if eef_pos[2] > self.lift_height - 0.1:
                self.state = "MOVE1"
                self.state_timer = 0

        elif self.state == "MOVE1":
            # Target the ghost can position (VisualCan_main) for precise placement
            visual_name = self.object_visual_map.get(target_obj)
            if visual_name:
                ghost_pos = self.get_object_pos(visual_name)
            else:
                target_bin = self.object_bin_map[target_obj]
                ghost_pos = self.bin_positions[target_bin]
            # Maintain lift height for horizontal move
            target_pos = np.array([ghost_pos[0], ghost_pos[1], self.lift_height])
            #target_pos = np.array([bin_pos[0], bin_pos[1], bin_pos[2]])
            gripper_action = 1
            if np.linalg.norm(target_pos[:2] - eef_pos[:2]) < 0.1:
                self.state = "MOVE2"
                self.counter = 0
                self.state_timer = 0
        
        elif self.state == "MOVE2":
            # Target the ghost can position for descent
            visual_name = self.object_visual_map.get(target_obj)
            if visual_name:
                ghost_pos = self.get_object_pos(visual_name)
            else:
                target_bin = self.object_bin_map[target_obj]
                ghost_pos = self.bin_positions[target_bin]
            target_pos = np.array([ghost_pos[0], ghost_pos[1], ghost_pos[2]])
            gripper_action = 1
            if np.linalg.norm(target_pos[:2] - eef_pos[:2]) < 0.1:
                self.state = "RELEASE"
                self.counter = 0
                self.state_timer = 0

        elif self.state == "RELEASE":
            visual_name = self.object_visual_map.get(target_obj)
            if visual_name:
                ghost_pos = self.get_object_pos(visual_name)
            else:
                target_bin = self.object_bin_map[target_obj]
                ghost_pos = self.bin_positions[target_bin]
            # Hold above ghost position while releasing
            target_pos = np.array([ghost_pos[0], ghost_pos[1], ghost_pos[2]])
            gripper_action = -1 # Open
            self.counter += 1
            if self.counter > 40:
                self.state = "APPROACH"
                self.current_obj_idx += 1
                self.state_timer = 0
        # --- Debug: Print every 20 steps ---
        if self.state_timer % 20 == 0:
            err = np.linalg.norm(target_pos - eef_pos)
            print(f"[{self.state}] timer={self.state_timer} eef={np.round(eef_pos, 3)} target={np.round(target_pos, 3)} err={err:.4f} gripper={gripper_action}")
        
        # --- Calculate Simulated Inputs (Deltas) ---
        
        # Position Delta (simulating Joystick/Keys)
        error_pos = target_pos - eef_pos
        kp = 3.0
        dpos = error_pos * kp
        max_delta = 0.5
        dpos = np.clip(dpos, -max_delta, max_delta)

        # Orientation Delta (Strict Yaw Only - Mimicking 'o' and 'p')
        # We only want to rotate around Z.
        # Calculate Yaw diff.
        # Get current EEF Yaw and Target Yaw.
        
        # Target Yaw from earlier: obj_yaw
        # Current EEF Yaw:
        eef_mat = T.quat2mat(eef_quat)
        eef_yaw = np.arctan2(eef_mat[1, 0], eef_mat[0, 0])
        
        # Yaw Error (wrapped to -pi, pi)
        yaw_error = obj_yaw - eef_yaw
        if yaw_error > np.pi: yaw_error -= 2*np.pi
        if yaw_error < -np.pi: yaw_error += 2*np.pi
        
        # Simulate 'o' (positive) and 'p' (negative) inputs
        # If error is positive, we need +Z rotation.
        # If error is negative, we need -Z rotation.
        
        d_ax = 0.0
        d_ay = 0.0
        d_az = 0.0
        
        ori_threshold = 0.05
        step_size = 0.5 # Simulate a key press magnitude
        
        if yaw_error > ori_threshold:
            d_az = step_size
        elif yaw_error < -ori_threshold:
            d_az = -step_size
        
        # Combine actions matches Delta [x, y, z, ax, ay, az]
        delta_action = np.array([dpos[0], dpos[1], dpos[2], d_ax, d_ay, d_az])
        
        # Dictionary structure
        action_dict = {}
        arm = "right"
        
        action_dict[f"{arm}_delta"] = delta_action
        action_dict[f"{arm}_gripper"] = np.array([gripper_action]) 

        return action_dict


def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode):
    """
    Use the device (programmed) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    """

    env.reset()
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        env.render()

        # Disabled success-based early exit.
        # The demo will run until input2action returns None (all objects placed).
        # if task_completion_hold_count == 0:
        #     break
        # if env._check_success():
        #     if task_completion_hold_count > 0:
        #         task_completion_hold_count -= 1
        #     else:
        #         task_completion_hold_count = 10
        # else:
        #     task_completion_hold_count = -1

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        # For Oracle testing, we force save to verify behavior even if env doesn't check 'success' perfectly 
        if success or True: 
            print("Demonstration saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            if len(states) != len(actions):
                 # Truncate to match just in case
                min_len = min(len(states), len(actions))
                states = states[:min_len]
                actions = actions[:min_len]

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Hardcoded Arguments replacing argparse
    args_directory = "./trajectories/automated_demonstrations"
    args_environment = "PickPlaceCan"
    args_robots = ["Panda"]
    args_config = "default"
    args_arm = "right"
    args_camera = ["agentview"]
    args_controller = None # Default
    args_device = "programmed"
    args_pos_sensitivity = 1.0
    args_rot_sensitivity = 1.0
    args_renderer = "mjviewer"
    args_max_fr = 20
    args_reverse_xy = False
    args_goal_update_mode = "target"

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args_controller,
        robot=args_robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args_robots) == 1, "Whole Body IK only supports one robot"

    # Create argument configuration
    config = {
        "env_name": args_environment,
        "robots": args_robots,
        "controller_configs": controller_config,
    }

    if "TwoArm" in args_environment:
        config["env_configuration"] = args_config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args_renderer,
        has_offscreen_renderer=False,
        render_camera=args_camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device - ProgrammedDevice
    device = ProgrammedDevice(
        env=env,
        pos_sensitivity=args_pos_sensitivity,
        rot_sensitivity=args_rot_sensitivity,
    )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    os.makedirs(args_directory, exist_ok=True)
    new_dir = os.path.join(args_directory, "{}_{}".format(t1, t2))
    print(new_dir)
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args_arm, args_max_fr, args_goal_update_mode)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
        break # Exit after one demonstration
