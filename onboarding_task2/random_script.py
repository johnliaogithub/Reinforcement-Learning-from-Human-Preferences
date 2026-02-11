"""
A script to collect a batch of random demonstrations.
"""

import csv
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

class RandomDevice:
    """
    A 'Device' that generates random actions.
    """
    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0):
        self.env = env
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.active_robot = 0 # Single robot
        
        # Action space bounds (assumed normalized -1 to 1 for delta)
        self.action_dim = 7 # x, y, z, ax, ay, az, gripper
        
        self.t = 0
        self.max_steps = 1000 # Max steps per episode

    def reset(self):
        self.t = 0
        # Re-seed if needed, or just let numpy be random
        pass

    def start_control(self):
        pass
    
    def get_object_pos(self, obj_name):
        obj_id = self.env.sim.model.body_name2id(obj_name)
        return self.env.sim.data.body_xpos[obj_id]
        
    def get_eef_pos(self):
        return self.env.sim.data.site_xpos[self.env.sim.model.site_name2id("gripper0_right_grip_site")]

    def get_eef_quat(self):
        mat = self.env.sim.data.site_xmat[self.env.sim.model.site_name2id("gripper0_right_grip_site")].reshape(3, 3)
        return T.mat2quat(mat)

    # Hardcode bin positions for results calculation
    @property
    def bin_positions(self):
        return {
            "bin1": np.array([0.1, -0.25, 0.8]),
            "bin2": np.array([0.1, 0.28, 0.8])
        }

    def input2action(self, goal_update_mode="target"):
        self.t += 1
        if self.t > self.max_steps:
            return None # Trigger end of episode logic in loop
            
        # Generate random action
        # Position: [-1, 1] scaled by sensitivity
        # For random exploration, usually we want somewhat correlated noise or just small steps.
        # Let's use uniform random [-0.5, 0.5] for position deltas
        dpos = np.random.uniform(-0.5, 0.5, size=3)
        
        # Orientation: [-0.2, 0.2]
        drot = np.random.uniform(-0.2, 0.2, size=3)
        
        # Gripper: Randomly switch state? 
        # Let's bias towards keeping state to avoid chatter
        # But for pure random, let's just pick [-1, 1]
        gripper = np.random.choice([-1, 1])
        
        delta_action = np.concatenate([dpos, drot])
        
        action_dict = {}
        arm = "right"
        action_dict[f"{arm}_delta"] = delta_action
        action_dict[f"{arm}_gripper"] = np.array([gripper])
        
        return action_dict

def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode, episode_num, csv_path):
    """
    Use the device (random) to collect a demonstration.
    """

    env.reset()
    # env.render()
    device.reset()

    task_completion_hold_count = -1 
    device.start_control()

    # Data logging initialization
    episode_data = []
    episode_start_time = time.time()
    is_success = False
    
    file_exists = os.path.isfile(csv_path)

    for robot in env.robots:
        robot.print_action_info_dict()

    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    while True:
        start = time.time()
        active_robot = env.robots[device.active_robot]
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        if input_ac_dict is None:
            break

        from copy import deepcopy
        action_dict = deepcopy(input_ac_dict)

        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        # env.render()
        
        # --- Log Data ---
        current_eef_pos = device.get_eef_pos()
        current_eef_quat = device.get_eef_quat()
        try:
            current_can_pos = device.get_object_pos("Can_main")
        except:
             current_can_pos = np.zeros(3)

        step_record = {
            "time": time.time(),
            "eef_pos": current_eef_pos,
            "eef_quat": current_eef_quat,
            "can_pos": current_can_pos,
            "action": env_action
        }
        episode_data.append(step_record)
        # ----------------

        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
                is_success = True
        else:
            task_completion_hold_count = -1

        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # --- Episode End Logic ---
    duration = time.time() - episode_start_time
    failure_reason = ""
    if not is_success:
        failure_reason = "random_policy"
        
    # Calculate final distance
    try:
        final_can_pos = device.get_object_pos("Can_main")
        # Target is bin2 for Can_main
        target_bin_pos = device.bin_positions["bin2"]
        final_dist = np.linalg.norm(final_can_pos - target_bin_pos)
    except:
        final_dist = -1.0

    # Save to demo.csv
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            headers = ['episode', 'step', 'time', 'duration', 'success', 'failure_type', 
                       'eef_pos_x', 'eef_pos_y', 'eef_pos_z', 
                       'eef_quat_w', 'eef_quat_x', 'eef_quat_y', 'eef_quat_z',
                       'can_pos_x', 'can_pos_y', 'can_pos_z']
            if len(episode_data) > 0:
                ac_dim = len(episode_data[0]['action'])
                headers.extend([f'action_{i}' for i in range(ac_dim)])
            writer.writerow(headers)

        for i, record in enumerate(episode_data):
            row = [
                episode_num,
                i,
                record['time'],
                duration,
                is_success,
                failure_reason,
                record['eef_pos'][0], record['eef_pos'][1], record['eef_pos'][2],
                record['eef_quat'][0], record['eef_quat'][1], record['eef_quat'][2], record['eef_quat'][3],
                record['can_pos'][0], record['can_pos'][1], record['can_pos'][2]
            ]
            row.extend(record['action'])
            writer.writerow(row)
            
    print(f"Logged episode {episode_num} to {csv_path}")

    # --- Save to Results CSV ---
    # Global path: trajectories/random/results.csv
    # csv_path is .../random/<timestamp>/demo.csv
    results_path = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "results.csv")
    folder_name = os.path.basename(os.path.dirname(csv_path))
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_exists = os.path.isfile(results_path)
    
    with open(results_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not results_exists or os.path.getsize(results_path) == 0:
             writer.writerow(['id', 'timestamp', 'success', 'failure_type', 'steps', 'duration', 'final_dist'])
        
        writer.writerow([
            f"{folder_name}_{episode_num}",
            timestamp_str,
            is_success,
            failure_reason,
            len(episode_data),
            duration,
            final_dist
        ])
    print(f"Logged summary to {results_path}")

    env.close()

def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a single hdf5 file.
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")
    num_eps = 0
    env_name = None

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

        if success or True: 
            print("Demonstration saved")
            del states[-1]
            if len(states) != len(actions):
                min_len = min(len(states), len(actions))
                states = states[:min_len]
                actions = actions[:min_len]

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    f.close()


if __name__ == "__main__":
    args_directory = "./trajectories/random"
    args_environment = "PickPlaceCan"
    args_robots = ["Panda"]
    args_config = "default"
    args_arm = "right"
    args_camera = ["agentview"]
    args_controller = None
    args_pos_sensitivity = 1.0
    args_rot_sensitivity = 1.0
    args_renderer = "mjviewer"
    args_max_fr = 20
    args_goal_update_mode = "target"

    controller_config = load_composite_controller_config(
        controller=args_controller,
        robot=args_robots[0],
    )
    
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args_robots) == 1, "Whole Body IK only supports one robot"

    config = {
        "env_name": args_environment,
        "robots": args_robots,
        "controller_configs": controller_config,
    }

    env = suite.make(
        **config,
        has_renderer=False,
        renderer="mjviewer",
        has_offscreen_renderer=False,
        render_camera=args_camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # env = VisualizationWrapper(env)
    env_info = json.dumps(config)
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    device = RandomDevice(
        env=env,
        pos_sensitivity=args_pos_sensitivity,
        rot_sensitivity=args_rot_sensitivity,
    )

    t1, t2 = str(time.time()).split(".")
    os.makedirs(args_directory, exist_ok=True)
    new_dir = os.path.join(args_directory, "{}_{}".format(t1, t2))
    print(new_dir)
    os.makedirs(new_dir)

    episode_count = 0
    csv_filepath = os.path.join(new_dir, "demo.csv")
    print(f"Collecting demonstrations... CSV will be saved to {csv_filepath}")
    
    while episode_count < 30:
        collect_human_trajectory(env, device, args_arm, args_max_fr, args_goal_update_mode, episode_count, csv_filepath)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
        episode_count += 1
