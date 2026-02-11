import h5py
import numpy as np
import os
import cv2
import robosuite
import json
import glob
from robosuite.controllers import load_composite_controller_config

def get_latest_dir(path):
    dirs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def load_frames(hdf5_path, env):
    frames = []
    if not os.path.exists(hdf5_path):
        print(f"File not found: {hdf5_path}")
        return frames

    f = h5py.File(hdf5_path, "r")
    
    # Use the first valid demo
    demos = list(f["data"].keys())
    # Filter for 'demo_' keys just in case
    demos = [d for d in demos if "demo_" in d]
    if not demos:
        print("No demos found in file")
        return frames
        
    ep = demos[0] # Just pick the first one
    print(f"Loading {ep} from {hdf5_path}")

    # Load model xml
    model_xml = f[f"data/{ep}"].attrs["model_file"]
    
    # Configure env
    env.reset()
    xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    
    # Load initial state and actions
    states = f[f"data/{ep}/states"][()]
    actions = f[f"data/{ep}/actions"][()]
    
    # Set initial state
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    
    # Pre-action frame
    obs = env._get_observations()
    img = obs["agentview_image"]
    img = cv2.flip(img, 0) # Robosuite images are often upside down
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    frames.append(img)
    
    print(f"Rendering {len(actions)} steps...")
    for action in actions:
        env.step(action)
        
        # Capture image
        obs = env._get_observations()
        img = obs["agentview_image"]
        
        # Flip image
        img = cv2.flip(img, 0)
        
        # Convert RGB to BGR for CV2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frames.append(img)
        
    f.close()
    return frames

def main():
    base_dir = "./trajectories"
    auto_dir = get_latest_dir(os.path.join(base_dir, "automated_demonstrations"))
    rand_dir = get_latest_dir(os.path.join(base_dir, "random"))
    
    if not auto_dir or not rand_dir:
        print("Could not find demonstrations directories.")
        if auto_dir: print(f"Found Auto: {auto_dir}")
        else: print("Auto dir missing")
        if rand_dir: print(f"Found Random: {rand_dir}")
        else: print("Random dir missing")
        return

    print(f"Automated Demo: {auto_dir}")
    print(f"Random Demo: {rand_dir}")

    # Use load_composite_controller_config to get the correct structure for Panda
    controller_config = load_composite_controller_config(
        controller=None,
        robot="Panda"
    )

    config = {
        "env_name": "PickPlaceCan",
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }
    
    # Setup Env with Offscreen Renderer
    env = robosuite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=256,
        camera_widths=256,
        reward_shaping=True,
        control_freq=20,
    )

    frames_auto = load_frames(os.path.join(auto_dir, "demo.hdf5"), env)
    frames_rand = load_frames(os.path.join(rand_dir, "demo.hdf5"), env)
    
    if not frames_auto or not frames_rand:
        print("Failed to load frames.")
        return

    # Playback
    max_len = max(len(frames_auto), len(frames_rand))
    
    # Video Writer Setup
    h, w, _ = frames_auto[0].shape
    video_path = "comparison.mp4"
    # codec 'mp4v' for .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (w * 2, h))
    print(f"Saving video to {video_path}...")
    
    print("Starting playback. Press 'q' to quit.")
    
    for i in range(max_len):
        # Freeze logic
        idx_a = min(i, len(frames_auto) - 1)
        idx_r = min(i, len(frames_rand) - 1)
        
        img_a = frames_auto[idx_a]
        img_r = frames_rand[idx_r]
        
        # Add labels
        cv2.putText(img_a, "Automated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_r, "Random", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Combine
        combined_img = np.concatenate((img_a, img_r), axis=1)
        
        # Write frame
        out.write(combined_img)
        
        cv2.imshow("Comparison", combined_img)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    out.release()
    cv2.destroyAllWindows()
    env.close()
    print("Video saved.")

if __name__ == "__main__":
    main()
