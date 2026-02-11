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

# ... (omitted)

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
        controller="OSC_POSE",
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
    
    print("Starting playback. Press 'q' to quit.")
    
    for i in range(max_len):
        # Freeze logic
        idx_a = min(i, len(frames_auto) - 1)
        idx_r = min(i, len(frames_rand) - 1)
        
        img_a = frames_auto[idx_a]
        img_r = frames_rand[idx_r]
        
        # Add labels
        cv2.putText(img_a, "Automated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_r, "Random", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Combine
        combined_img = np.concatenate((img_a, img_r), axis=1)
        
        cv2.imshow("Comparison", combined_img)
        
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
