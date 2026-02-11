import h5py
import numpy as np
import pandas as pd
import argparse

def export_full_trajectory(hdf5_path, output_csv="full_trajectory_data.csv"):
    # 1. Define Headers based on your mapping
    headers = [
        "robot0_j1", "robot0_j2", "robot0_j3", "robot0_j4", "robot0_j5", "robot0_j6", "robot0_j7",
        "finger_j1", "finger_j2",
        "milk_x", "milk_y", "milk_z", 
        "milk_quat_w", "milk_quat_x", "milk_quat_y", "milk_quat_z",
        "bread_x", "bread_y", "bread_z",
        "bread_quat_w", "bread_quat_x", "bread_quat_y", "bread_quat_z",
        "cereal_x", "cereal_y", "cereal_z",
        "cereal_quat_w", "cereal_quat_x", "cereal_quat_y", "cereal_quat_z",
        "can_x", "can_y", "can_z",
        "can_quat_w", "can_quat_x", "can_quat_y", "can_quat_z"
    ]
    
    # Fill in the rest as generic 'v_index' for the velocity/extra values
    current_len = len(headers)
    for i in range(current_len, 71):
        headers.append(f"state_{i}")

    all_frames = []

    with h5py.File(hdf5_path, "r") as f:
        root = f["data"] if "data" in f else f
        demos = list(root.keys())
        
        for demo_id in demos:
            print(f"Exporting {demo_id}...")
            states = root[f"{demo_id}/states"][:]
            
            for t in range(len(states)):
                # Create a dictionary for this specific timestep
                row = {
                    "demo_id": demo_id,
                    "timestep": t
                }
                # Map the 71 values to our headers
                for i, val in enumerate(states[t]):
                    if i < len(headers):
                        row[headers[i]] = val
                
                all_frames.append(row)

    # 2. Convert to DataFrame and Save
    df = pd.DataFrame(all_frames)
    df.to_csv(output_csv, index=False)
    print(f"\nDone! Exported {len(df)} total timesteps to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    export_full_trajectory(args.path)
