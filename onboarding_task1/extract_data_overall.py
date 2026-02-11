import h5py
import numpy as np
import pandas as pd
import argparse
import os

def calculate_3d_iou(pos1, size1, pos2, size2):
    """Calculates 3D IoU between two volumes."""
    low1, high1 = pos1 - size1/2, pos1 + size1/2
    low2, high2 = pos2 - size2/2, pos2 + size2/2
    inter_dims = np.maximum(0, np.minimum(high1, high2) - np.maximum(low1, low2))
    inter_vol = np.prod(inter_dims)
    union_vol = np.prod(size1) + np.prod(size2) - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0

def process_to_csv(hdf5_path, output_csv="metrics_report.csv"):
    if not os.path.exists(hdf5_path):
        print(f"Error: {hdf5_path} not found.")
        return

    # Define Target Bin Position (Adjust based on your arena setup)
    # Usually around [0.1, -0.2, 0.82] for PickPlace
    target_pos = np.array([0.1, -0.2, 0.82]) 
    obj_size = np.array([0.045, 0.045, 0.045]) # Approx size of Milk carton
    goal_size = np.array([0.08, 0.08, 0.08])   # Size of the bin target area

    all_results = []

    with h5py.File(hdf5_path, "r") as f:
        # Navigate to the demos
        root = f["data"] if "data" in f else f
        demos = list(root.keys())
        
        for demo_id in demos:
            states = root[f"{demo_id}/states"][:]
            
            # --- EXTRACT BASED ON YOUR MAPPING ---
            # Index 9, 10, 11 is Milk Position (X, Y, Z)
            milk_traj = states[:, 9:12]
            final_pos = milk_traj[-1]
            
            # Calculate metrics
            iou = calculate_3d_iou(final_pos, obj_size, target_pos, goal_size)
            max_z = np.max(milk_traj[:, 2])
            
            # Failure Mode Logic
            if iou > 0.3:
                mode = "Success"
            elif max_z < 0.83: # If it never really left the table (table ~0.8)
                mode = "Grasp Failure"
            else:
                mode = "Placement Miss"

            all_results.append({
                "demo_id": demo_id,
                "final_x": round(final_pos[0], 4),
                "final_y": round(final_pos[1], 4),
                "final_z": round(final_pos[2], 4),
                "max_z": round(max_z, 4),
                "iou_score": round(iou, 4),
                "result": mode,
                "steps": len(states)
            })

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    
    # Print Summary to Terminal
    print(f"\nSaved report to {output_csv}")
    print("-" * 30)
    print(df['result'].value_counts(normalize=True).map(lambda n: f'{n:.1%}') )
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to .hdf5 file")
    args = parser.parse_args()
    
    process_to_csv(args.path)
