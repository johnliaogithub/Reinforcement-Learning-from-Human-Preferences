import pandas as pd
import numpy as np
import os
import glob

def calculate_3d_iou(pos1, size1, pos2, size2):
    """Calculates 3D IoU between two volumes."""
    low1, high1 = pos1 - size1/2, pos1 + size1/2
    low2, high2 = pos2 - size2/2, pos2 + size2/2
    inter_dims = np.maximum(0, np.minimum(high1, high2) - np.maximum(low1, low2))
    inter_vol = np.prod(inter_dims)
    union_vol = np.prod(size1) + np.prod(size2) - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0

def generate_metrics_directory(input_dir, output_dir="metrics_output"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Settings for your specific environment
    target_pos = np.array([0.1, -0.2, 0.82]) 
    obj_size = np.array([0.045, 0.045, 0.045])
    goal_size = np.array([0.08, 0.08, 0.08])

    # Find all trajectory CSVs
    csv_files = glob.glob(os.path.join(input_dir, "full_trajectory_data*.csv"))
    
    summary_list = []

    for file_path in csv_files:
        print(f"Processing {os.path.basename(file_path)}...")
        df = pd.read_csv(file_path)
        
        # Group by demo_id to analyze each demo separately
        for demo_id, group in df.groupby('demo_id'):
            # Extract Milk positions (Indices 9, 10, 11)
            # Using column names if they exist, otherwise indices
            milk_cols = ['milk_x', 'milk_y', 'milk_z']
            
            final_pos = group[milk_cols].iloc[-1].values
            start_pos = group[milk_cols].iloc[0].values
            max_z = group['milk_z'].max()
            
            # Metrics
            iou = calculate_3d_iou(final_pos, obj_size, target_pos, goal_size)
            
            # Categorization Logic
            if iou > 0.3:
                status = "Success"
            elif max_z < (start_pos[2] + 0.03): # If it didn't lift at least 3cm
                status = "Grasp Failure"
            else:
                status = "Placement Miss"

            metrics_data = {
                "demo_id": demo_id,
                "source_file": os.path.basename(file_path),
                "final_iou": round(iou, 4),
                "max_lift_height": round(max_z, 4),
                "status": status,
                "duration_steps": len(group)
            }
            summary_list.append(metrics_data)

            # Save individual metric file for this demo
            demo_metric_filename = f"metrics_{demo_id}.csv"
            pd.DataFrame([metrics_data]).to_csv(os.path.join(output_dir, demo_metric_filename), index=False)

    # Save a master summary file
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv("master_metrics_summary.csv", index=False)
    
    print("\n--- Processing Complete ---")
    print(f"Individual metrics saved to: {output_dir}/")
    print("Master summary saved to: master_metrics_summary.csv")
    print(summary_df['status'].value_counts())

if __name__ == "__main__":
    # Run from your current trajectories directory
    generate_metrics_directory("/home/johnl/RLHF/trajectories")
