import pandas as pd
import numpy as np

def calculate_multi_object_metric(csv_path):
    df = pd.read_csv(csv_path)
    
    # Define standard PickPlace targets for the four objects
    # These are based on the second bin (Bin 2) center and offsets
    targets = {
        'milk':   np.array([-0.05765991, 0.10625742, 0.87513866]),
        'bread':  np.array([0.19826297, 0.06911306, 0.84220913]),
        'cereal': np.array([-0.01669539, 0.3772177, 0.87566261]),
        'can':    np.array([0.15380939, 0.31662346, 0.86005651])
    }

    results = []

    for demo_id, group in df.groupby('demo_id'):
        final_frame = group.iloc[-1]
        cols = list(group.columns)
        
        total_distance = 0
        obj_distances = {}

        for obj, target_pos in targets.items():
            # Shift Right Logic:
            # We find the index of the named column and move +1
            try:
                x_idx = cols.index(f'{obj}_x') + 1
                y_idx = cols.index(f'{obj}_y') + 1
                z_idx = cols.index(f'{obj}_z') + 1
                
                current_pos = np.array([
                    final_frame.iloc[x_idx],
                    final_frame.iloc[y_idx],
                    final_frame.iloc[z_idx]
                ])
                
                dist = np.linalg.norm(current_pos - target_pos)
                obj_distances[f'{obj}_dist'] = round(dist, 4)
                total_distance += dist
                
            except (ValueError, IndexError):
                continue

        # Compile demo results
        results.append({
            "demo_id": demo_id,
            "total_dist_sum": round(total_distance, 4),
            **obj_distances
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    metrics_df = calculate_multi_object_metric("full_trajectory_data4.csv")
    print("\n--- Multi-Object Placement Accuracy ---")
    print(metrics_df.to_string(index=False))
    
    # Save to CSV for your report
    metrics_df.to_csv("object_placement_distances.csv", index=False)
