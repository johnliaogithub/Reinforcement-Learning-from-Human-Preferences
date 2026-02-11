import pandas as pd
import numpy as np

def extract_ground_truth_targets(csv_path):
    df = pd.read_csv(csv_path)
    # Filter for the last few frames of a known successful demo
    final_settle_period = df[df['demo_id'] == 'demo_1'].tail(10)
    
    objects = ['milk', 'bread', 'cereal', 'can']
    cols = list(df.columns)
    
    found_targets = {}
    for obj in objects:
        # Apply your 'Shift Right' discovery
        x_idx = cols.index(f'{obj}_x') + 1
        y_idx = cols.index(f'{obj}_y') + 1
        z_idx = cols.index(f'{obj}_z') + 1
        
        # Mean position at the end of a successful run = The Target
        avg_pos = [
            final_settle_period.iloc[:, x_idx].mean(),
            final_settle_period.iloc[:, y_idx].mean(),
            final_settle_period.iloc[:, z_idx].mean()
        ]
        found_targets[obj] = np.array(avg_pos)
        
    return found_targets

# Run this on your data3.csv
targets = extract_ground_truth_targets("./trajectories/full_trajectory_data3.csv")
for obj, pos in targets.items():
    print(f"Discovered {obj} target: {pos}")
