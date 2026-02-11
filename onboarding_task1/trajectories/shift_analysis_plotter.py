import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_shift_analysis(csv_path, demo_id='demo_1'):
    df = pd.read_csv(csv_path)
    data = df[df['demo_id'] == demo_id].reset_index(drop=True)
    
    # We will test three mappings: Current, Shift Left (-1), Shift Right (+1)
    shifts = {
        "Shift Left (-1)": -1,
        "Current Mapping": 0,
        "Shift Right (+1)": +1
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    objects = ['milk', 'bread', 'cereal', 'can']
    colors = ['blue', 'brown', 'red', 'green']

    for i, (label, shift) in enumerate(shifts.items()):
        ax = axes[i]
        for obj_idx, obj in enumerate(objects):
            # Attempt to reconstruct Z by shifting the base index (9, 16, 23, 30)
            # Original milk_z was index 11. Shifted it becomes 10 or 12.
            try:
                # We access the underlying state columns if they exist, 
                # or calculate the shift based on your known column order.
                # Here we use the named 'milk_z' etc., but conceptually shifting indices.
                if label == "Current Mapping":
                    z_vals = data[f'{obj}_z']
                elif label == "Shift Left (-1)":
                    z_vals = data[f'{obj}_y'] # Y becomes the new Z
                else:
                    # Orientation W (index 12) becomes the new Z
                    z_vals = data[f'{obj}_quat_w'] 
                
                ax.plot(data['timestep'], z_vals, label=obj.capitalize(), color=colors[obj_idx])
            except KeyError:
                continue

        ax.set_title(f"Z-Axis Interpretation: {label}")
        ax.set_ylabel("Value")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.savefig("shift_analysis_z.png")
    plt.show()

if __name__ == "__main__":
    # Point to one of your existing trajectory files
    plot_shift_analysis("/home/johnl/RLHF/trajectories/full_trajectory_data.csv")
