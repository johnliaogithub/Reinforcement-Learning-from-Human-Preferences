import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_height_trajectories(input_dir, output_dir="z_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    objects = {
        'milk': 'blue',
        'bread': 'brown',
        'cereal': 'red',
        'can': 'green'
    }

    # Find all trajectory CSVs in the specified directory
    csv_files = glob.glob(os.path.join(input_dir, "full_trajectory_data*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        base_name = os.path.basename(csv_path).replace(".csv", "")
        
        for demo_id in df['demo_id'].unique():
            data = df[df['demo_id'] == demo_id]
            
            plt.figure(figsize=(10, 5))
            
            for obj, color in objects.items():
                z_col = f'{obj}_z'
                if z_col in data.columns:
                    # Plot Timestep vs Z-coordinate
                    plt.plot(data['timestep'], data[z_col], label=obj.capitalize(), color=color)
            
            plt.axhline(y=0.8, color='black', linestyle='--', label='Approx Table Height')
            plt.xlabel('Timestep')
            plt.ylabel('Z Position (Height)')
            plt.title(f"Z-Axis Height Over Time: {demo_id}")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(output_dir, f"z_debug_{base_name}_{demo_id}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Generated plot: {save_path}")

if __name__ == "__main__":
    # Update this path if your files are in a different location
    plot_height_trajectories("/home/johnl/RLHF/trajectories")
