import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

def batch_plot_trajectories(input_dir, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use the same mapping we verified earlier
    objects = {
        'milk': 'blue',
        'bread': 'brown',
        'cereal': 'red',
        'can': 'green'
    }

    csv_files = glob.glob(os.path.join(input_dir, "full_trajectory_data*.csv"))

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        base_name = os.path.basename(csv_path).replace(".csv", "")
        
        for demo_id in df['demo_id'].unique():
            data = df[df['demo_id'] == demo_id]
            
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            for obj, color in objects.items():
                # Check if columns exist to avoid KeyErrors
                if f'{obj}_x' in data.columns:
                    ax.plot(data[f'{obj}_x'], data[f'{obj}_y'], data[f'{obj}_z'], 
                            label=obj.capitalize(), color=color)
            
            ax.set_title(f"Trajectory: {demo_id} ({base_name})")
            ax.legend()
            
            # Save instead of showing
            plt.savefig(os.path.join(output_dir, f"{base_name}_{demo_id}.png"))
            plt.close(fig)
            
    print(f"Batch plotting complete. Check the '{output_dir}' directory.")

if __name__ == "__main__":
    batch_plot_trajectories("/home/johnl/RLHF/trajectories")
