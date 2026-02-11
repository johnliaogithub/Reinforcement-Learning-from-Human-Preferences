import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def plot_trajectories(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    unique_demos = df['demo_id'].unique()
    demo_to_plot = unique_demos[0]
    data = df[df['demo_id'] == demo_to_plot].reset_index(drop=True)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define the objects and their colors
    objects = {
        'milk': 'blue',
        'bread': 'brown',
        'cereal': 'red',
        'can': 'green'
    }

    # Column list to find indices
    cols = list(data.columns)

    for obj, color in objects.items():
        # Shifting Right: 
        # The actual X is in the column currently labeled Y
        # The actual Y is in the column currently labeled Z
        # The actual Z is in the column currently labeled QUAT_W
        try:
            x_col_idx = cols.index(f'{obj}_x') + 1
            y_col_idx = cols.index(f'{obj}_y') + 1
            z_col_idx = cols.index(f'{obj}_z') + 1

            x = data.iloc[:, x_col_idx]
            y = data.iloc[:, y_col_idx]
            z = data.iloc[:, z_col_idx]
            
            # Plot the path
            ax.plot(x, y, z, label=obj.capitalize(), color=color, linewidth=2)
            
            # Mark the start (o) and end (X) points
            ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color=color, s=50, marker='o') 
            ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color=color, s=100, marker='X') 
        except (ValueError, IndexError):
            print(f"Warning: Could not shift indices for {obj}. Check column names.")

    ax.set_xlabel('Real X Position')
    ax.set_ylabel('Real Y Position')
    ax.set_zlabel('Real Z Position (Height)')
    ax.set_title(f'Corrected Object Trajectories (Shift Right) - {demo_to_plot}')
    ax.legend()
    
    ax.view_init(elev=20, azim=45)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="full_trajectory_data4.csv")
    args = parser.parse_args()
    plot_trajectories(args.path)
