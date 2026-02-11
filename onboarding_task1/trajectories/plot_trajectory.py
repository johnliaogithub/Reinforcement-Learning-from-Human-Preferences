import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def plot_trajectories(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # We'll plot each demo in a separate figure, or just the first one found
    unique_demos = df['demo_id'].unique()
    demo_to_plot = unique_demos[0]
    data = df[df['demo_id'] == demo_to_plot]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define the objects and their colors
    objects = {
        'milk': 'blue',
        'bread': 'brown',
        'cereal': 'red',
        'can': 'green'
    }

    for obj, color in objects.items():
        x = data[f'{obj}_x']
        y = data[f'{obj}_y']
        z = data[f'{obj}_z']
        
        # Plot the path
        ax.plot(x, y, z, label=obj.capitalize(), color=color, linewidth=2)
        
        # Mark the start and end points
        ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color=color, s=50, marker='o') # Start
        ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color=color, s=100, marker='X') # End

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position (Height)')
    ax.set_title(f'Object Trajectories for {demo_to_plot}')
    ax.legend()
    
    # Set a reasonable view angle to see the lift
    ax.view_init(elev=20, azim=45)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="full_trajectory_data3.csv")
    args = parser.parse_args()
    plot_trajectories(args.path)
