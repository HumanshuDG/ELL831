import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Read the grid search and optimization results
df = pd.read_csv("grid_search_results.csv")

# Get parameter names
param_names = [col for col in df.columns if col.startswith("w")]

# Create plots for each pair of parameters
for param1, param2 in itertools.combinations(param_names, 2):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot grid search points
    grid_points = df[df["point_type"] == "grid"]
    ax.scatter(
        grid_points[param1],
        grid_points[param2],
        grid_points["objective_value"],
        c="lightgray",
        alpha=0.5,
        label="Grid Points",
    )

    # Plot optimization paths
    for start_id in range(1, 9):
        path_points = df[df["point_type"] == f"path_{start_id}"]
        if not path_points.empty:
            ax.plot(
                path_points[param1],
                path_points[param2],
                path_points["objective_value"],
                "o-",
                linewidth=2,
                markersize=4,
                label=f"Path {start_id}",
            )

    # Customize plot
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel("Objective Value")
    ax.set_title(f"Optimization Landscape: {param1} vs {param2}")

    # Add legend
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left")

    # Adjust view
    ax.view_init(elev=30, azim=45)

    # Save plot
    plt.savefig(f"optimization_3d_{param1}_{param2}.png", dpi=300, bbox_inches="tight")
    plt.close()

# Create 2D contour plots with paths
for param1, param2 in itertools.combinations(param_names, 2):
    plt.figure(figsize=(12, 8))

    # Create contour from grid points
    grid_points = df[df["point_type"] == "grid"]

    # Create grid for contour
    x = np.linspace(df[param1].min(), df[param1].max(), 100)
    y = np.linspace(df[param2].min(), df[param2].max(), 100)
    X, Y = np.meshgrid(x, y)

    # Interpolate objective values
    from scipy.interpolate import griddata

    Z = griddata(
        (grid_points[param1], grid_points[param2]),
        grid_points["objective_value"],
        (X, Y),
        method="cubic",
    )

    # Plot contour
    plt.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)
    plt.colorbar(label="Objective Value")

    # Plot optimization paths
    for start_id in range(1, 9):
        path_points = df[df["point_type"] == f"path_{start_id}"]
        if not path_points.empty:
            plt.plot(
                path_points[param1],
                path_points[param2],
                "o-",
                linewidth=2,
                markersize=4,
                label=f"Path {start_id}",
            )

    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(f"Optimization Contour: {param1} vs {param2}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save plot
    plt.savefig(
        f"optimization_contour_{param1}_{param2}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
