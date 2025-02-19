import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv("optimization_results.csv")

# Take absolute value of objective_value
df["objective_value"] = df["objective_value"].abs()

# Create the plot
plt.figure(figsize=(12, 8))

# Plot a line for each start_id
for start_id in range(1, 9):
    data = df[df["start_id"] == start_id]
    plt.plot(
        data["iteration"],
        data["objective_value"],
        label=f"Start {start_id}",
        marker="o",
        markersize=4,
        linewidth=2,
    )

# Customize the plot
plt.yscale("log")  # Use log scale for y-axis due to large value range
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("|Objective Value|", fontsize=12)
plt.title("Optimization Progress by Start ID", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig("optimization_progress.png", dpi=300, bbox_inches="tight")
