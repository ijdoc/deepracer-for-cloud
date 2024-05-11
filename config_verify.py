import os
import argparse
import numpy as np
from custom_files.reward_function import (
    CONFIG,
    sigmoid,
    get_waypoint_difficulty,
    get_direction,
    get_histogram_value,
    get_direction_change,
    get_next_distinct_index,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import wandb
from utils import reward_config_utils as rcu


def plot_line(line, axs):
    x = [point[0] for point in line]
    y = [point[1] for point in line]
    axs.plot(x, y, linestyle="-", color="gray")


def plot_index(line, axs):
    x = [point[0] for point in line]
    y = [point[1] for point in line]
    for i, (xi, yi) in enumerate(zip(x, y)):
        axs.text(xi, yi, str(i), color="black", fontsize=6, ha="center", va="center")


def main(args):

    # step_reward = CONFIG["step_reward"]
    projected_steps = [i for i in range(100, 501, 10)]
    # step_reward_plot = [
    #     sigmoid(
    #         i,
    #         k=step_reward["k"],
    #         x0=step_reward["x0"],
    #         ymin=step_reward["ymin"],
    #         ymax=step_reward["ymax"],
    #     )
    #     for i in projected_steps
    # ]
    # aggregated_reward = [
    #     projected_steps[i] * step_reward_plot[i] for i in range(len(projected_steps))
    # ]

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Check if the track file exists
    if os.path.exists(f"{CONFIG['track']}.npy"):
        npy_data = np.load(f"{CONFIG['track']}.npy")
    else:
        # Download the track
        npy_data = rcu.download(CONFIG["track"])

    waypoint_count = npy_data.shape[0]
    waypoints = list([tuple(sublist[0:2]) for sublist in npy_data])
    inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
    outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])

    # Start the plots
    for j in range(2):
        plot_line(inner_line, axs[j])
        plot_line(outer_line, axs[j])
        plot_index(inner_line, axs[j])
        plot_index(outer_line, axs[j])
        axs[j].grid(True)

    # Second pass to plot verifications
    for i in range(waypoint_count):
        direction = get_direction(i, waypoints)
        dir_change, difficulty, norm_difficulty = get_waypoint_difficulty(
            i,
            waypoints,
            skip_ahead=CONFIG["difficulty"]["skip-ahead"],
            look_ahead=CONFIG["difficulty"]["look-ahead"],
            max_val=CONFIG["difficulty"]["max"],
            min_val=CONFIG["difficulty"]["min"],
        )
        # weighted_difficulty = sigmoid(
        #     difficulty,
        #     k=CONFIG["difficulty"]["weighting"]["k"],
        #     x0=CONFIG["difficulty"]["weighting"]["x0"],
        #     ymin=CONFIG["difficulty"]["weighting"]["ymin"],
        #     ymax=CONFIG["difficulty"]["weighting"]["ymax"],
        # )
        importance = get_histogram_value(
            get_direction_change(i, waypoints),
            CONFIG["importance"],
        )
        throttle = get_histogram_value(
            difficulty,
            CONFIG["difficulty"]["histogram"],
        )
        throttle -= CONFIG["agent"]["speed"]["low"]
        throttle /= CONFIG["agent"]["speed"]["high"] - CONFIG["agent"]["speed"]["low"]

        # Define the vertices of the polygon, (x, y) pairs
        next_waypoint = get_next_distinct_index(i, waypoints)
        vertices = [
            (outer_line[i][0], outer_line[i][1]),
            (outer_line[next_waypoint][0], outer_line[next_waypoint][1]),
            (inner_line[next_waypoint][0], inner_line[next_waypoint][1]),
            (inner_line[i][0], inner_line[i][1]),
        ]

        color = "red"
        if norm_difficulty < 0.5:
            color = "green"
        plot_difficulty = abs((norm_difficulty * 2) - 1)
        axs[0].add_patch(
            Polygon(
                vertices,
                closed=True,
                color=color,
                alpha=plot_difficulty,
                linewidth=0,
            )
        )
        color = "red"
        if throttle > 0.5:
            color = "green"
        plot_throttle = abs((throttle * 2) - 1)
        axs[1].add_patch(
            Polygon(
                vertices,
                closed=True,
                color=color,
                alpha=plot_throttle,
                linewidth=0,
            )
        )
        # axs[1].cla()
        # axs[1].grid(True)
        # axs[1, 1].plot(projected_steps, step_reward_plot, linestyle="-", color="black")
        # axs[1, 1].plot(projected_steps, aggregated_reward, linestyle="-", color="red")

    axs[0].set_title(
        f"Normalized Difficulty (look_ahead={CONFIG['difficulty']['look-ahead']})"
    )
    axs[1].set_title(f"Throttle")
    # axs[1, 1].set_title(f"Aggregated Step Reward")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    args = argparser.parse_args()

    main(args)
