import argparse
import requests
import numpy as np
from io import BytesIO
from custom_files import reward_function
import matplotlib.pyplot as plt
import math


def download(track):
    # URL of the npy file
    npy_url = f"https://github.com/aws-deepracer-community/deepracer-race-data/raw/main/raw_data/tracks/npy/{track}.npy"

    # Download the npy file
    response = requests.get(npy_url)
    if response.status_code == 200:
        # Load the npy file and count the number of inner_line
        npy_data = np.load(BytesIO(response.content))
        waypoint_count = npy_data.shape[0]
    else:
        waypoint_count = "Error downloading npy file"

    print(f"Downloaded {track} with {waypoint_count} inner_line")
    return npy_data, waypoint_count


def plot_numbered_line(line):
    x = [point[0] for point in line]
    y = [point[1] for point in line]
    plt.plot(x, y, linestyle="-", color="red")
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), color="black", fontsize=8, ha="center", va="center")


def main(track):
    npy_data, waypoint_count = download(track)
    waypoints = list([tuple(sublist[0:2]) for sublist in npy_data])
    inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
    outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])
    limits = {
        "max_importance": -1000,
        "min_importance": 1000,
        "max_difficulty": -1000,
        "min_difficulty": 1000,
        "max_factor": -1000,
        "min_factor": 1000,
    }

    plot_numbered_line(inner_line)
    plot_numbered_line(outer_line)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    importance = [0 for _ in range(len(waypoints))]
    difficulty = [0 for _ in range(len(waypoints))]
    change = [0 for _ in range(len(waypoints))]
    for i in range(waypoint_count):
        # Plot cross-waypoint lines
        plt.plot(
            [inner_line[i][0], outer_line[i][0]],
            [inner_line[i][1], outer_line[i][1]],
            linestyle="-",
            color="gray",
        )
        # Plot waypoint direction
        direction = reward_function.get_direction(i, waypoints)
        x_start = waypoints[i][0]
        y_start = waypoints[i][1]
        length = 0.1
        x_end = x_start + length * math.cos(direction)
        y_end = y_start + length * math.sin(direction)
        plt.arrow(
            x_start,
            y_start,
            x_end - x_start,
            y_end - y_start,
            head_width=0.025,
            head_length=0.025,
            fc="blue",
            ec="blue",
            linestyle="-",
            color="blue",
            width=0.001,
        )
        # Calculate cumulative direction change
        # j = i
        # difficulty[i] = direction
        change[i] = reward_function.get_direction_change(i, waypoints)
        difficulty[i] = reward_function.get_waypoint_difficulty(
            i, waypoints, max_val=1.7394709392167267, factor=7.6
        )
        importance[i] = reward_function.get_waypoint_importance(i, waypoints)
        # for _ in range(limits["ahead"]):
        #     j = reward_function.get_next_distinct_index(j, waypoints)
        # difficulty[i] = abs(difficulty[i])
        # Calculate limits
        if difficulty[i] > limits["max_difficulty"]:
            limits["max_difficulty"] = difficulty[i]
        if difficulty[i] < limits["min_difficulty"]:
            limits["min_difficulty"] = difficulty[i]
        if importance[i] > limits["max_importance"]:
            limits["max_importance"] = importance[i]
        if importance[i] < limits["min_importance"]:
            limits["min_importance"] = importance[i]
        if importance[i] + difficulty[i] > limits["max_factor"]:
            limits["max_factor"] = importance[i] + difficulty[i]
        if importance[i] + difficulty[i] < limits["min_factor"]:
            limits["min_factor"] = importance[i] + difficulty[i]
        # x_start = waypoints[i][0]
        # y_start = waypoints[i][1]
        # x_end = x_start + length * math.cos(difficulty[i])
        # y_end = y_start + length * math.sin(difficulty[i])
        # plt.arrow(
        #     x_start,
        #     y_start,
        #     x_end - x_start,
        #     y_end - y_start,
        #     head_width=0.025,
        #     head_length=0.025,
        #     fc="green",
        #     ec="green",
        #     linestyle="-",
        #     color="green",
        #     width=0.001,
        # )
        print(
            i,
            f"{change[i]:0.02f}",
            f"{difficulty[i]:0.02f}",
            f"{importance[i]:0.02f}",
            f"{importance[i] + difficulty[i]:0.02f}",
        )
        # print(
        #     i, f"{change[i]:0.02f}", f"{difficulty[i]:0.02f}"
        # )
    print(limits)

    # Calculate bin counts and bin edges
    counts, bin_edges = np.histogram(change, bins="auto")
    factors = sum(counts) / counts
    factors = (factors / min(factors))
    print(f"Counts: \t{counts}")
    print(f"Factors:\t{factors}")
    print(f"Bins:   \t{bin_edges}")
    print({"weights": factors.tolist(), "edges": bin_edges.tolist()})
    # plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--track",
        required=True,
        help="The name of the track (from https://github.com/aws-deepracer-community/deepracer-race-data/blob/main/raw_data/tracks/README.md - ends in '.npy')",
        type=str,
    )
    args = argparser.parse_args()

    main(args.track)
