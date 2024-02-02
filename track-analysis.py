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


def print_numbered_line(line):
    x = [point[0] for point in line]
    y = [point[1] for point in line]
    plt.plot(x, y, linestyle="-", color="red")
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), color="black", fontsize=8, ha="center", va="center")


def main(track):
    npy_data, waypoint_count = download(track)
    center_line = list([tuple(sublist[0:2]) for sublist in npy_data])
    inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
    outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])
    limits = {
        "ahead": 2,
        "heading": [0 for _ in range(len(center_line))],
    }

    print_numbered_line(inner_line)
    print_numbered_line(outer_line)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    for i in range(waypoint_count):
        # Plot cross-waypoint lines
        plt.plot(
            [inner_line[i][0], outer_line[i][0]],
            [inner_line[i][1], outer_line[i][1]],
            linestyle="-",
            color="black",
        )
        # Plot waypoint direction
        direction = reward_function.get_direction(i, center_line)
        x_start = center_line[i][0]
        y_start = center_line[i][1]
        x_end = x_start + 0.1 * math.cos(direction)
        y_end = y_start + 0.1 * math.sin(direction)
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
        limits["ahead"] = 2
        change = 0.0
        j = i
        for _ in range(limits["ahead"]):
            j = reward_function.get_next_distinct_index(j, center_line)
            change += reward_function.get_direction_change(j, center_line)
        # change = abs(change / limits["ahead"])

        # Plot look ahead direction
        limits["heading"][i] = math.degrees(direction + change)
        x_start = center_line[i][0]
        y_start = center_line[i][1]
        x_end = x_start + 0.1 * math.cos(direction + change)
        y_end = y_start + 0.1 * math.sin(direction + change)
        plt.arrow(
            x_start,
            y_start,
            x_end - x_start,
            y_end - y_start,
            head_width=0.025,
            head_length=0.025,
            fc="green",
            ec="green",
            linestyle="-",
            color="green",
            width=0.001,
        )

    # print(f"{track} direction change limits: {limits}")
    for i in range(len(limits["heading"])):
        print(i, limits["heading"][i])

    plt.show()


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
