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
        "max": -1000,
        "min": 1000,
    }

    print_numbered_line(inner_line)
    print_numbered_line(outer_line)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    change_line = [0 for _ in range(len(center_line))]
    print("[", end="")
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
        j = i
        change_line[i] = direction
        for _ in range(limits["ahead"]):
            j = reward_function.get_next_distinct_index(j, center_line)
            change_line[i] += reward_function.get_direction_change(j, center_line)
        # change_line[i] = abs(change_line[i])
        # Calculate limits
        # if change_line[i] > limits["max"]:
        #     limits["max"] = change_line[i]
        # if change_line[i] < limits["min"]:
        #     limits["min"] = change_line[i]
        x_start = center_line[i][0]
        y_start = center_line[i][1]
        x_end = x_start + length * math.cos(change_line[i])
        y_end = y_start + length * math.sin(change_line[i])
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
        # print(
        #     i,
        #     round(math.degrees(math.atan2(math.sin(direction), math.cos(direction)))),
        #     round(
        #         math.degrees(
        #             math.atan2(math.sin(change_line[i]), math.cos(change_line[i]))
        #         )
        #     ),
        # )
        print(
            round(
                math.degrees(
                    math.atan2(math.sin(change_line[i]), math.cos(change_line[i]))
                )
            ),
            end=",",
        )
    print("]")

    # Plot expected throttle values
    # for i in range(waypoint_count):
    #     direction = reward_function.get_direction(i, center_line)
    # normalized = (change_line[i] - limits["min"]) / (limits["max"] - limits["min"])
    # length = (1.0 - normalized) * 1.0
    # print(i, 1.0 - normalized)
    # print(limits)

    plt.show()

# Example usage
x_values = range(-180, 181, 1)  # Generate a range of values from -180 to 180
center = -170  # Center of the bell curve
width = 45  # Width of the bell curve
bell_curve_values = [
    reward_function.wrapped_bell_curve(x, center, width) for x in x_values
]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x_values, bell_curve_values, label="Bell Curve", color="blue")
plt.title("Wrapped Bell Curve")
plt.xlabel("Angle (degrees)")
plt.ylabel("Value")
plt.grid(True)
plt.xticks(range(-180, 181, 30))  # Setting x-axis ticks for better readability
plt.legend()
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
