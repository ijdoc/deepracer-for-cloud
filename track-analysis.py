import argparse
import requests
import numpy as np
from io import BytesIO
from custom_files import reward_function
import matplotlib.pyplot as plt


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


def main(track):
    difficulty = {"min": 100.0, "max": -100.0}
    npy_data, waypoint_count = download(track)
    center_line = list([tuple(sublist[0:2]) for sublist in npy_data])
    inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
    outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])

    for i in range(waypoint_count):
        difficulty["ahead"] = 4
        j = i
        change = 0.0
        for _ in range(difficulty["ahead"]):
            change += reward_function.get_direction_change(j, center_line)
            j = reward_function.get_next_distinct_index(j, center_line)
        change = abs(change / difficulty["ahead"])
        if change < difficulty["min"]:
            difficulty["min"] = change
        if change > difficulty["max"]:
            difficulty["max"] = change
        print(f"{i}: {change:0.4f}")
        plt.plot(
            [inner_line[i][0], outer_line[i][0]],
            [inner_line[i][1], outer_line[i][1]],
            linestyle="-",
            color="black",
        )

    print(f"{track} direction change limits: {difficulty}")

    x = [point[0] for point in center_line]
    y = [point[1] for point in center_line]
    plt.plot(x, y, linestyle="-", color="white")
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), color="black", fontsize=8, ha="center", va="center")
    x = [point[0] for point in inner_line]
    y = [point[1] for point in inner_line]
    plt.plot(x, y, linestyle="-", color="red")
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), color="black", fontsize=8, ha="center", va="center")
    x = [point[0] for point in outer_line]
    y = [point[1] for point in outer_line]
    plt.plot(x, y, linestyle="-", color="red")
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), color="black", fontsize=8, ha="center", va="center")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("inner_line")
    plt.grid(True)
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
