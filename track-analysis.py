import argparse
import requests
import numpy as np
from io import BytesIO
from custom_files import reward_function


def download(track):
    # URL of the npy file
    npy_url = f"https://github.com/aws-deepracer-community/deepracer-race-data/raw/main/raw_data/tracks/npy/{track}.npy"

    # Download the npy file
    response = requests.get(npy_url)
    if response.status_code == 200:
        # Load the npy file and count the number of waypoints
        npy_data = np.load(BytesIO(response.content))
        waypoint_count = npy_data.shape[0]
    else:
        waypoint_count = "Error downloading npy file"

    print(f"Downloaded {track} with {waypoint_count} waypoints")
    return npy_data, waypoint_count

def main(track):
    direction_change = {"min": 100.0, "max": -100.0}
    npy_data, waypoint_count = download(track)
    waypoints = list([tuple(sublist[2:4]) for sublist in npy_data])
    print(waypoints)
    for i in range(waypoint_count):
        change = abs(reward_function.get_direction_change(i, waypoints))
        if change < direction_change["min"]:
            direction_change["min"] = change
        if change > direction_change["max"]:
            direction_change["max"] = change
        print(f"{i}: {change:0.4f}")
    print(f"{track} direction change limits: {direction_change}")


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
