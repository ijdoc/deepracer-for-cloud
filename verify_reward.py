import argparse
import requests
import numpy as np
from io import BytesIO
from custom_files import reward_function
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import wandb


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

    print(f"Downloaded {track} with {waypoint_count} waypoints.")
    return npy_data, waypoint_count


def plot_numbered_line(line):
    x = [point[0] for point in line]
    y = [point[1] for point in line]
    plt.plot(x, y, linestyle="-", color="red")
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), color="black", fontsize=8, ha="center", va="center")


def main(debug=False):
    if not debug:
        run = wandb.init(
            entity="iamjdoc", project="dr-reborn", job_type="verify_reward"
        )

    # Download the track
    track = reward_function.TRACK["name"]
    npy_data, waypoint_count = download(track)
    waypoints = list([tuple(sublist[0:2]) for sublist in npy_data])
    inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
    outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])

    # Obtain track histogram
    changes = []
    for i in range(waypoint_count):
        changes.append(reward_function.get_direction_change(i, waypoints))
    counts, bin_edges = np.histogram(
        changes,
        bins=reward_function.TRACK["importance"]["histogram"]["bin_count"],
    )
    print("Bin counts: ", counts)
    factors = sum(counts) / counts
    factors = factors / min(factors)
    f_min = min(factors)
    f_max = max(factors)
    factors = [round((num - f_min) / (f_max - f_min), 4) for num in factors.tolist()]
    abs_dir_change = [abs(num) for num in changes]
    # TODO: Edit reward_function.py directly instead of printing & waiting for input
    print(
        {
            "name": track,
            "waypoint_count": waypoint_count,
            "importance": {
                "histogram": {
                    "bin_count": reward_function.TRACK["importance"]["histogram"][
                        "bin_count"
                    ],
                    "weights": factors,
                    "edges": bin_edges.tolist(),
                }
            },
            "difficulty": {"max": max(abs_dir_change), "min": min(abs_dir_change)},
        }
    )
    if debug:
        input("Copy this into reward_function.py and press Enter to continue...")

    columns = [
        "waypoint",
        "direction",
        "dir_change",
        "difficulty",
        "importance",
        "aggregate",
    ]
    if not debug:
        factor_table = wandb.Table(columns=columns)
    for i in range(waypoint_count):
        row = [i]
        row.append(reward_function.get_direction(i, waypoints))
        dir_change = reward_function.get_direction_change(i, waypoints)
        difficulty = reward_function.get_waypoint_difficulty(
            i,
            waypoints,
        )
        importance = reward_function.get_waypoint_importance(i, waypoints)
        # location = i / (waypoint_count - 1.0)
        # aggregate = (difficulty[i] + importance + location) / 3.0
        importance_factor = 1.0
        aggregate = (difficulty + (importance_factor * importance)) / (
            1.0 + importance_factor
        )
        next_waypoint = reward_function.get_next_distinct_index(i, waypoints)

        row.append(dir_change)
        row.append(difficulty)
        row.append(importance)
        # row.append(location)
        row.append(aggregate)
        if not debug:
            factor_table.add_data(*row)

        # Define the vertices of the polygon, (x, y) pairs
        vertices = [
            (outer_line[i][0], outer_line[i][1]),
            (outer_line[next_waypoint][0], outer_line[next_waypoint][1]),
            (inner_line[next_waypoint][0], inner_line[next_waypoint][1]),
            (inner_line[i][0], inner_line[i][1]),
        ]

        # Create the polygon with the vertices, set the alpha for the fill color
        plt.gca().add_patch(
            Polygon(
                vertices,
                closed=True,
                color="red",
                # alpha=0.0,  # Alpha controls transparency
                alpha=aggregate,  # Alpha controls transparency
            )
        )

    # Start track plot
    plot_numbered_line(inner_line)
    plot_numbered_line(outer_line)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    if debug:
        plt.show()
    else:
        # TODO: Plot and log the step progress curve
        plt.savefig("track.png")
        run.log({"factor_table": factor_table, "track": wandb.Image("track.png")})
        artifact = wandb.Artifact(f"{track}", type="reward_artifacts")
        artifact.add_file("track.png")
        artifact.add_file("custom_files/reward_function.py")
        run.log_artifact(artifact).wait()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--debug",
        action="store_true",
        help="Debug only (do not log to W&B)",
    )

    args = argparser.parse_args()

    main(args.debug)
