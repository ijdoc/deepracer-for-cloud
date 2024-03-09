import argparse
import requests
import numpy as np
from io import BytesIO
from custom_files import reward_function
import matplotlib.pyplot as plt
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


def main(track):
    with wandb.init(
        entity="iamjdoc", project="dr-reborn", job_type="verify_reward"
    ) as run:
        npy_data, waypoint_count = download(track)
        waypoints = list([tuple(sublist[0:2]) for sublist in npy_data])
        inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
        outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])

        # Start track plot
        plot_numbered_line(inner_line)
        plot_numbered_line(outer_line)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        change = [0 for _ in range(len(waypoints))]
        difficulty = [0 for _ in range(len(waypoints))]
        columns = ["waypoint", "direction", "dir_change", "difficulty"]
        for i in reward_function.TRACK["trial_starts"]:
            columns.append(f"rank_{i}")
        factor_table = wandb.Table(columns=columns)
        for i in range(waypoint_count):
            # Plot cross-waypoint lines
            plt.plot(
                [inner_line[i][0], outer_line[i][0]],
                [inner_line[i][1], outer_line[i][1]],
                linestyle="-",
                color="gray",
            )

            row = [i]
            row.append(reward_function.get_direction(i, waypoints))
            change[i] = reward_function.get_direction_change(i, waypoints)
            row.append(change[i])
            max_importance_weight = max(
                reward_function.TRACK["importance"]["histogram"]["weights"]
            )
            real_difficulty, normalized_difficulty = (
                reward_function.get_waypoint_difficulty(
                    i,
                    waypoints,
                    reward_function.TRACK["difficulty"]["max"],
                    max_importance_weight,
                )
            )
            difficulty[i] = real_difficulty
            row.append(normalized_difficulty)
            for j in reward_function.TRACK["trial_starts"]:
                row.append(
                    reward_function.get_waypoint_batch_rank(i, j, max_importance_weight)
                )
            factor_table.add_data(*row)

        # Calculate bin counts and bin edges
        counts, bin_edges = np.histogram(change, bins="auto")
        factors = sum(counts) / counts
        factors = factors / min(factors)
        factors = [round(num, 3) for num in factors.tolist()]
        print(
            {
                "name": track,
                "importance": {
                    "histogram": {"weights": factors, "edges": bin_edges.tolist()}
                },
                "difficulty": {"max": max(difficulty)},
                "waypoint_count": waypoint_count,
            }
        )
        run.log({"factor_table": factor_table})
        # TODO: Draw importance + difficulty weights on the track plot
        # TODO: Plot and log the step progress curve
        run.log({"track": plt})


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
