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


def plot_line(line, axs):
    x = [point[0] for point in line]
    y = [point[1] for point in line]
    axs.plot(x, y, linestyle="-", color="gray")


def plot_index(line, axs):
    x = [point[0] for point in line]
    y = [point[1] for point in line]
    for i, (xi, yi) in enumerate(zip(x, y)):
        axs.text(xi, yi, str(i), color="black", fontsize=8, ha="center", va="center")


def main(args):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    if not args.debug:
        run = wandb.init(
            entity="iamjdoc", project="dr-reborn", job_type="verify_reward"
        )

    # Download the track
    npy_data, waypoint_count = download(args.track)
    waypoints = list([tuple(sublist[0:2]) for sublist in npy_data])
    inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
    outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])

    # Fist pass to obtain baseline data
    changes = []
    difficulties = []
    for i in range(waypoint_count):
        change, difficulty = reward_function.get_waypoint_difficulty(
            i, waypoints, look_ahead=args.look_ahead
        )
        changes.append(change)
        difficulties.append(difficulty)

    # Obtain aggregate values/weights
    counts, bin_edges = np.histogram(
        changes,
        bins=args.bin_count,
    )
    factors = sum(counts) / counts
    factors = factors / min(factors)
    f_min = min(factors)
    f_max = max(factors)
    factors = [round((num - f_min) / (f_max - f_min), 4) for num in factors.tolist()]
    abs_dir_change = [abs(num) for num in changes]
    reward_config = {
        "histogram": {
            "bin_count": args.bin_count,
            "weights": factors,
            "edges": bin_edges.tolist(),
        },
        "difficulty": {"max": max(difficulties), "min": min(difficulties)},
    }
    print(f"Absolute Difficulty (look-ahead: {args.look_ahead})")
    print("\t- Max: ", reward_config["difficulty"]["max"])
    print("\t- Min: ", reward_config["difficulty"]["min"])

    columns = [
        "waypoint",
        "direction",
        "dir_change",
        "difficulty",
    ]

    # Start the plots
    for i in range(2):
        plot_line(inner_line, axs[i])
        plot_line(outer_line, axs[i])
        plot_index(inner_line, axs[i])
        plot_index(outer_line, axs[i])
        axs[i].grid(True)

    # Second pass to plot verifications
    if not args.debug:
        factor_table = wandb.Table(columns=columns)
    for i in range(waypoint_count):
        direction = reward_function.get_direction(i, waypoints)
        dir_change, difficulty = reward_function.get_waypoint_difficulty(
            i,
            waypoints,
            look_ahead=args.look_ahead,
            max_val=reward_config["difficulty"]["max"],
            min_val=reward_config["difficulty"]["min"],
        )
        next_waypoint = reward_function.get_next_distinct_index(i, waypoints)

        row = [i]
        row.append(direction)
        row.append(dir_change)
        row.append(difficulty)
        if not args.debug:
            factor_table.add_data(*row)

        # Define the vertices of the polygon, (x, y) pairs
        vertices = [
            (outer_line[i][0], outer_line[i][1]),
            (outer_line[next_waypoint][0], outer_line[next_waypoint][1]),
            (inner_line[next_waypoint][0], inner_line[next_waypoint][1]),
            (inner_line[i][0], inner_line[i][1]),
        ]

        color = "red"
        if dir_change < 0:
            color = "blue"
        axs[0].add_patch(
            Polygon(
                vertices,
                closed=True,
                color=color,
                # alpha=0.0,  # Alpha controls transparency
                alpha=difficulty,  # Alpha controls transparency
                linewidth=0,
            )
        )

        length = 0.35
        heading = reward_function.get_target_heading(i, waypoints)
        xstart = waypoints[i][0]
        ystart = waypoints[i][1]
        x1 = xstart + length * math.cos(heading + (math.pi / 6.0))
        y1 = ystart + length * math.sin(heading + (math.pi / 6.0))
        x2 = xstart + length * math.cos(heading - (math.pi / 6.0))
        y2 = ystart + length * math.sin(heading - (math.pi / 6.0))
        xend = xstart + length * math.cos(direction)
        yend = ystart + length * math.sin(direction)
        vertices = [
            (xstart, ystart),
            (x1, y1),
            (x2, y2),
        ]

        axs[1].add_patch(
            Polygon(
                vertices,
                closed=True,
                color=color,
                # alpha=0.0,  # Alpha controls transparency
                alpha=difficulty,  # Alpha controls transparency
                linewidth=0,
            )
        )
        axs[1].arrow(
            xstart,
            ystart,
            xend - xstart,
            yend - ystart,
            head_width=0.025,
            head_length=0.025,
            fc="black",
            ec="black",
            linestyle="-",
            color="black",
            width=0.001,
        )

    axs[0].set_title(f"Normalized Difficulty (look_ahead={args.look_ahead})")
    axs[1].set_title(f"Direction & Spread")

    if args.debug:
        plt.tight_layout()
        plt.show()
    else:
        # TODO: Plot and log the step progress curve
        plt.savefig("track.png")
        run.log({"factor_table": factor_table, "track": wandb.Image("track.png")})
        artifact = wandb.Artifact(f"{track}", type="reward_artifacts")
        artifact.add_file("custom_files/reward_function.py")
        run.log_artifact(artifact).wait()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--track",
        help="The track used to verify the reward function",
        default="caecer_gp",
        required=False,
    )
    argparser.add_argument(
        "--bin-count",
        help="The number of bins to consider for learning importance histogram",
        default=4,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--look-ahead",
        help="The number of waypoints to use to calculate look-ahead metrics",
        default=4,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--debug",
        action="store_true",
        help="Debug only (do not log to W&B)",
    )

    args = argparser.parse_args()

    main(args)
