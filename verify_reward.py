import argparse
import requests
import numpy as np
from io import BytesIO
from custom_files import reward_function
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import wandb

# Cumulative max is ~150@312 steps in our case
step_progress = {
    "ymax": 0.625,
    "ymin": 0.0,
    "k": -0.015,
    "x0": 400,
}


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
        axs.text(xi, yi, str(i), color="black", fontsize=6, ha="center", va="center")


def main(args):

    projected_steps = [i for i in range(300, 1001, 10)]
    step_reward = [
        reward_function.sigmoid(
            i,
            k=step_progress["k"],
            x0=step_progress["x0"],
            ymin=step_progress["ymin"],
            ymax=step_progress["ymax"],
        )
        for i in projected_steps
    ]
    aggregated_reward = [
        projected_steps[i] * step_reward[i] for i in range(len(projected_steps))
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
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
        changes.append(reward_function.get_direction_change(i, waypoints))
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
        "difficulty": {
            "look-ahead": args.look_ahead,
            "max": max(difficulties),
            "min": min(difficulties),
        },
        "histogram": {
            "counts": counts.tolist(),
            "weights": factors,
            "edges": bin_edges.tolist(),
        },
        "step_progress": step_progress,
    }
    print(reward_config)

    columns = [
        "waypoint",
        "direction",
        "dir_change",
        "difficulty",
        "importance",
    ]

    # Start the plots
    for i in range(2):
        for j in range(2):
            plot_line(inner_line, axs[i, j])
            plot_line(outer_line, axs[i, j])
            plot_index(inner_line, axs[i, j])
            plot_index(outer_line, axs[i, j])
            axs[i, j].grid(True)

    # Second pass to plot verifications
    if not args.debug:
        factor_table = wandb.Table(columns=columns)
    for i in range(waypoint_count):
        direction = reward_function.get_direction(i, waypoints)
        dir_change, difficulty = reward_function.get_waypoint_difficulty(
            i,
            waypoints,
            look_ahead=reward_config["difficulty"]["look-ahead"],
            max_val=reward_config["difficulty"]["max"],
            min_val=reward_config["difficulty"]["min"],
        )
        importance = reward_function.get_waypoint_importance(
            reward_function.get_direction_change(i, waypoints),
            reward_config["histogram"],
        )

        row = [i]
        row.append(direction)
        row.append(dir_change)
        row.append(difficulty)
        row.append(importance)
        if not args.debug:
            factor_table.add_data(*row)

        # Define the vertices of the polygon, (x, y) pairs
        next_waypoint = reward_function.get_next_distinct_index(i, waypoints)
        vertices = [
            (outer_line[i][0], outer_line[i][1]),
            (outer_line[next_waypoint][0], outer_line[next_waypoint][1]),
            (inner_line[next_waypoint][0], inner_line[next_waypoint][1]),
            (inner_line[i][0], inner_line[i][1]),
        ]

        plot_difficulty = 2.0 * (difficulty - 0.5)
        color = "red"
        if plot_difficulty < 0:
            color = "blue"
        axs[0, 0].add_patch(
            Polygon(
                vertices,
                closed=True,
                color=color,
                alpha=abs(plot_difficulty),
                linewidth=0,
            )
        )
        axs[0, 1].add_patch(
            Polygon(
                vertices,
                closed=True,
                color=color,
                alpha=importance,
                linewidth=0,
            )
        )
        axs[1, 1].cla()
        axs[1, 1].grid(True)
        axs[1, 1].plot(projected_steps, step_reward, linestyle="-", color="black")
        axs[1, 1].plot(projected_steps, aggregated_reward, linestyle="-", color="red")

        length = 0.35
        heading = reward_function.get_target_heading(
            i,
            waypoints,
            delay=5,
            look_ahead=reward_config["difficulty"]["look-ahead"],
            min_val=reward_config["difficulty"]["min"],
            max_val=reward_config["difficulty"]["max"],
        )
        xstart = waypoints[i][0]
        ystart = waypoints[i][1]
        x1 = xstart + (length * math.cos(heading + (math.pi / 6.0)))
        y1 = ystart + (length * math.sin(heading + (math.pi / 6.0)))
        x2 = xstart + (length * math.cos(heading - (math.pi / 6.0)))
        y2 = ystart + (length * math.sin(heading - (math.pi / 6.0)))
        xend = xstart + (length * math.cos(direction))
        yend = ystart + (length * math.sin(direction))
        xapex = xstart + (length * difficulty * 3.0 * math.cos(heading))
        yapex = ystart + (length * difficulty * 3.0 * math.sin(heading))
        vertices = [
            (xstart, ystart),
            (x1, y1),
            (x2, y2),
        ]

        # axs[1, 0].add_patch(
        #     Polygon(
        #         vertices,
        #         closed=True,
        #         color=color,
        #         alpha=difficulty,
        #         linewidth=0,
        #     )
        # )
        axs[1, 0].arrow(
            xstart,
            ystart,
            xend - xstart,
            yend - ystart,
            head_width=0.025,
            head_length=0.025,
            fc="gray",
            ec="gray",
            linestyle="-",
            color="gray",
            width=0.001,
        )
        axs[1, 0].arrow(
            xstart,
            ystart,
            xapex - xstart,
            yapex - ystart,
            head_width=0.025,
            head_length=0.025,
            fc="green",
            ec="green",
            linestyle="-",
            color="green",
            width=difficulty / 100.0,
            alpha=difficulty,
        )

    axs[0, 0].set_title(
        f"Normalized Difficulty (look_ahead={reward_config['difficulty']['look-ahead']})"
    )
    axs[0, 1].set_title(f"Importance ({args.bin_count} bins)")
    axs[1, 0].set_title(
        f"Target Heading (look_ahead={reward_config['difficulty']['look-ahead']})"
    )
    axs[1, 1].set_title(f"Aggregated Step Reward")

    if args.debug:
        plt.tight_layout()
        plt.show()
    else:
        # TODO: Plot and log the step progress curve
        plt.savefig("track.png")
        run.log({"factor_table": factor_table, "track": wandb.Image("track.png")})
        # TODO: Automatically update reward_function.py
        # TODO: Log reward_function.py as an artifact
        # artifact = wandb.Artifact(f"{args.track}", type="reward_artifacts")
        # artifact.add_file("custom_files/reward_function.py")
        # run.log_artifact(artifact).wait()


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
