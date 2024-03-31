import argparse
import requests
import numpy as np
from io import BytesIO
from custom_files.reward_function import get_direction_change, get_waypoint_difficulty
import wandb
import libcst as cst
from black import FileMode, format_str
import json

# Cumulative max is ~150@312 steps in our case
step_reward = {
    "ymax": 0.625,
    "ymin": 0.0,
    "k": -0.015,
    "x0": 400,
}

difficulty_weighting = {
    "ymax": 1.0,
    "ymin": 0.0,
    "k": 30,
    "x0": 0.50,
}

aggregated_factor = 0.5

class VariableTransformer(cst.CSTTransformer):
    def __init__(self, variable_name, new_value):
        self.variable_name = variable_name
        self.new_value = new_value

    def leave_Assign(self, original_node, updated_node):
        for target in original_node.targets:
            if (
                isinstance(target.target, cst.Name)
                and target.target.value == self.variable_name
            ):
                new_value_node = cst.parse_expression(self.new_value)
                return updated_node.with_changes(value=new_value_node)
        return updated_node


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


def main(args):

    if not args.debug:
        wandb.init(entity="iamjdoc", project="dr-reborn", job_type="update_reward")

    # Download the track
    npy_data, waypoint_count = download(args.track)
    waypoints = list([tuple(sublist[0:2]) for sublist in npy_data])
    inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
    outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])

    changes = []
    difficulties = []
    for i in range(waypoint_count):
        change, difficulty = get_waypoint_difficulty(
            i, waypoints, look_ahead=args.look_ahead
        )
        changes.append(get_direction_change(i, waypoints))
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
    reward_config = {
        "track": args.track,
        "waypoint_count": len(waypoints),
        "difficulty": {
            "look-ahead": args.look_ahead,
            "max": max(difficulties),
            "min": min(difficulties),
            "weighting": difficulty_weighting,
        },
        "histogram": {
            "counts": counts.tolist(),
            "weights": factors,
            "edges": bin_edges.tolist(),
        },
        "step_reward": step_reward,
        "heading": {"delay": args.delay, "offset": args.offset},
        "aggregated_factor": aggregated_factor,
    }
    # Open the JSON file for reading
    with open("custom_files/model_metadata.json", "r") as json_file:
        # Load the JSON data into a Python dictionary
        model_dict = json.load(json_file)
        reward_config["agent"] = model_dict["action_space"]

    # Replace in reward_function.py
    filename = "custom_files/reward_function.py"
    variable_name = "CONFIG"
    new_value = reward_config

    # Read the code from the file
    with open(filename, "r") as file:
        source_code = file.read()

    # Parse the source code into a CST
    tree = cst.parse_module(source_code)

    # Transform the CST
    transformer = VariableTransformer(variable_name, str(new_value))
    modified_tree = tree.visit(transformer)

    # Format the code using black
    formatted_code = format_str(modified_tree.code, mode=FileMode())

    # Write the formatted code back to the file
    with open(filename, "w") as file:
        file.write(formatted_code)

    if not args.debug:
        # Log reward_function.py as an artifact
        artifact = wandb.Artifact("config", type="inputs")
        artifact.add_dir("custom_files")
        wandb.log_artifact(artifact)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--track",
        help="the track used to verify the reward function",
        default="caecer_gp",
        required=False,
    )
    argparser.add_argument(
        "--bin-count",
        help="the number of bins to consider for learning importance histogram",
        default=4,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--look-ahead",
        help="the number of waypoints used to calculate look-ahead metrics",
        default=4,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--delay",
        help="the number of waypoints used to delay heading calculation",
        default=4,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--offset",
        help="the change factor used to calculate target heading",
        default=1.0,
        required=False,
        type=float,
    )
    argparser.add_argument(
        "--debug",
        action="store_true",
        help="Debug only (do not log to W&B)",
    )

    args = argparser.parse_args()

    main(args)
