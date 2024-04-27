import os
import argparse
import numpy as np
from custom_files.reward_function import get_direction_change, get_waypoint_difficulty
import libcst as cst
from black import FileMode, format_str
import json
from utils import reward_config_utils as rcu

# Cumulative max is ~141@161 steps in our case
step_reward = {
    "ymax": 1,
    "ymin": 0.0,
    "k": -0.05,
    "x0": 200,
}

difficulty_weighting = {
    "ymax": 1.0,
    "ymin": 0.0,
    "k": 30,
    "x0": 0.50,
}

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


def main(args):

    # Check if the track file exists
    if os.path.exists(f"{args.track}.npy"):
        npy_data = np.load(f"{args.track}.npy")
    else:
        # Download the track
        npy_data = rcu.download(args.track)

    waypoint_count = npy_data.shape[0]
    waypoints = list([tuple(sublist[0:2]) for sublist in npy_data])
    inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
    outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])

    changes = []
    difficulties = []
    for i in range(waypoint_count):
        change, _, difficulty = get_waypoint_difficulty(
            i, waypoints, skip_ahead=args.skip_ahead, look_ahead=args.look_ahead
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
        "reward_type": args.reward_type,
        "waypoint_count": len(waypoints),
        "difficulty": {
            "skip-ahead": args.skip_ahead,
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
    }

    # Open the agent file for reading
    with open("custom_files/model_metadata.json", "r") as json_file:
        # Load the JSON data into a Python dictionary
        model_dict = json.load(json_file)

    model_dict["action_space"]["speed"]["high"] = args.agent_speed_high
    model_dict["action_space"]["speed"]["low"] = args.agent_speed_low

    # Overwrite the agent speed values
    with open("custom_files/model_metadata.json", "w") as json_file:
        json.dump(model_dict, json_file, indent=4)

    # Update agent params in reward_config
    reward_config["agent"] = model_dict["action_space"]

    # Open the hyperparameters file for reading
    with open("custom_files/hyperparameters.json", "r") as json_file:
        # Load the JSON data into a Python dictionary
        params_dict = json.load(json_file)

    # Replace learning rate
    params_dict["lr"] = args.learning_rate

    # Overwrite hyperparameters
    with open("custom_files/hyperparameters.json", "w") as json_file:
        json.dump(params_dict, json_file, indent=4)

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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--track",
        help="the track used to verify the reward function",
        default="dubai_open_ccw",
        required=False,
    )
    argparser.add_argument(
        "--agent-speed-high",
        help="the maximum speed of the agent",
        required=True,
        type=float,
    )
    argparser.add_argument(
        "--agent-speed-low",
        help="the minimum speed of the agent",
        required=True,
        type=float,
    )
    argparser.add_argument(
        "--learning-rate",
        help="the learning rate of the agent",
        required=True,
        type=float,
    )
    argparser.add_argument(
        "--skip-ahead",
        help="the number of waypoints to skip when calculating upcoming track difficulty. 0 means start from the current waypoint.",
        default=1,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--look-ahead",
        help="the number of waypoints to consider when calculating upcoming track difficulty. 0 means consider the current waypoint only.",
        default=0,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--bin-count",
        help="the number of bins to consider for learning importance histogram",
        default=12,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--reward-type",
        help="the reward function to use",
        type=int,
        required=True,
    )

    args = argparser.parse_args()

    main(args)
