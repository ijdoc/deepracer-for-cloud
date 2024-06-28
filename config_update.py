import os
import argparse
import numpy as np
import libcst as cst
from black import FileMode, format_str
import json
from utils import reward_config_utils as rcu


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

    # Open env file for reading
    with open("./run.env", "r") as run_file:
        # Open the file in read mode
        for line in run_file.readlines():
            if line.startswith("DR_WORLD_NAME="):
                track_name = line.split("=")[1].strip()
                break

    # Check if the track file exists
    if os.path.exists(f"{track_name}.npy"):
        npy_data = np.load(f"{track_name}.npy")
    else:
        # Download the track
        npy_data = rcu.download(track_name)

    waypoint_count = npy_data.shape[0]
    waypoints = list([tuple(sublist[0:2]) for sublist in npy_data])
    inner_line = list([tuple(sublist[2:4]) for sublist in npy_data])
    outer_line = list([tuple(sublist[4:6]) for sublist in npy_data])

    reward_config = {
        "track": track_name,
        "waypoint_count": len(waypoints),
        "aggregate": args.aggregate,
        "throttle_factor_diff": args.throttle_factor_diff,
    }

    # Open the agent file for reading
    with open("custom_files/model_metadata.json", "r") as json_file:
        # Load the JSON data into a Python dictionary
        model_dict = json.load(json_file)

    if args.agent_speed_high == args.agent_speed_low:
        args.agent_speed_high += 1e-3
        args.agent_speed_low -= 1e-3
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
        "--look-ahead",
        help="the number of waypoints to consider when calculating upcoming track difficulty. 0 means consider the current waypoint only.",
        default=0,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--throttle-factor-diff",
        help="the throttle factor difference between high and low speeds",
        default=1.0,
        required=False,
        type=int,
    )
    argparser.add_argument(
        "--aggregate",
        help="the number of steps to aggregate for the mean reward",
        default=15,
        required=False,
        type=int,
    )

    args = argparser.parse_args()

    main(args)
