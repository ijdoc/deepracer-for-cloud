import docker
import threading
import os
import wandb
import boto3
import json
from datetime import datetime
import time
import numpy as np
import subprocess
import argparse

# FIXME: Define from command line arguments in parent script
os.environ["WANDB_RUN_GROUP"] = "2402"
GLOBAL_MIN_STEPS = 192.0

# Create ArgumentParser
parser = argparse.ArgumentParser(description="Log testing metrics")

# Define arguments
parser.add_argument(
    "--pretrained", action="store_true", help="This is a pretrained model"
)
parser.add_argument(
    "--debug", action="store_true", help="Log to console instead of W&B"
)

# Parse the arguments
args = parser.parse_args()

DEBUG = args.debug
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
if DEBUG:
    print(f"{datetime.now()} Script path: {SCRIPT_PATH}")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["DR_LOCAL_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["DR_LOCAL_SECRET_ACCESS_KEY"],
    endpoint_url="http://localhost:9000",
)

# Configure project path
os.environ["WANDB_ENTITY"] = "iamjdoc"
os.environ["WANDB_PROJECT"] = "dr-reborn"

# Don't litter the console
os.environ["WANDB_SILENT"] = "true"

# Make sure it is possible to resume & auto-create runs
os.environ["WANDB_RESUME"] = "allow"


def reset_iter_metrics():
    return {
        "test": {"reward": None, "steps": [], "progress": [], "speed": []},
        "train": {"reward": [], "steps": [], "progress": [], "speed": []},
        "learn": {"loss": [], "KL_div": [], "entropy": []},
    }


def reset_tables():
    return {
        "test": wandb.Table(
            columns=[
                "step",
                "waypoint",
                "progress",
                "step_progress",
                "difficulty",
                "reward",
                "action",
            ]
        ),
        "train": wandb.Table(
            columns=[
                "step",
                "waypoint",
                "progress",
                "step_progress",
                "difficulty",
                "reward",
                "action",
            ]
        ),
    }


step_metrics = {
    "test": {"reward": None, "steps": None, "progress": None, "speed": None},
    "train": {"reward": None, "steps": None, "progress": None, "speed": None},
    "learn": {"loss": None, "KL_div": None, "entropy": None},
}
iter_metrics = reset_iter_metrics()
best_metrics = {"reward": -1.0, "progress": 0.0, "speed": 0.0, "steps": 100000.0}
is_testing = False
train_metrics = {"speed": [], "progress": None, "steps": None}
test_metrics = {"speed": [], "progress": None, "steps": None}
last_episode = 0


def update_run_env(name, checkpoint):
    world_name = ""
    # Open the file in read and write mode
    file_path = "./run.env"
    with open(file_path, "r") as file:
        lines = file.readlines()
    # Modify the content in memory
    new_lines = []
    for line in lines:
        if line.startswith("DR_WORLD_NAME="):
            world_name = line.split("=")[1]
        if line.startswith("DR_UPLOAD_S3_PREFIX="):
            new_lines.append(f"DR_UPLOAD_S3_PREFIX={name}-{checkpoint}\n")
        else:
            new_lines.append(line)
    # Write the modified content back to the file
    with open(file_path, "w") as file:
        file.writelines(new_lines)
    return world_name


# Open the JSON file for reading
with open("./custom_files/hyperparameters.json", "r") as json_file:
    # Load the JSON data into a Python dictionary
    config_dict = json.load(json_file)

# Start training job
if not DEBUG:
    if args.pretrained:
        wandb.init(config=config_dict, job_type="retrain")
    else:
        wandb.init(config=config_dict, job_type="train")
    # Log input files
    config_files = wandb.Artifact(name="config", type="inputs")
    env_files = wandb.Artifact(name="env", type="inputs")
    config_files.add_dir("./custom_files")
    env_files.add_file("./run.env")
    env_files.add_file("./system.env")
    wandb.use_artifact(config_files)
    wandb.use_artifact(env_files)
    tables = reset_tables()


def get_float(string):
    try:
        x = float(string)
        return x
    except ValueError:
        return None


def process_line(line):
    # Process training episodes and policy training
    global iter_metrics
    global step_metrics
    global is_testing
    global best_metrics
    global last_episode
    global tables

    timestamp = datetime.now()
    if "MY_TRACE_LOG" in line:
        # f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{speed},{difficulty},{reward},{is_finished}"
        parts = line.split("MY_TRACE_LOG:")[1].split("\t")[0].split("\n")[0].split(",")
        steps = int(float(parts[0]))
        waypoint = int(float(parts[1]))
        progress = float(parts[2])
        speed = float(parts[3])
        difficulty = float(parts[4])
        reward = float(parts[5])
        is_finished = int(parts[6])
        action = int(parts[7])
        if is_testing:
            if not DEBUG:
                tables["test"].add_data(
                    steps,
                    waypoint,
                    progress,
                    speed,
                    difficulty,
                    reward,
                    action,
                )
            test_metrics["speed"].append(speed)
            if is_finished == 1:
                iter_metrics["test"]["steps"].append(steps)
                iter_metrics["test"]["progress"].append(progress)
                iter_metrics["test"]["speed"].append(np.mean(test_metrics["speed"]))
                test_metrics["speed"] = []
        else:
            if not DEBUG:
                tables["train"].add_data(
                    steps,
                    waypoint,
                    progress,
                    speed,
                    difficulty,
                    reward,
                    action,
                )
            train_metrics["speed"].append(speed)
            if is_finished == 1:
                iter_metrics["train"]["steps"].append(steps)
                iter_metrics["train"]["progress"].append(progress)
                iter_metrics["train"]["speed"].append(np.mean(train_metrics["speed"]))
                train_metrics["speed"] = []
        if DEBUG:
            print(f"{timestamp} {line}")

    elif "Training>" in line and "[SAGE]" in line:
        # Capture training episode metrics
        metrics = line.split(",")
        last_episode = int(metrics[2].split("=")[1])
        reward = float(metrics[3].split("=")[1])
        iter_metrics["train"]["reward"].append(reward)
    elif "Policy training>" in line:
        metrics = line.split(",")
        # epoch = int(metrics[3].split("=")[1])
        loss = float(metrics[0].split("=")[1])
        divergence = float(metrics[1].split("=")[1])
        entropy = float(metrics[2].split("=")[1])
        metrics = {
            "learn/loss": loss,
            "learn/KL_div": divergence,
            "learn/entropy": entropy,
        }
        iter_metrics["learn"]["loss"].append(loss)
        iter_metrics["learn"]["KL_div"].append(divergence)
        iter_metrics["learn"]["entropy"].append(entropy)
    elif "[BestModelSelection] Evaluation episode reward mean:" in line:
        test_reward = get_float(line.split(":")[1].strip())
        if not test_reward is None:
            checkpoint = round((last_episode / 10) - 1)
            # Calculate means and log everything here!!
            step_metrics["test"]["reward"] = float(test_reward)
            step_metrics["test"]["speed"] = np.mean(iter_metrics["test"]["speed"])
            step_metrics["test"]["progress"] = np.mean(iter_metrics["test"]["progress"])
            # Projected steps to completion
            step_metrics["test"]["steps"] = (
                100.0 * np.mean(iter_metrics["test"]["steps"])
            ) / step_metrics["test"]["progress"]
            step_metrics["train"]["reward"] = np.mean(iter_metrics["train"]["reward"])
            step_metrics["train"]["progress"] = np.mean(
                iter_metrics["train"]["progress"]
            )
            # Projected steps to completion
            step_metrics["train"]["steps"] = (
                100.00 * np.mean(iter_metrics["train"]["steps"])
            ) / step_metrics["train"]["progress"]
            step_metrics["train"]["speed"] = np.mean(iter_metrics["train"]["speed"])
            step_metrics["learn"]["loss"] = np.mean(iter_metrics["learn"]["loss"])
            step_metrics["learn"]["KL_div"] = np.mean(iter_metrics["learn"]["KL_div"])
            step_metrics["learn"]["entropy"] = np.mean(iter_metrics["learn"]["entropy"])
            # Reset immediately to free up while things continue logging
            iter_metrics = reset_iter_metrics()
            # Update best metrics for summary
            if (
                step_metrics["test"]["progress"] >= best_metrics["progress"]
                and step_metrics["test"]["steps"] < best_metrics["steps"]
            ):
                best_metrics["reward"] = step_metrics["test"]["reward"]
                best_metrics["speed"] = step_metrics["test"]["speed"]
                best_metrics["steps"] = step_metrics["test"]["steps"]
                best_metrics["progress"] = step_metrics["test"]["progress"]
                print(
                    f"{timestamp} model {checkpoint}: {step_metrics['test']['reward']:0.2f}, {step_metrics['test']['progress']:0.2f}%, {step_metrics['test']['steps']:0.2f} steps (improved)"
                )
                if (
                    not DEBUG
                    and best_metrics["progress"] >= 100.0
                    and step_metrics["test"]["steps"] < GLOBAL_MIN_STEPS
                ):
                    print(
                        f"{timestamp} 🚀 Uploading full progress checkpoint {checkpoint} expecting {best_metrics['steps']:0.2f} steps)"
                    )
                    wandb.config["world_name"] = update_run_env(
                        wandb.run.name, checkpoint
                    ).replace("\n", "")
                    # FIXME: Get the model reference and log it to W&B
                    subprocess.run(f"./upload.sh", shell=True)
            else:
                print(
                    f"{timestamp} model {checkpoint}: {step_metrics['test']['reward']:0.2f}, {step_metrics['test']['progress']:0.2f}%, {step_metrics['test']['steps']:0.2f} steps"
                )
            if DEBUG:
                print(f"{timestamp} {step_metrics}")
            else:
                wandb.log(
                    {
                        "train/reward": step_metrics["train"]["reward"],
                        "train/steps": step_metrics["train"]["steps"],
                        "train/progress": step_metrics["train"]["progress"],
                        "train/speed": step_metrics["train"]["speed"],
                        "learn/loss": step_metrics["learn"]["loss"],
                        "learn/KL_div": step_metrics["learn"]["KL_div"],
                        "learn/entropy": step_metrics["learn"]["entropy"],
                        "test/reward": step_metrics["test"]["reward"],
                        "test/speed": step_metrics["test"]["speed"],
                        "test/steps": step_metrics["test"]["steps"],
                        "test/progress": step_metrics["test"]["progress"],
                        f"train_trace": tables["train"],
                        f"test_trace": tables["test"],
                    }
                )
                # Update test metrics summary
                wandb.run.summary["test/reward"] = best_metrics["reward"]
                wandb.run.summary["test/speed"] = best_metrics["speed"]
                wandb.run.summary["test/steps"] = best_metrics["steps"]
                wandb.run.summary["test/progress"] = best_metrics["progress"]
                # Reset tables
                tables = reset_tables()
        else:
            # Resetting inside else for completion (need this pattern to ensure proper reset)
            iter_metrics = reset_iter_metrics()
        is_testing = False
    elif "Starting evaluation phase" in line:
        is_testing = True
    else:
        if DEBUG:
            print(f"{timestamp} {line}")


def aggregate_logs(container, label):
    buffer = ""
    logs = container.logs(stream=True, follow=True)
    for chunk in logs:
        # Process each line of the log here
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if line.strip() != "":
                process_line(f"{label} {line.strip()}")


# Get containers
client = docker.from_env()
containers = client.containers.list()

for container in containers:
    tags = container.image.attrs["RepoTags"]
    if len(tags) > 0:
        if "sagemaker" in tags[0]:
            sagemaker = container
        if "robomaker" in tags[0]:
            robomaker = container

# Create a new thread
sage_thread = threading.Thread(target=aggregate_logs, args=(robomaker, "[ROBO]"))

# Start the thread
sage_thread.start()

# Wait for the thread to finish (if needed)
# sage_thread.join()

aggregate_logs(sagemaker, "[SAGE]")

if not DEBUG:
    print(f"{datetime.now()} Finishing...")
    wandb.finish()
