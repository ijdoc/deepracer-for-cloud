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

# FIXME: Define from command line arguments in parent script?
os.environ["WANDB_RUN_GROUP"] = "2403"
GLOBAL_MIN_STEPS = 300.0

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

# Don't litter the console
os.environ["WANDB_SILENT"] = "true"


def reset_iter_metrics():
    return {
        "test": {"reward": [], "steps": [], "progress": []},
        "train": {"reward": [], "steps": [], "progress": []},
        "learn": {"loss": [], "KL_div": [], "entropy": []},
    }


def reset_tables():
    columns = [
        "episode",
        "step",
        "waypoint",
        "progress",
        "projected_steps",
        "step_reward",
        "importance",
        "rank",
        "difficulty",
        "factor",
        "reward",
    ]
    return {
        "test": wandb.Table(columns=columns),
        "train": wandb.Table(columns=columns),
    }


ckpt_metrics = {
    "test": {"reward": None, "steps": None, "progress": None},
    "train": {"reward": None, "steps": None, "progress": None},
    "learn": {"loss": None, "KL_div": None, "entropy": None},
}
iter_metrics = reset_iter_metrics()
best_metrics = {"reward": -1.0, "progress": 0.0, "steps": 100000.0, "checkpoint": -1}
is_testing = False
step_metrics = {
    "train": {"reward": []},
    "test": {"reward": []},
}
checkpoint = -1
episode = {"train": 1, "test": 1}


def update_run_env(name, checkpoint):
    # Open the file in read mode
    file_path = "./run.env"
    with open(file_path, "r") as file:
        lines = file.readlines()
    # Modify the content in memory
    new_lines = []
    for line in lines:
        if line.startswith("DR_UPLOAD_S3_PREFIX="):
            new_lines.append(f"DR_UPLOAD_S3_PREFIX={name}-{checkpoint}\n")
        else:
            new_lines.append(line)
    # Write the modified content back to the file
    with open(file_path, "w") as file:
        file.writelines(new_lines)


# Open the JSON file for reading
with open("./custom_files/hyperparameters.json", "r") as json_file:
    # Load the JSON data into a Python dictionary
    config_dict = json.load(json_file)

# Open the JSON file for reading
with open("./custom_files/model_metadata.json", "r") as json_file:
    # Load the JSON data into a Python dictionary
    model_dict = json.load(json_file)
    logged_dict = {}
    logged_dict["version"] = model_dict["version"]
    logged_dict["type"] = model_dict["action_space_type"]
    logged_dict["algo"] = model_dict["training_algorithm"]
    logged_dict["net"] = model_dict["neural_network"]
    if logged_dict["type"] == "continuous":
        logged_dict["steer_max"] = model_dict["action_space"]["steering_angle"]["high"]
        logged_dict["steer_min"] = model_dict["action_space"]["steering_angle"]["low"]
        logged_dict["speed_max"] = model_dict["action_space"]["speed"]["high"]
        logged_dict["speed_min"] = model_dict["action_space"]["speed"]["low"]
    else:
        logged_dict["action_count"] = len(model_dict["action_space"])
        steer_min = 1000.0
        steer_max = -1000.0
        speed_min = 1000.0
        speed_max = -1000.0
        for action in model_dict["action_space"]:
            steer_min = min(steer_min, action["steering_angle"])
            steer_max = max(steer_max, action["steering_angle"])
            speed_min = min(speed_min, action["speed"])
            speed_max = max(speed_max, action["speed"])
        logged_dict["steer_max"] = steer_max
        logged_dict["steer_min"] = steer_min
        logged_dict["speed_max"] = speed_max
        logged_dict["speed_min"] = speed_min
    config_dict["m"] = logged_dict


# Open env file for reading
with open("./run.env", "r") as run_file:
    # Open the file in read mode
    for line in run_file.readlines():
        if line.startswith("DR_WORLD_NAME="):
            config_dict["world_name"] = line.split("=")[1]
            break

# Start training job
if not DEBUG:
    if args.pretrained:
        wandb.init(
            config=config_dict,
            entity="iamjdoc",
            project="dr-reborn",
            job_type="retrain",
        )
    else:
        wandb.init(
            config=config_dict, entity="iamjdoc", project="dr-reborn", job_type="train"
        )
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
    global ckpt_metrics
    global is_testing
    global best_metrics
    global checkpoint
    global tables
    global step_metrics

    timestamp = datetime.now()
    if "MY_TRACE_LOG" in line:
        parts = line.split("MY_TRACE_LOG:")[1].split("\t")[0].split("\n")[0].split(",")
        steps = int(float(parts[0]))
        waypoint = int(float(parts[1]))
        progress = float(parts[2])
        projected_steps = float(parts[3])
        step_reward = float(parts[4])
        importance = float(parts[5])
        rank = float(parts[6])
        difficulty = float(parts[7])
        factor = float(parts[8])
        reward = float(parts[9])
        is_finished = int(parts[10])
        job = "train"
        if is_testing:
            job = "test"
        if not DEBUG:
            tables[job].add_data(
                episode[job],
                steps,
                waypoint,
                progress,
                projected_steps,
                step_reward,
                importance,
                rank,
                difficulty,
                factor,
                reward,
            )
        step_metrics[job]["reward"].append(reward)
        if is_finished == 1:
            reward = np.sum(step_metrics[job]["reward"])
            steps = 100.0 * steps / progress
            iter_metrics[job]["reward"].append(reward)
            iter_metrics[job]["steps"].append(steps)
            iter_metrics[job]["progress"].append(progress)
            step_metrics[job] = {"reward": []}
        if DEBUG:
            print(f"{timestamp} {line}")

    elif "Training>" in line and "[SAGE]" in line:
        episode["train"] += 1
        # print(f'Next train episode: {episode["train"]}')

    elif "Testing>" in line and "[ROBO]" in line:
        episode["test"] += 1
        # print(f'Next episode: {episode["test"]}')

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
            # Aggregate metrics and log everything!!
            checkpoint += 1
            for job in ["test", "train"]:
                ckpt_metrics[job]["reward"] = np.mean(iter_metrics[job]["reward"])
                ckpt_metrics[job]["progress"] = np.mean(iter_metrics[job]["progress"])
                ckpt_metrics[job]["steps"] = np.mean(iter_metrics[job]["steps"])
            # print(
            #     f'{timestamp} Same? {ckpt_metrics["test"]["reward"]:0.3f}, {float(test_reward):0.3f}'
            # )

            ckpt_metrics["learn"]["loss"] = np.mean(iter_metrics["learn"]["loss"])
            ckpt_metrics["learn"]["KL_div"] = np.mean(iter_metrics["learn"]["KL_div"])
            ckpt_metrics["learn"]["entropy"] = np.mean(iter_metrics["learn"]["entropy"])

            # Update best metrics & summary
            if ckpt_metrics["test"]["progress"] > best_metrics["progress"] or (
                ckpt_metrics["test"]["progress"] >= best_metrics["progress"]
                and ckpt_metrics["test"]["steps"] < best_metrics["steps"]
            ):
                best_metrics["reward"] = ckpt_metrics["test"]["reward"]
                best_metrics["steps"] = ckpt_metrics["test"]["steps"]
                best_metrics["progress"] = ckpt_metrics["test"]["progress"]
                best_metrics["checkpoint"] = checkpoint
                print(
                    f'{timestamp} ckpt {checkpoint}: {ckpt_metrics["test"]["reward"]:0.2f}, {ckpt_metrics["test"]["progress"]:0.2f}%, {ckpt_metrics["test"]["steps"]:0.2f} steps (improved)'
                )
                if (
                    not DEBUG
                    and best_metrics["progress"] >= 100.0
                    # and ckpt_metrics["test"]["steps"] <= GLOBAL_MIN_STEPS
                ):
                    print(
                        f'{timestamp} 🚀 Uploading full progress checkpoint {checkpoint} expecting {best_metrics["steps"]:0.2f} steps)'
                    )
                    update_run_env(wandb.run.name, checkpoint)
                    subprocess.run("./upload.sh", shell=True)
                    # subprocess.Popen(["./upload.sh"])  # Non-blocking!
                    # print(
                    #     f"TODO: Create model reference to s3://jdoc-one-deepracer-data-b5pi7cdvar/{wandb.run.name}-{checkpoint}/"
                    # )
                    wandb.log_model(
                        f"s3://jdoc-one-deepracer-data-b5pi7cdvar/{wandb.run.name}-{checkpoint}/"
                    )
            else:
                print(
                    f'{timestamp} ckpt {checkpoint}: {ckpt_metrics["test"]["reward"]:0.2f}, {ckpt_metrics["test"]["progress"]:0.2f}%, {ckpt_metrics["test"]["steps"]:0.2f} steps'
                )
            if DEBUG:
                print(f"{timestamp} {ckpt_metrics}")
            else:
                wandb.log(
                    {
                        "train/reward": ckpt_metrics["train"]["reward"],
                        "train/steps": ckpt_metrics["train"]["steps"],
                        "train/progress": ckpt_metrics["train"]["progress"],
                        "learn/loss": ckpt_metrics["learn"]["loss"],
                        "learn/KL_div": ckpt_metrics["learn"]["KL_div"],
                        "learn/entropy": ckpt_metrics["learn"]["entropy"],
                        "test/reward": ckpt_metrics["test"]["reward"],
                        "test/steps": ckpt_metrics["test"]["steps"],
                        "test/progress": ckpt_metrics["test"]["progress"],
                        "train_trace": tables["train"],
                        "test_trace": tables["test"],
                    }
                )
                # Update test metrics summary
                wandb.run.summary["test/reward"] = best_metrics["reward"]
                wandb.run.summary["test/steps"] = best_metrics["steps"]
                wandb.run.summary["test/progress"] = best_metrics["progress"]
                wandb.run.summary["best_checkpoint"] = best_metrics["checkpoint"]
        # Resetting tracker variables
        iter_metrics = reset_iter_metrics()
        tables = reset_tables()
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
