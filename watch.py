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
from custom_files.reward_function import CONFIG

# FIXME: Define from command line arguments in parent script?
os.environ["WANDB_RUN_GROUP"] = "2404"
GLOBAL_MIN_STEPS = 320.0
MIN_ENTROPY = -1.0

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


# def reset_tables():
#     columns = [
#         "episode",
#         "step",
#         "waypoint",
#         "progress",
#         "step_reward",
#         "reward",
#     ]
#     return {
#         "test": wandb.Table(columns=columns),
#         "train": wandb.Table(columns=columns),
#     }


ckpt_metrics = {
    "test": {"reward": None, "steps": None, "progress": None, "combo": None},
    "train": {"reward": None, "steps": None, "progress": None},
    "learn": {"loss": None, "KL_div": None, "entropy": None},
}
iter_metrics = reset_iter_metrics()
best_metrics = {
    "reward": -1.0,
    "progress": 0.0,
    "steps": 100000.0,
    "checkpoint": -1,
    "entropy": 100.0,
    "combo": 0.0,
}
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
    logged_dict["space"] = model_dict["action_space"]
    config_dict["m"] = logged_dict

logged_dict = {}
logged_dict["step"] = CONFIG["step_reward"]
logged_dict["difficulty_weighting"] = CONFIG["difficulty"]["weighting"]
config_dict["r"] = logged_dict
config_dict["reward-type"] = CONFIG["reward_type"]
config_dict["bin-count"] = len(CONFIG["histogram"]["counts"])
config_dict["skip-ahead"] = CONFIG["difficulty"]["skip-ahead"]
config_dict["look-ahead"] = CONFIG["difficulty"]["look-ahead"]


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
            config=config_dict,
            entity="iamjdoc",
            project="dr-reborn",
            job_type="train",
            resume="allow",  # Needed to assume the launch-provided id
        )
        # Log input configurations
        params_art = wandb.Artifact("hyperparams", type="inputs")
        agent_art = wandb.Artifact("agent", type="inputs")
        reward_art = wandb.Artifact("reward", type="inputs")
        sys_art = wandb.Artifact("env_sys", type="inputs")
        run_art = wandb.Artifact("env_run", type="inputs")
        params_art.add_file("custom_files/hyperparameters.json")
        agent_art.add_file("custom_files/model_metadata.json")
        reward_art.add_file("custom_files/reward_function.py")
        sys_art.add_file("system.env")
        run_art.add_file("run.env")
        wandb.use_artifact(params_art)
        wandb.use_artifact(agent_art)
        wandb.use_artifact(reward_art)
        wandb.use_artifact(sys_art)
        wandb.use_artifact(run_art)
    subprocess.run(f"git branch {wandb.run.name}", shell=True)
    subprocess.run(f"git push -u origin {wandb.run.name}", shell=True)

    # tables = reset_tables()


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
    # global tables
    global step_metrics

    timestamp = datetime.now()
    if "MY_TRACE_LOG" in line:
        parts = line.split("MY_TRACE_LOG:")[1].split("\t")[0].split("\n")[0].split(",")
        steps = int(float(parts[0]))
        waypoint = int(float(parts[1]))
        progress = float(parts[2])
        step_reward = float(parts[3])
        reward = float(parts[4])
        is_finished = int(parts[5])
        job = "train"
        if is_testing:
            job = "test"
        # if not DEBUG:
        #     tables[job].add_data(
        #         episode[job],
        #         steps,
        #         waypoint,
        #         progress,
        #         step_reward,
        #         reward,
        #     )
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
                for metric in ["reward", "progress", "steps"]:
                    if len(iter_metrics[job][metric]):
                        ckpt_metrics[job][metric] = np.mean(iter_metrics[job][metric])
                    else:
                        print(
                            f'{timestamp} WARNING: Empty list. Skipped {metric} mean calculation for ckpt {checkpoint}'
                        )

            # print(
            #     f'{timestamp} Same? {ckpt_metrics["test"]["reward"]:0.3f}, {float(test_reward):0.3f}'
            # )

            for metric in ["loss", "KL_div", "entropy"]:
                if len(iter_metrics["learn"][metric]):
                    ckpt_metrics["learn"][metric] = np.mean(iter_metrics["learn"][metric])
                else:
                    print(
                        f'{timestamp} WARNING: Empty list. Skipped {metric} mean calculation for ckpt {checkpoint}'
                    )

            # Estimate projected exits based on entropy
            projected_exits = ckpt_metrics["learn"]["entropy"] - MIN_ENTROPY

            # Estimate projected lap time
            if projected_exits > 0.0:
                projected_lap_time = (
                    ckpt_metrics["test"]["steps"]
                    + (projected_exits * 150.0)  # Each exit costs 10s or 150 steps
                ) / 15.0
            else:
                projected_lap_time = ckpt_metrics["test"]["steps"] / 15.0

            # Divide progress by projected lap time
            ckpt_metrics["test"]["combo"] = (
                ckpt_metrics["test"]["progress"] / projected_lap_time
            )

            # Update best metrics & summary
            if ckpt_metrics["test"]["combo"] > best_metrics["combo"]:
                best_metrics["reward"] = ckpt_metrics["test"]["reward"]
                best_metrics["steps"] = ckpt_metrics["test"]["steps"]
                best_metrics["progress"] = ckpt_metrics["test"]["progress"]
                best_metrics["checkpoint"] = checkpoint
                best_metrics["entropy"] = ckpt_metrics["learn"]["entropy"]
                best_metrics["combo"] = ckpt_metrics["test"]["combo"]
                print(
                    f'{timestamp} ckpt {checkpoint}: {ckpt_metrics["test"]["progress"]:0.2f}% in {ckpt_metrics["test"]["steps"]:0.2f} steps with {ckpt_metrics["test"]["reward"]:0.2f} reward and {ckpt_metrics["learn"]["entropy"]:0.2f} entropy ({ckpt_metrics["test"]["combo"]:0.2f} combo) → improved 👍'
                )
                if (
                    not DEBUG
                    and best_metrics["progress"] >= 100.0
                    and ckpt_metrics["test"]["steps"] <= GLOBAL_MIN_STEPS
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
                        path=f"s3://jdoc-one-deepracer-data-b5pi7cdvar/{wandb.run.name}-{checkpoint}/",
                        name=f"{wandb.run.name}",
                    )
            else:
                print(
                    f'{timestamp} ckpt {checkpoint}: {ckpt_metrics["test"]["progress"]:0.2f}% in {ckpt_metrics["test"]["steps"]:0.2f} steps with {ckpt_metrics["test"]["reward"]:0.2f} reward and {ckpt_metrics["learn"]["entropy"]:0.2f} entropy ({ckpt_metrics["test"]["combo"]:0.2f} combo)'
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
                        "test/combo": ckpt_metrics["test"]["combo"],
                        # "train_trace": tables["train"],
                        # "test_trace": tables["test"],
                    }
                )
                # Update test metrics summary
                wandb.run.summary["test/reward"] = best_metrics["reward"]
                wandb.run.summary["test/steps"] = best_metrics["steps"]
                wandb.run.summary["test/progress"] = best_metrics["progress"]
                wandb.run.summary["test/combo"] = best_metrics["combo"]
                wandb.run.summary["best_checkpoint"] = best_metrics["checkpoint"]
                wandb.run.summary["learn/entropy"] = best_metrics["entropy"]
            if ckpt_metrics["learn"]["entropy"] <= MIN_ENTROPY:
                subprocess.run("./stop-training.sh", shell=True)
        # Resetting tracker variables
        iter_metrics = reset_iter_metrics()
        # tables = reset_tables()
        is_testing = False

    elif "Starting evaluation phase" in line:
        is_testing = True
    elif "error" in line.lower() or "exception" in line.lower():
        print(f"{timestamp} {line}")
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
