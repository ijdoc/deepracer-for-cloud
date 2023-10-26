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

# Create ArgumentParser
parser = argparse.ArgumentParser(description='Log testing metrics')

# Define arguments
parser.add_argument('--pretrained', action='store_true', help='This is a pretrained model')
parser.add_argument('--debug', action='store_true', help='This is a pretrained model')
parser.add_argument('--episodes', type=int, help='The maximum number of episodes to train for', default=10000)
parser.add_argument('--progress', type=int, help='The average evaluation progress to stop at', default=100)

# Parse the arguments
args = parser.parse_args()

DEBUG = args.debug
MAX_EPISODES = args.episodes
MAX_PROGRESS = args.progress

s3 = boto3.client('s3',
                    aws_access_key_id=os.environ["DR_LOCAL_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["DR_LOCAL_SECRET_ACCESS_KEY"],
                    endpoint_url="http://localhost:9000")

# Configure project path
os.environ["WANDB_ENTITY"] = "iamjdoc"
os.environ["WANDB_PROJECT"] = "dr-reborn"

# Don't litter the console
os.environ["WANDB_SILENT"] = "true"

# Make sure it is possible to resume & auto-create runs
os.environ["WANDB_RESUME"] = "allow"

# running_job = jobs["train"]["id"]
iter_metrics = {"test":{"reward": None, "steps": [], "progress": [], "speed": []},
                "train":{"reward": [], "steps": [], "progress": [], "speed": []},
                "learn":{"loss": [], "KL_div": [], "entropy": []}}
best_metrics = {"reward": -1.0, "steps": 0, "progress": 0.0, "speed": 0.0}
is_stopped = True
start_step = {"timestamp": None, "progress": None}
is_testing = False
test_metrics = {"steps": None, "progress": None}
last_episode = 0

# Define group
os.environ["WANDB_RUN_GROUP"] = "2310"

def update_run_env(name):
    # Open the file in read and write mode
    file_path = "./run.env"
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Modify the content in memory
    new_lines = []
    for line in lines:
        if line.startswith("DR_UPLOAD_S3_PREFIX="):
            new_lines.append(f"DR_UPLOAD_S3_PREFIX={name}\n")
        else:
            new_lines.append(line)
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)

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
    update_run_env(wandb.run.name)
    # Log input files
    config_files = wandb.Artifact(name="config", type="inputs")
    env_files = wandb.Artifact(name="env", type="inputs")
    config_files.add_dir("./custom_files")
    env_files.add_file("./run.env")
    env_files.add_file("./system.env")
    wandb.use_artifact(config_files)
    wandb.use_artifact(env_files)

def get_float(string):
    try:
        x = float(string)
        return x
    except ValueError:
        return None

def process_line(line):
    # Process training episodes and policy training
    global iter_metrics
    global is_stopped
    global start_step
    global is_testing
    global best_metrics
    global last_episode

    timestamp = datetime.now()
    if "Training>" in line and "[SAGE]" in line:
        # Capture training episode metrics
        metrics = line.split(",")
        last_episode = int(metrics[2].split("=")[1])
        reward = float(metrics[3].split("=")[1])
        # steps = int(metrics[4].split("=")[1])
        # iter = int(metrics[5].split("=")[1])
        iter_metrics["train"]["reward"].append(reward)
    elif "Policy training>"  in line:
        metrics = line.split(",")
        # epoch = int(metrics[3].split("=")[1])
        loss = float(metrics[0].split("=")[1])
        divergence = float(metrics[1].split("=")[1])
        entropy = float(metrics[2].split("=")[1])
        metrics = {"learn/loss":loss, "learn/KL_div": divergence, "learn/entropy": entropy}
        iter_metrics["learn"]["loss"].append(loss)
        iter_metrics["learn"]["KL_div"].append(divergence)
        iter_metrics["learn"]["entropy"].append(entropy)
    elif "[BestModelSelection] Evaluation episode reward mean:" in line:
        test_reward = get_float(line.split(":")[1].strip())
        if not test_reward is None:
            # Calculate means and log everything here!!
            iter_metrics["test"]["reward"] = float(test_reward)
            iter_metrics["test"]["steps"] = np.mean(iter_metrics["test"]["steps"])
            iter_metrics["test"]["progress"] = np.mean(iter_metrics["test"]["progress"])
            iter_metrics["test"]["speed"] = np.mean(iter_metrics["test"]["speed"])
            iter_metrics["train"]["reward"] = np.mean(iter_metrics["train"]["reward"])
            iter_metrics["train"]["steps"] = np.mean(iter_metrics["train"]["steps"])
            iter_metrics["train"]["progress"] = np.mean(iter_metrics["train"]["progress"])
            iter_metrics["train"]["speed"] = np.mean(iter_metrics["train"]["speed"])
            iter_metrics["learn"]["loss"] = np.mean(iter_metrics["learn"]["loss"])
            iter_metrics["learn"]["KL_div"] = np.mean(iter_metrics["learn"]["KL_div"])
            iter_metrics["learn"]["entropy"] = np.mean(iter_metrics["learn"]["entropy"])
            # Update best metrics for summary
            if iter_metrics["test"]["reward"] > best_metrics["reward"]:
                best_metrics["reward"] = iter_metrics["test"]["reward"]
                best_metrics["steps"] = iter_metrics["test"]["steps"]
                best_metrics["progress"] = iter_metrics["test"]["progress"]
                best_metrics["speed"] = iter_metrics["test"]["speed"]
            if DEBUG:
                print(f"{timestamp} [W&B] {iter_metrics}")
            else:
                wandb.log({"train/reward": iter_metrics["train"]["reward"],
                           "train/steps": iter_metrics["train"]["steps"],
                           "train/progress": iter_metrics["train"]["progress"],
                           "train/speed": iter_metrics["train"]["speed"],
                           "learn/loss": iter_metrics["learn"]["loss"],
                           "learn/KL_div": iter_metrics["learn"]["KL_div"],
                           "learn/entropy": iter_metrics["learn"]["entropy"],
                           "test/reward": iter_metrics["test"]["reward"],
                           "test/steps": iter_metrics["test"]["steps"],
                           "test/progress": iter_metrics["test"]["progress"],
                           "test/speed": iter_metrics["test"]["speed"]
                           })
                # Update test metrics summary
                wandb.run.summary["test/reward"] = best_metrics["reward"]
                wandb.run.summary["test/steps"] = best_metrics["steps"]
                wandb.run.summary["test/progress"] = best_metrics["progress"]
                wandb.run.summary["test/speed"] = best_metrics["speed"]
            if last_episode >= MAX_EPISODES or iter_metrics["test"]["progress"] >= MAX_PROGRESS:
                subprocess.run("source bin/activate.sh run.env && dr-stop-training", shell=True)
            # Reset metrics
            iter_metrics = {"test":{"reward": None, "steps": [], "progress": [], "speed": []},
                            "train":{"reward": [], "steps": [], "progress": [], "speed": []},
                            "learn":{"loss": [], "KL_div": [], "entropy": []}}
        is_testing = False
    elif "[BestModelSelection] Updating the best checkpoint" in line:
        name = line.split("\"")[1]
        name = name.split(".")[0]
        print(f"{timestamp} Best checkpoint: {name} at episode {last_episode}")
    elif "SIM_TRACE_LOG" in line:
        parts = line.split("SIM_TRACE_LOG:")[1].split('\t')[0].split('\n')[0].split(",")
        if is_stopped:
            timestamp = float(parts[14])
            progress = float(parts[11])
            start_step = {"timestamp": timestamp, "progress": progress}
            if DEBUG:
                print(f"{timestamp} [DEBUG] {start_step}")
            is_stopped = False
        if 'True' in parts[9]:
            timestamp = float(parts[14])
            steps = int(parts[1])
            progress = float(parts[11])
            # speed = ((progress / 100.0) * float(parts[13])) / (timestamp - start_step["timestamp"])
            speed = progress / steps
            if DEBUG:
                print(f"{timestamp} [DEBUG] Steps: {steps} Progress: {progress} Speed: {speed}")
            iter_metrics["train"]["steps"].append(steps)
            iter_metrics["train"]["progress"].append(progress)
            iter_metrics["train"]["speed"].append(speed)
            is_stopped = True
    elif "Starting evaluation phase" in line:
        is_testing = True
    elif "Testing>" in line:
        iter_metrics["test"]["steps"].append(test_metrics["steps"]) 
        iter_metrics["test"]["progress"].append(test_metrics["progress"]) 
        iter_metrics["test"]["speed"].append(test_metrics["progress"] / test_metrics["steps"]) 
        if DEBUG:
            print(f"{timestamp} [W&B] test metrics: {test_metrics}")
    elif "MY_TRACE_LOG" in line:
        if is_testing:
            parts = line.split("MY_TRACE_LOG:")[1].split('\t')[0].split('\n')[0].split(",")
            test_metrics["steps"] = int(float(parts[0]))
            test_metrics["progress"] = float(parts[1])
            if DEBUG:
                print(f"{timestamp} {line}")
    else:
        if DEBUG:
            print(f"{timestamp} {line}")

def aggregate_logs(container, label):
    buffer = ''
    logs = container.logs(stream=True, follow=True)
    for chunk in logs:
        # Process each line of the log here
        buffer += chunk.decode('utf-8')
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            if line.strip() != "":
                process_line(f'{label} {line.strip()}')

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

model_found = False
while not model_found:
    # List the objects in the bucket
    response = s3.list_objects_v2(Bucket=os.environ["DR_LOCAL_S3_BUCKET"], Prefix="rl-deepracer-sagemaker")
    # Display the contents of the bucket
    for obj in response.get('Contents', []):
        if "model.tar.gz" in obj['Key']:
            model_found = True
            print(f"[W&B] Model found")
            # Download model
            s3.download_file(os.environ["DR_LOCAL_S3_BUCKET"], "rl-deepracer-sagemaker/model.tar.gz", "model.tar.gz")
            print(f"[W&B] Model updated")


# FIXME: Upload to bucket and then reference
# Log model!
subprocess.run("source bin/activate.sh run.env && dr-upload-model -b", shell=True)
if not DEBUG:
    # resume_job(jobs["train"])
    model = wandb.Artifact(f"racer-model", type="model")
    model.add_file("./model.tar.gz", "model.tar.gz")
    wandb.log_artifact(model)
    print(f"[W&B] Model logged")
    print(f"[W&B] Finishing...")
    wandb.finish()
