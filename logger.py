import docker
import threading
import os
import wandb
import boto3
from datetime import datetime

DEBUG = True

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

# Define run IDs
jobs = dict(train = dict(name = "Train", id = wandb.util.generate_id()),
            learn = dict(name = "Learn", id = wandb.util.generate_id()),
            test = dict(name = "Test", id = wandb.util.generate_id()),)

running_job = jobs["train"]["id"]
phase = jobs["train"]["name"]

# Define group
os.environ["WANDB_RUN_GROUP"] = jobs["learn"]["id"]

# Start with training job
if not DEBUG:
    run = wandb.init(id=jobs["train"]["id"], job_type=jobs["train"]["name"], resume=True)

def resume_job(job):
    global running_job
    wandb.finish()
    running_job = job["id"]
    return wandb.init(id=job["id"], job_type=job["name"], resume=True)

def process_line(line):
    # global run
    # Process training episodes and policy training
    timestamp = datetime.now()
    if "Training>" in line and "[SAGE]" in line:
        # Capture training episode metrics
        metrics = line.split(",")
        # episode = int(metrics[2].split("=")[1])
        reward = float(metrics[3].split("=")[1])
        steps = int(metrics[4].split("=")[1])
        iter = int(metrics[5].split("=")[1])
        #TODO: Calculate progress and total time!
        metrics = {"train/reward":reward, "train/steps": steps, "train/iteration": iter}
        print(f"[W&B] {timestamp}", metrics)
        if not DEBUG:
            if running_job != jobs["train"]["id"]:
                run = resume_job(jobs["train"])
            run.log(metrics)
    elif "Policy training>"  in line:
        metrics = line.split(",")
        # epoch = int(metrics[3].split("=")[1])
        loss = float(metrics[0].split("=")[1])
        divergence = float(metrics[1].split("=")[1])
        entropy = float(metrics[2].split("=")[1])
        metrics = {"learn/loss":loss, "learn/KL_div": divergence, "learn/entropy": entropy}
        print(f"[W&B] {timestamp}", metrics)
        if not DEBUG:
            if running_job != jobs["learn"]["id"]:
                run = resume_job(jobs["learn"])
            run.log(metrics)
    elif "Testing>"  in line:
        # Capture testing episode metrics
        metrics = line.split(",")
        reward = float(metrics[3].split("=")[1])
        #TODO: Calculate progress and total time!
        metrics = {"test/reward":reward}
        print(f"[W&B] {timestamp}", metrics)
        if not DEBUG:
            if running_job != jobs["test"]["id"]:
                run = resume_job(jobs["test"])
            run.log(metrics)
    elif "[BestModelSelection] Updating the best checkpoint" in line:
        name = line.split("\"")[1]
        print(f"[W&B]  {timestamp} Best checkpoint: {name}")
        # List the objects in the bucket
        response = s3.list_objects_v2(Bucket=os.environ["DR_LOCAL_S3_BUCKET"], Key="rl-deepracer-sagemaker")
        # Display the contents of the bucket
        for obj in response.get('Contents', []):
            print(f"[W&B]  Object Key: {obj['Key']} - Last Modified: {obj['LastModified']} - Size: {obj['Size']} bytes")
    elif "Starting evaluation phase" in line:
        # Switch to evaluation
        if phase != jobs["test"]["name"]:
            phase = jobs["test"]["name"]
    elif "Reset agent finished" in line:
        # Start timer if evaluation
        if phase == jobs["test"]["name"]:
            pass
    elif "SIM_TRACE_LOG" in line:
        if phase != jobs["train"]["name"]:
            phase = jobs["train"]["name"]
    else:
        print(line)

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