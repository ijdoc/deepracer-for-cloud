#!/bin/bash
# SSH into the host and execute the command
ssh -i /root/.ssh/id_ed25519 jdoc@workhorse \
  "WANDB_BASE_URL=$WANDB_BASE_URL \
   WANDB_PROJECT=$WANDB_PROJECT \
   WANDB_ENTITY=$WANDB_ENTITY \
   WANDB_LAUNCH=$WANDB_LAUNCH \
   WANDB_RUN_ID=$WANDB_RUN_ID \
   WANDB_DOCKER=$WANDB_DOCKER \
   WANDB_USERNAME=$WANDB_USERNAME \
   WANDB_LAUNCH_QUEUE_NAME=$WANDB_LAUNCH_QUEUE_NAME \
   WANDB_LAUNCH_QUEUE_ENTITY=$WANDB_LAUNCH_QUEUE_ENTITY \
   WANDB_LAUNCH_TRACE_ID=$WANDB_LAUNCH_TRACE_ID \
   WANDB_CONFIG=\"$WANDB_CONFIG\" \
   source ~/.sshbashrc && \
   cd ~/repos/deepracer-for-cloud && \
   ./start-training.sh $@"

# Skipping variables that mess up the call
#  WANDB_ARTIFACTS=$WANDB_ARTIFACTS \
#  WANDB_API_KEY=$WANDB_API_KEY \
