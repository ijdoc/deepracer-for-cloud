#!/bin/bash

# Escape quotes in complex environment variables
ESCAPED_WANDB_ARTIFACTS=$(echo "${WANDB_ARTIFACTS}" | sed 's/"/\\"/g; s/'\''/\\'\''/g')
ESCAPED_WANDB_CONFIG=$(echo "${WANDB_CONFIG}" | sed 's/"/\\"/g; s/'\''/\\'\''/g')

# SSH into the host and execute the command
ssh -i /root/.ssh/id_ed25519 jdoc@workhorse \
  "export WANDB_BASE_URL='$WANDB_BASE_URL' \
   && export WANDB_PROJECT='$WANDB_PROJECT' \
   && export WANDB_ENTITY='$WANDB_ENTITY' \
   && export WANDB_LAUNCH='$WANDB_LAUNCH' \
   && export WANDB_DOCKER='$WANDB_DOCKER' \
   && export WANDB_USERNAME='$WANDB_USERNAME' \
   && export WANDB_RUN_ID='$WANDB_RUN_ID' \
   && export WANDB_LAUNCH_QUEUE_NAME='$WANDB_LAUNCH_QUEUE_NAME' \
   && export WANDB_LAUNCH_QUEUE_ENTITY='$WANDB_LAUNCH_QUEUE_ENTITY' \
   && export WANDB_LAUNCH_TRACE_ID='$WANDB_LAUNCH_TRACE_ID' \
   && export WANDB_CONFIG=\"$ESCAPED_WANDB_CONFIG\" \
   && export WANDB_ARTIFACTS=\"$ESCAPED_WANDB_ARTIFACTS\" \
   && source ~/.sshbashrc \
   && cd ~/repos/deepracer-for-cloud \
   && ./start-training.sh $@"

# Skipping variables that mess up the call
#  WANDB_ARTIFACTS=$WANDB_ARTIFACTS \
#  WANDB_API_KEY=$WANDB_API_KEY \
