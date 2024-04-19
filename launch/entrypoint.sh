#!/bin/bash
# SSH into the host and execute the command
ssh -i /root/.ssh/id_ed25519 jdoc@workhorse "source ~/.sshbashrc && cd ~/repos/deepracer-for-cloud && ./start-training.sh $@"