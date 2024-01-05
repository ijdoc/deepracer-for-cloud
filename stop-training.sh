#!/usr/bin/env bash
source ./header.sh

source bin/activate.sh run.env
test_command_outcome "[upload.sh] Load environment"
dr-stop-training
test_command_outcome "[upload.sh] Stop training"