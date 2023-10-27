#!/usr/bin/env bash
source ./header.sh

source bin/activate.sh run.env
test_command_outcome "[upload.sh] Load environment"
dr-upload-model -bfw
test_command_outcome "[upload.sh] Upload model"