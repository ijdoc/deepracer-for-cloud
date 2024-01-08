#!/usr/bin/env bash
set -e # Exit script immediately on first error.
source ./header.sh

source bin/activate.sh run.env
test_command_outcome "[upload.sh] Load environment"
dr-upload-model -bfw
test_command_outcome "[upload.sh] Upload model"