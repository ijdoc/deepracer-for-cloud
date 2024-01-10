#!/usr/bin/env bash
set -e # Exit script immediately on first error.
source ./header.sh

# Initialize flags
pretrained_flag=0

# Function to display the script's usage/help information
display_help() {
    python watch.py -h
}

# Parse command-line options
debug_flag=0
while getopts ":h-:" opt; do
    case "$opt" in
        h)
            display_help
            exit 0
            ;;
        -)
            case "$OPTARG" in
                help)
                    display_help
                    exit 0
                    ;;
                debug)
                    debug_flag=1
                    # Continue (just accept the option)
                    ;;
                pretrained)
                    pretrained_flag=1
                    ;;
                *)
                    echo "Invalid option: --$OPTARG"
                    exit 1
                    ;;
            esac
            ;;
        \?)
            echo "Top invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done

# Skip git check if debugging
if [ $debug_flag -ne 1 ]; then
    # Check if the branch is dirty
    if [[ -n $(git status --porcelain) ]]; then
      error "Your Git branch is dirty. Please commit your changes."
    fi
fi

source bin/activate.sh run.env
dr-stop-viewer && dr-stop-training
test_command_outcome "[$0] Stop previous training"
dr-reload
test_command_outcome "[$0] Reload"
dr-update && dr-update-env && dr-upload-custom-files
test_command_outcome "[$0] Update and upload training files"

# Placeholder if statement for the pretrained option
if [ $pretrained_flag -eq 1 ]; then
    dr-upload-custom-files
    test_command_outcome "[$0] Upload custom files"
    dr-increment-training
    test_command_outcome "[$0] Increment training"
    dr-start-training -wva
else
    # 'w' for overwrite, 'v' for start viewer, 'a' for follow all logs
    dr-start-training -wva
fi

sleep 1
python watch.py "$@"