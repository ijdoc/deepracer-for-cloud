#!/usr/bin/env bash
source ./header.sh

# Function to display the script's usage/help information
display_help() {
    python watch.py -h
}

# Parse command-line options
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
                    # Continue (just accept the option)
                    ;;
                progress)
                    # Continue (just accept the option)
                    ;;
                episodes)
                    # Continue (just accept the option)
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
dr-stop-viewer
dr-stop-training
dr-update && dr-update-env && dr-upload-custom-files
# 'w' for overwrite, 'v' for start viewer, 'a' for follow all logs
dr-start-training -wva
sleep 1
python watch.py "$@"
