#!/usr/bin/env bash
set -e # Exit script immediately on first error.
source ./header.sh

# Function to display the script's usage/help information
display_help() {
    python watch.py -h
}

# Define default values for command-line options
debug_flag=0
pretrained_flag=0
look_ahead_value=3
bin_count_value=10
delay_value=4
offset_value=1.4
agent_speed_high_value=2.4
agent_speed_low_value=0.8

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
                    ;;
                agent-speed-high)
                    # The next argument to $OPTARG will be the value
                    agent_speed_high_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value
                    ;;
                agent-speed-low)
                    # The next argument to $OPTARG will be the value
                    agent_speed_low_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value
                    ;;
                look-ahead)
                    # The next argument to $OPTARG will be the value
                    look_ahead_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value
                    ;;
                bin-count)
                    # The next argument to $OPTARG will be the value
                    bin_count_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value
                    ;;
                delay)
                    # The next argument to $OPTARG will be the value
                    delay_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value
                    ;;
                offset)
                    # The next argument to $OPTARG will be the value
                    offset_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value
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
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done

config_options="--agent-speed-high $agent_speed_high_value"
config_options="$config_options --agent-speed-low $agent_speed_low_value"
config_options="$config_options --look-ahead $look_ahead_value --bin-count $bin_count_value"
config_options="$config_options --delay $delay_value --offset $offset_value"

# Skip git check if debugging
debug_option="--debug"
if [ $debug_flag -ne 1 ]; then
    debug_option=""
fi

if [ $debug_flag -ne 1 ]; then
    # Check if the branch is dirty
    if [[ -n $(git status --porcelain) ]]; then
      error "Your Git branch is dirty. Please commit your changes."
    fi
fi

# Edited to be sweep compatible (only run config_update.py in debug mode)
python config_update.py $config_options --debug
# python config_verify.py $debug_option

source bin/activate.sh run.env
dr-stop-viewer && dr-stop-training
test_command_outcome "[$0] Stop previous training"
dr-reload
test_command_outcome "[$0] Reload"
sleep 10
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

sleep 5
python watch.py "$@"