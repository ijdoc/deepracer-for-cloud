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
            # Splitting the argument into the name and value parts
            option="${OPTARG%%=*}" # Extract the option name
            value="${OPTARG#*=}" # Extract the value
            case "$option" in
                help)
                    display_help
                    exit 0
                    ;;
                debug)
                    debug_flag=1
                    ;;
                agent-speed-high)
                    agent_speed_high_value="$value" # Directly use the parsed value
                    ;;
                agent-speed-low)
                    agent_speed_low_value="$value" # Directly use the parsed value
                    ;;
                look-ahead)
                    look_ahead_value="$value" # Directly use the parsed value
                    ;;
                bin-count)
                    bin_count_value="$value" # Directly use the parsed value
                    ;;
                delay)
                    delay_value="$value" # Directly use the parsed value
                    ;;
                offset)
                    offset_value="$value" # Directly use the parsed value
                    ;;
                pretrained)
                    pretrained_flag=1
                    ;;
                *)
                    echo "Invalid option: --$option"
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

debug_option="--debug"
if [ $debug_flag -ne 1 ]; then
    debug_option=""
fi

pretrained_option="--pretrained"
if [ $pretrained_flag -ne 1 ]; then
    pretrained_option=""
fi

# Sweep compatible changes ahead:
# 1. Run config_update.py in debug mode
python config_update.py $config_options --debug
# 2. Do not run config_verify.py
# python config_verify.py $debug_option

# 3. Do not check for dirty Git branch
# if [ $debug_flag -ne 1 ]; then
#     # Check if the branch is dirty
#     if [[ -n $(git status --porcelain) ]]; then
#       error "Your Git branch is dirty. Please commit your changes."
#     fi
# fi

# 4. Resent run.env (in case a model was uploaded in previous run)
git checkout -- run.env

# Go ahead and start actual training
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
python watch.py $debug_option $pretrained_option