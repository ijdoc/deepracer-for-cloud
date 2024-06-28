#!/usr/bin/env bash
set -e # Exit script immediately on first error.
source ./header.sh

# Function to display the script's usage/help information
display_help() {
    python watch.py -h
}

# Define default values for command-line options
agent_speed_high_value=2.0
agent_speed_low_value="$agent_speed_high_value"
reward_type_value=5
learning_rate_value=0.0002
bin_count_value=12
aggregate_value=15
skip_ahead_value=0
look_ahead_value=0
pretrained_flag=0
debug_flag=0

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
                agent-speed)
                    agent_speed_high_value="$value" # Directly use the parsed value
                    agent_speed_low_value="$value" # Directly use the parsed value
                    ;;
                agent-speed-high)
                    agent_speed_high_value="$value" # Directly use the parsed value
                    ;;
                agent-speed-low)
                    agent_speed_low_value="$value" # Directly use the parsed value
                    ;;
                reward-type)
                    reward_type_value="$value" # Directly use the parsed value
                    ;;
                learning-rate)
                    learning_rate_value="$value" # Directly use the parsed value
                    ;;
                bin-count)
                    bin_count_value="$value" # Directly use the parsed value
                    ;;
                aggregate)
                    aggregate_value="$value" # Directly use the parsed value
                    ;;
                skip-ahead)
                    skip_ahead_value="$value" # Directly use the parsed value
                    ;;
                look-ahead)
                    look_ahead_value="$value" # Directly use the parsed value
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
config_options="$config_options --learning-rate $learning_rate_value"
config_options="$config_options --reward-type $reward_type_value"
config_options="$config_options --bin-count $bin_count_value"
config_options="$config_options --aggregate $aggregate_value"
config_options="$config_options --skip-ahead $skip_ahead_value --look-ahead $look_ahead_value"

debug_option="--debug"
if [ $debug_flag -ne 1 ]; then
    debug_option=""
fi

pretrained_option="--pretrained"
if [ $pretrained_flag -ne 1 ]; then
    pretrained_option=""
fi


echo "WANDB_RUN_ID: $WANDB_RUN_ID"
echo "WANDB_CONFIG: $WANDB_CONFIG"
echo "WANDB_ARTIFACTS: $WANDB_ARTIFACTS"
echo "Config options: $config_options"

# Sweep compatible changes ahead:
# 1. Run config_update.py in debug mode
python config_update.py $config_options
# 2. Do not run config_verify.py
# python config_verify.py

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