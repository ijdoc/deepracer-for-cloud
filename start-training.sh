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
                look-ahead)
                    # The next argument to $OPTARG will be the value for look-ahead
                    look_ahead_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value for look-ahead
                    ;;
                bin-count)
                    # The next argument to $OPTARG will be the value for look-ahead
                    bin_count_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value for look-ahead
                    ;;
                delay)
                    # The next argument to $OPTARG will be the value for look-ahead
                    delay_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value for look-ahead
                    ;;
                offset)
                    # The next argument to $OPTARG will be the value for look-ahead
                    offset_value="${!OPTIND}" # Use indirect variable expansion to get the next argument
                    OPTIND=$((OPTIND + 1)) # Increment the option index to consume the value for look-ahead
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

reward_options="--look-ahead $look_ahead_value --bin-count $bin_count_value --delay $delay_value --offset $offset_value"

python reward_config_update.py $reward_options $debug_option
python reward_config_verify.py $debug_option

# Skip git check if debugging
debug_option="--debug"
if [ $debug_flag -ne 1 ]; then
    # Check if the branch is dirty
    if [[ -n $(git status --porcelain) ]]; then
      error "Your Git branch is dirty. Please commit your changes."
    fi
    debug_option=""
fi

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