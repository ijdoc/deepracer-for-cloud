import math
import time


class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [0.0] * size  # Initialize buffer with zeroes
        self.index = 0  # Start index

    def add_value(self, value):
        self.buffer[self.index] = value  # Set new value at the current index
        self.index = (self.index + 1) % self.size  # Update index circularly

    def get_values(self):
        return self.buffer

    def get_mean(self):
        return sum(self.buffer) / self.size


# Globals
CONFIG = {
    "track": "jyllandsringen_open_ccw",
    "waypoint_count": 156,
    "aggregate": 15,
    "agent": {
        "steering_angle": {"high": 30.0, "low": -30.0},
        "speed": {"high": 2.6, "low": 2.1},
    },
}
LAST_PROGRESS = 0.0
PROGRESS_BUFFER = None

# Values we don't want to keep recalculating
THROTTLE_RANGE = CONFIG["agent"]["speed"]["high"] - CONFIG["agent"]["speed"]["low"]


def reward_function(params):
    global LAST_PROGRESS
    global PROGRESS_BUFFER

    this_waypoint = params["closest_waypoints"][0]

    if params["steps"] <= 2:
        # Reset progress at the beginning of the episode
        LAST_PROGRESS = 0.0
        PROGRESS_BUFFER = CircularBuffer(CONFIG["aggregate"])

    # Get the mean step progress
    PROGRESS_BUFFER.add_value(params["progress"] - LAST_PROGRESS)
    mean_progress = PROGRESS_BUFFER.get_mean()
    LAST_PROGRESS = params["progress"]

    if mean_progress == 0.0:
        projected_steps = 10000.0
    else:
        projected_steps = 100.0 / mean_progress

    # Trottle_factor ranges from 1.2 to 0.8, where 1.2 is given to max speed
    throttle_factor = (
        CONFIG["throttle_factor_diff"]
        * (CONFIG["agent"]["speed"]["high"] - params["speed"])
        / THROTTLE_RANGE
    ) + 1.0

    reward = float(mean_progress * throttle_factor)

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for train & test logging
    print(
        f'MY_TRACE_LOG:{params["steps"]},{this_waypoint},{params["progress"]},{mean_progress},{params["speed"]},{params["steering_angle"]},{reward},{is_finished}'
    )

    return reward
