import math


class CircularBuffer:
    def __init__(self, size, value=0.0):
        self.size = size
        self.buffer = [value] * size  # Initialize with value
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
    "track": "2022_april_pro_ccw",
    "waypoint_count": 227,
    "aggregate": 15,
    "agent": {
        "steering_angle": {"high": 30.0, "low": -30.0},
        "speed": {"high": 2.001, "low": 1.999},
    },
}
LAST = {
    "progress": 0.0,
}
BUFFER = {}


def reward_function(params):
    global LAST
    global BUFFER

    this_waypoint = params["closest_waypoints"][0]

    if params["steps"] <= 2:
        # Reset values at the beginning of the episode
        LAST = {
            "progress": 0.0,
        }
        BUFFER = {
            "progress": CircularBuffer(CONFIG["aggregate"], LAST["progress"]),
        }

    # Get the mean step progress
    BUFFER["progress"].add_value(params["progress"] - LAST["progress"])
    mean_progress = BUFFER["progress"].get_mean()
    LAST["progress"] = params["progress"]

    reward = float(mean_progress)

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for train & test logging
    print(
        f'MY_TRACE_LOG:{params["steps"]},{this_waypoint},{params["progress"]},{mean_progress},{reward},{is_finished}'
    )

    return reward
