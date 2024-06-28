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
    "reward_type": 0,
    "waypoint_count": 156,
    "aggregate": 15,
    "importance": {
        "counts": [2, 2, 5, 6, 6, 64, 37, 15, 5, 4, 5, 5],
        "values": [
            1.0,
            1.0,
            0.3806,
            0.3118,
            0.3118,
            0.0,
            0.0235,
            0.1054,
            0.3806,
            0.4839,
            0.3806,
            0.3806,
        ],
        "edges": [
            -0.44222918675310563,
            -0.36422850757194436,
            -0.2862278283907831,
            -0.20822714920962176,
            -0.1302264700284605,
            -0.052225790847299225,
            0.0257748883338621,
            0.10377556751502337,
            0.18177624669618464,
            0.2597769258773459,
            0.3377776050585072,
            0.41577828423966845,
            0.49377896342082983,
        ],
    },
    "difficulty": {
        "skip-ahead": 0,
        "look-ahead": 5,
        "max": 2.028487986274303,
        "min": 0.0,
        "histogram": {
            "counts": [122, 34],
            "edges": [0.0, 1.0142439931371514, 2.028487986274303],
            "values": [2.3, 2.3],
        },
    },
    "agent": {
        "steering_angle": {"high": 30.0, "low": -30.0},
        "speed": {"high": 2.3009999999999997, "low": 2.299},
    },
}
LAST_PROGRESS = 0.0
PROGRESS_BUFFER = None

# Values we don't want to keep recalculating
THROTTLE_RANGE = CONFIG["agent"]["speed"]["high"] - CONFIG["agent"]["speed"]["low"]
REWARD_TYPE = CONFIG["reward_type"]


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
        0.4 * (params["speed"] - CONFIG["agent"]["speed"]["low"]) / THROTTLE_RANGE
    ) + 0.8

    if REWARD_TYPE == 0:
        reward = float(mean_progress)
    if REWARD_TYPE == 1:
        reward = float(mean_progress * throttle_factor)

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for train & test logging
    print(
        f'MY_TRACE_LOG:{params["steps"]},{this_waypoint},{params["progress"]},{mean_progress},{params["speed"]},{params["steering_angle"]},{reward},{is_finished}'
    )

    return reward
