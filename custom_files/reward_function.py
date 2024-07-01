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

    def get_sum(self):
        return sum(self.buffer)


# Globals
CONFIG = {
    "track": "2022_april_pro_ccw",
    "waypoint_count": 227,
    "aggregate": 22,
    "agent": {
        "steering_angle": {"high": 30.0, "low": -30.0},
        "speed": {"high": 1.6095110023512869, "low": 1.607511002351287},
    },
}
LAST = {
    "progress": 0.0,
    "position": [0.0, 0.0],
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
            "position": [params["x"], params["y"]],
        }
        BUFFER = {
            "progress": CircularBuffer(CONFIG["aggregate"], LAST["progress"]),
            "distance": CircularBuffer(CONFIG["aggregate"], 0.0),
        }

    # Get the path efficiency
    BUFFER["progress"].add_value(params["progress"] - LAST["progress"])
    BUFFER["distance"].add_value(
        math.hypot(params["x"] - LAST["position"][0], params["y"] - LAST["position"][1])
    )
    path_efficiency = 0.0
    progress = BUFFER["progress"].get_sum()
    distance_travelled = BUFFER["distance"].get_sum()
    if distance_travelled > 0.0:
        path_efficiency = progress / distance_travelled
    LAST["progress"] = params["progress"]
    LAST["position"] = [params["x"], params["y"]]

    if params["steps"] <= (CONFIG["aggregate"] / 2):
        reward = 0.0
    else:
        reward = float(path_efficiency) / 10.0

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for train & test logging
    print(
        f'MY_TRACE_LOG:{params["steps"]},{this_waypoint},{params["progress"]},{progress},{distance_travelled},{path_efficiency},{reward},{is_finished}'
    )

    return reward
