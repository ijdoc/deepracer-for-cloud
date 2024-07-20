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

    def get_mean(self):
        return sum(self.buffer) / self.size


# Globals
CONFIG = {
    "track": "2022_april_pro_ccw",
    "waypoint_count": 227,
    "aggregate": 10,
    "look-ahead": 7,
    "agent": {
        "steering_angle": {"high": 30.0, "low": -30.0},
        "speed": {"high": 2.2502132028215445, "low": 1.4220324493442158},
    },
}
LAST = {
    "progress": 0.0,
    "position": [0.0, 0.0],
}
BUFFER = {}


def get_next_distinct_index(i, waypoints):
    """
    Get the next distinct waypoint index in a path, after i (skip duplicate waypoints)
    """
    ahead = i + 1
    if i == len(waypoints) - 1:
        ahead = 0
    if (
        waypoints[i][0] == waypoints[ahead][0]
        and waypoints[i][1] == waypoints[ahead][1]
    ):
        return get_next_distinct_index(ahead, waypoints)
    return ahead


def get_prev_distinct_index(i, waypoints):
    """
    Get the previous distinct waypoint index in a path, before i (skip duplicate waypoints)
    """
    prev = i - 1
    if i == 0:
        prev = len(waypoints) - 1
    if waypoints[i][0] == waypoints[prev][0] and waypoints[i][1] == waypoints[prev][1]:
        return get_prev_distinct_index(prev, waypoints)
    return prev


def get_direction(i, waypoints):
    """
    Get direction at waypoint i
    """
    behind = get_prev_distinct_index(i, waypoints)
    ahead = get_next_distinct_index(i, waypoints)
    behindx = (waypoints[i][0] + waypoints[behind][0]) / 2.0
    behindy = (waypoints[i][1] + waypoints[behind][1]) / 2.0
    behind = [behindx, behindy]
    aheadx = (waypoints[ahead][0] + waypoints[i][0]) / 2.0
    aheady = (waypoints[ahead][1] + waypoints[i][1]) / 2.0
    ahead = [aheadx, aheady]
    diffx = ahead[0] - behind[0]
    diffy = ahead[1] - behind[1]
    return math.atan2(diffy, diffx)


def get_direction_change_ahead(i, waypoints):
    ahead = get_next_distinct_index(i, waypoints)
    this_direction = get_direction(i, waypoints)
    diff = get_direction(ahead, waypoints) - this_direction
    diff = math.atan2(math.sin(diff), math.cos(diff))
    return diff


def sigmoid(x, k=3.9, x0=0.6, ymin=0.0, ymax=1.2):
    """Parametrized sigmoid function as seen on:
       https://www.desmos.com/calculator/wbdyedqfwp

    Args:
        x (float): the input value
        k (float): _summary_
        x0 (float): _summary_
        ymin (float): _summary_
        ymax (float): _summary_

    Returns:
        float: the value of the sigmoid function at x
    """

    # Prevent math range error for large values of k * (x - x0)
    exp_arg = -k * (x - x0)
    if exp_arg < -700:  # any number smaller than this will cause underflow
        return ymin
    if exp_arg > 700:  # any number larger than this will cause overflow
        return ymax

    return ((ymax - ymin) / (1 + math.exp(-k * (x - x0)))) + ymin


def reward_function(params):
    global LAST
    global BUFFER

    this_waypoint = params["closest_waypoints"][0]
    next_waypoint = this_waypoint
    difficulty = 0.0
    for i in range(CONFIG["look-ahead"]):
        difficulty += get_direction_change_ahead(next_waypoint, params["waypoints"])
        next_waypoint = get_next_distinct_index(next_waypoint, params["waypoints"])
    difficulty = math.fabs(difficulty) / CONFIG["look-ahead"]

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
    else:
        # Add values to the buffer
        BUFFER["progress"].add_value(params["progress"] - LAST["progress"])
        BUFFER["distance"].add_value(
            math.hypot(
                params["x"] - LAST["position"][0], params["y"] - LAST["position"][1]
            )
        )
    buffer_progress = BUFFER["progress"].get_sum()
    distance_travelled = BUFFER["distance"].get_sum()
    path_efficiency = 0.0
    if distance_travelled > 0.0:
        path_efficiency = buffer_progress / distance_travelled
    mean_progress = BUFFER["progress"].get_mean()
    LAST["progress"] = params["progress"]
    LAST["position"] = [params["x"], params["y"]]

    # Goes from 0 to 1 as speed goes from low to high
    speed_fraction = (params["speed"] - CONFIG["agent"]["speed"]["low"]) / (
        CONFIG["agent"]["speed"]["high"] - CONFIG["agent"]["speed"]["low"]
    )
    # Goes from 1 to 0 as difficulty increases from 0 to ~pi/10
    speed_importance = sigmoid(difficulty, k=-40, x0=0.1, ymin=0.0, ymax=1.0)
    speed_factor = (1.0 - speed_importance) + (speed_fraction * speed_importance)

    reward = float(mean_progress * abs(path_efficiency) * speed_factor)

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for train & test logging
    print(
        f'MY_TRACE_LOG:{params["steps"]},{this_waypoint},{params["progress"]},{mean_progress},{path_efficiency},{difficulty},{params["speed"]},{reward},{is_finished}'
    )

    return reward
