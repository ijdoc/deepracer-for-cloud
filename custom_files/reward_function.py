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

    def get_absnormcumdiff(self):
        cumdiff = 0.0
        for i in range(1, self.size):
            cumdiff += self.buffer[i] - self.buffer[i - 1]
        return abs(cumdiff) / self.size


# Globals
CONFIG = {
    "track": "jyllandsringen_open_ccw",
    "waypoint_count": 156,
    "aggregate": 15,
    "agent": {
        "steering_angle": {"high": 30.0, "low": -30.0},
        "speed": {"high": 2.2, "low": 1.9},
    },
}
LAST = {
    "progress": 0.0,
    "direction": 0.0,
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


def reward_function(params):
    global LAST
    global BUFFER

    this_waypoint = params["closest_waypoints"][0]
    this_direction = get_direction(this_waypoint, params["waypoints"])

    if params["steps"] <= 2:
        # Reset values at the beginning of the episode
        LAST = {
            "progress": 0.0,
            "direction": this_direction,
        }
        BUFFER = {
            "progress": CircularBuffer(CONFIG["aggregate"], LAST["progress"]),
            "direction": CircularBuffer(CONFIG["aggregate"], LAST["direction"]),
        }

    # Get the mean step progress
    BUFFER["progress"].add_value(params["progress"] - LAST["progress"])
    BUFFER["direction"].add_value(this_direction)
    mean_progress = BUFFER["progress"].get_mean()
    difficulty = 5.0 * BUFFER["direction"].get_absnormcumdiff()
    LAST["progress"] = params["progress"]
    LAST["direction"] = this_direction
    base_reward = 0.1

    reward = float(mean_progress * (1.0 + difficulty))

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for train & test logging
    print(
        f'MY_TRACE_LOG:{params["steps"]},{this_waypoint},{params["progress"]},{mean_progress},{difficulty},{reward},{is_finished}'
    )

    return reward
