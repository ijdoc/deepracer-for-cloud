import math
import time


class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [0.0] * size  # Initialize buffer with zero
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
    "track": "Albert",
    "reward_type": 6,
    "waypoint_count": 344,
    "aggregate": 15,
    "histogram": {
        "counts": [3, 5, 18, 29, 38, 49, 54, 59, 31, 30, 15, 13],
        "weights": [
            1.0,
            0.5786,
            0.122,
            0.0554,
            0.0296,
            0.0109,
            0.005,
            0.0,
            0.0484,
            0.0518,
            0.1571,
            0.1896,
        ],
        "edges": [
            -0.1818631112092999,
            -0.1567841240486577,
            -0.1317051368880155,
            -0.1066261497273733,
            -0.0815471625667311,
            -0.056468175406088894,
            -0.0313891882454467,
            -0.006310201084804484,
            0.018768786075837707,
            0.0438477732364799,
            0.06892676039712212,
            0.09400574755776434,
            0.11908473471840651,
        ],
    },
    "difficulty": {
        "skip-ahead": 0,
        "look-ahead": 5,
        "max": 0.9612126837475136,
        "min": 0.001130939116824159,
    },
    "agent": {
        "steering_angle": {"high": 30.0, "low": -30.0},
        "speed": {"high": 3.2, "low": 2.48},
    },
}
LAST_PROGRESS = 0.0
PROGRESS_BUFFER = None
SMOOTHNESS_BUFFER = None
LAST_THROTTLE = 0.0
LAST_STEERING = 0.0

# Values we don't want to keep recalculating
throttle_range = CONFIG["agent"]["speed"]["high"] - CONFIG["agent"]["speed"]["low"]
novelty_factor = max(CONFIG["histogram"]["counts"]) / min(CONFIG["histogram"]["counts"])


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


def get_direction_change(i, waypoints):
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


def get_waypoint_difficulty(
    i, waypoints, skip_ahead=1, look_ahead=3, min_val=0.0, max_val=1.0
):
    """
    Get difficulty at waypoint i, calculated as the normalized amount of direction change.
    """
    next_idx = i
    for _ in range(skip_ahead):
        next_idx = get_next_distinct_index(next_idx, waypoints)
    aggregate_change = get_direction_change(next_idx, waypoints)
    for _ in range(look_ahead):
        aggregate_change += get_direction_change(next_idx, waypoints)
        next_idx = get_next_distinct_index(next_idx, waypoints)
    difficulty = abs(aggregate_change)
    normalized_difficulty = (difficulty - min_val) / (max_val - min_val)
    return aggregate_change, difficulty, normalized_difficulty


def get_waypoint_importance(change, histogram):
    """
    Get waypoint weight based on how rare the direction change is in the entire track.
    This can be used to give more weight to waypoints that are less likely to occur during
    training, and therefore are more important to learn.
    """
    for j in range(len(histogram["weights"])):
        if change >= histogram["edges"][j] and change < histogram["edges"][j + 1]:
            return histogram["weights"][j]
    return histogram["weights"][j]  # when change is equal to the last edge


def subtract_angles_rad(a, b):
    diff = a - b
    return math.atan2(math.sin(diff), math.cos(diff))


def reward_function(params):
    global LAST_THROTTLE
    global LAST_STEERING
    global LAST_PROGRESS
    global PROGRESS_BUFFER
    global SMOOTHNESS_BUFFER

    this_waypoint = params["closest_waypoints"][0]

    if params["steps"] <= 2:
        # Reset progress at the beginning of the episode
        LAST_THROTTLE = 0.0
        LAST_STEERING = 0.0
        LAST_PROGRESS = 0.0
        PROGRESS_BUFFER = CircularBuffer(CONFIG["aggregate"])
        SMOOTHNESS_BUFFER = CircularBuffer(CONFIG["aggregate"])

    # Get the mean step progress
    PROGRESS_BUFFER.add_value(params["progress"] - LAST_PROGRESS)
    mean_progress = PROGRESS_BUFFER.get_mean()
    LAST_PROGRESS = params["progress"]

    # Get the action change
    agent_change = (
        abs(params["steering_angle"] - LAST_STEERING)
        / (
            CONFIG["agent"]["steering_angle"]["high"]
            - CONFIG["agent"]["steering_angle"]["low"]
        )
        + abs(params["speed"] - LAST_THROTTLE)
        / (CONFIG["agent"]["speed"]["high"] - CONFIG["agent"]["speed"]["low"])
    ) / 2.0
    LAST_STEERING = params["steering_angle"]
    LAST_THROTTLE = params["speed"]

    # Smoothness ranges from 1 to -1, where 1 is the smoothest
    SMOOTHNESS_BUFFER.add_value(2.0 * (0.5 - agent_change))
    mean_smoothness = SMOOTHNESS_BUFFER.get_mean()

    if mean_progress == 0.0:
        projected_steps = 10000.0
    else:
        projected_steps = 100.0 / mean_progress

    # if mean_progress >= 0.0:
    #     # We reward projected_steps based on each step's progress.
    #     # The sigmoid saturates the reward to a maximum value below the
    #     # projected_steps.
    #     mean_progress_reward = sigmoid(
    #         projected_steps,
    #         k=CONFIG["step_reward"]["k"],
    #         x0=CONFIG["step_reward"]["x0"],
    #         ymin=CONFIG["step_reward"]["ymin"],
    #         ymax=CONFIG["step_reward"]["ymax"],
    #     )
    # else:
    #     # We are going backwards
    #     mean_progress_reward = -sigmoid(
    #         -projected_steps,
    #         k=CONFIG["step_reward"]["k"],
    #         x0=CONFIG["step_reward"]["x0"],
    #         ymin=CONFIG["step_reward"]["ymin"],
    #         ymax=CONFIG["step_reward"]["ymax"],
    #     )

    _, difficulty, norm_difficulty = get_waypoint_difficulty(
        this_waypoint,
        params["waypoints"],
        skip_ahead=CONFIG["difficulty"]["skip-ahead"],
        look_ahead=CONFIG["difficulty"]["look-ahead"],
        max_val=CONFIG["difficulty"]["max"],
        min_val=CONFIG["difficulty"]["min"],
    )

    # Calculate difficulty-weighted target throttle and associated factor
    target_throttle = (norm_difficulty * throttle_range) + CONFIG["agent"]["speed"][
        "low"
    ]
    throttle_diff = abs(target_throttle - params["speed"])
    # Trottle_factor ranges from 1 to -1, where 1 is given to the target throttle
    throttle_factor = (2.0 * (1.0 - (throttle_diff / throttle_range))) - 1.0

    # Calculate weighted difficulty
    # weighted_difficulty = sigmoid(
    #     norm_difficulty,
    #     k=CONFIG["difficulty"]["weighting"]["k"],
    #     x0=CONFIG["difficulty"]["weighting"]["x0"],
    #     ymin=CONFIG["difficulty"]["weighting"]["ymin"],
    #     ymax=CONFIG["difficulty"]["weighting"]["ymax"],
    # )

    # Calculate novelty weight
    novelty = get_waypoint_importance(
        get_direction_change(this_waypoint, params["waypoints"]), CONFIG["histogram"]
    )
    novelty_weight = 1.0 + (novelty * (novelty_factor - 1.0))

    reward_type = CONFIG["reward_type"]

    if reward_type >= 0 and reward_type < 20:
        base_reward = mean_progress
    # elif reward_type >= 20 and reward_type < 40:
    #     base_reward = mean_progress_reward
    if base_reward < 0.0:
        # Everything else should be positive
        if mean_smoothness < 0.0:
            mean_smoothness = -mean_smoothness
        if throttle_factor < 0.0:
            throttle_factor = -throttle_factor

    if reward_type == 0:
        reward = float(
            novelty_weight * (mean_progress * (throttle_factor + mean_smoothness))
        )
    if reward_type == 1:
        reward = float(novelty_weight * mean_progress * mean_smoothness)
    if reward_type == 2:
        reward = float(novelty_weight * mean_progress * throttle_factor)
    if reward_type == 3:
        reward = float(novelty_weight * mean_progress)
    if reward_type == 4:
        reward = float(mean_progress * (throttle_factor + mean_smoothness))
    if reward_type == 5:
        reward = float(mean_progress * mean_smoothness)
    if reward_type == 6:
        reward = float(mean_progress * throttle_factor)
    if reward_type == 7:
        reward = float(mean_progress)

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for train & test logging
    print(
        f'MY_TRACE_LOG:{params["steps"]},{this_waypoint},{params["progress"]},{mean_progress},{params["speed"]},{params["steering_angle"]},{reward},{is_finished}'
    )

    return reward
