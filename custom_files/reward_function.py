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
    "track": "dubai_open_ccw",
    "reward_type": 5,
    "waypoint_count": 138,
    "difficulty": {
        "skip-ahead": 0,
        "look-ahead": 0,
        "max": 0.5466957475920301,
        "min": 0.00011722494281351734,
        "weighting": {"ymax": 1.0, "ymin": 0.0, "k": 30, "x0": 0.5},
    },
    "histogram": {
        "counts": [3, 3, 3, 2, 4, 63, 19, 14, 11, 7, 7, 2],
        "weights": [
            0.6557,
            0.6557,
            0.6557,
            1.0,
            0.4836,
            0.0,
            0.0759,
            0.1148,
            0.155,
            0.2623,
            0.2623,
            1.0,
        ],
        "edges": [
            -0.5316530778825277,
            -0.44179067575964787,
            -0.35192827363676804,
            -0.26206587151388827,
            -0.17220346939100845,
            -0.08234106726812862,
            0.007521334854751149,
            0.09738373697763103,
            0.1872461391005108,
            0.27710854122339057,
            0.36697094334627045,
            0.4568333454691502,
            0.5466957475920301,
        ],
    },
    "step_reward": {"ymax": 1, "ymin": 0.0, "k": -0.05, "x0": 200},
    "agent": {
        "steering_angle": {"high": 30.0, "low": -30.0},
        "speed": {"high": 2.4, "low": 1.2},
    },
}
LAST_PROGRESS = 0.0
PROGRESS_BUFFER = None
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


def get_waypoint_novelty(change, histogram):
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
    global LAST_PROGRESS
    global PROGRESS_BUFFER
    global LAST_THROTTLE
    global LAST_STEERING

    this_waypoint = params["closest_waypoints"][0]

    if params["steps"] <= 2:
        # Reset progress at the beginning of the episode
        LAST_PROGRESS = 0.0
        PROGRESS_BUFFER = CircularBuffer(CONFIG["difficulty"]["look-ahead"])
        LAST_THROTTLE = 0.0
        LAST_STEERING = 0.0

    # Get the step progress
    step_progress = params["progress"] - LAST_PROGRESS
    PROGRESS_BUFFER.add_value(step_progress)
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

    # Smoothness ranges from -1 to 1, where 1 is the smoothest
    smoothness = 2.0 * (0.5 - agent_change)

    if step_progress == 0.0:
        projected_steps = 10000.0
    else:
        projected_steps = 100.0 / step_progress

    if step_progress >= 0.0:
        # We reward projected_steps based on each step's progress.
        # The sigmoid saturates the reward to a maximum value below the
        # projected_steps.
        step_reward = sigmoid(
            projected_steps,
            k=CONFIG["step_reward"]["k"],
            x0=CONFIG["step_reward"]["x0"],
            ymin=CONFIG["step_reward"]["ymin"],
            ymax=CONFIG["step_reward"]["ymax"],
        )
    else:
        # We are going backwards
        step_reward = -sigmoid(
            -projected_steps,
            k=CONFIG["step_reward"]["k"],
            x0=CONFIG["step_reward"]["x0"],
            ymin=CONFIG["step_reward"]["ymin"],
            ymax=CONFIG["step_reward"]["ymax"],
        )

    if mean_progress >= 0.0:
        # We reward projected_steps based on each step's progress.
        # The sigmoid saturates the reward to a maximum value below the
        # projected_steps.
        mean_progress_reward = sigmoid(
            projected_steps,
            k=CONFIG["step_reward"]["k"],
            x0=CONFIG["step_reward"]["x0"],
            ymin=CONFIG["step_reward"]["ymin"],
            ymax=CONFIG["step_reward"]["ymax"],
        )
    else:
        # We are going backwards
        mean_progress_reward = -sigmoid(
            -projected_steps,
            k=CONFIG["step_reward"]["k"],
            x0=CONFIG["step_reward"]["x0"],
            ymin=CONFIG["step_reward"]["ymin"],
            ymax=CONFIG["step_reward"]["ymax"],
        )

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
    # Trottle_factor ranges from -1 to 1, where 1 is given to the target throttle
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
    novelty = get_waypoint_novelty(
        get_direction_change(this_waypoint, params["waypoints"]), CONFIG["histogram"]
    )
    novelty_weight = 1.0 + (novelty * (novelty_factor - 1.0))

    reward_type = CONFIG["reward_type"]

    if reward_type >= 0 and reward_type < 10:
        base_reward = step_reward
    elif reward_type >= 10 and reward_type < 20:
        base_reward = mean_progress_reward
    elif reward_type >= 20 and reward_type < 30:
        base_reward = mean_progress
    elif reward_type >= 30 and reward_type < 40:
        base_reward = step_progress
    if base_reward < 0.0:
        # Everything else should be positive
        if smoothness < 0.0:
            smoothness = -smoothness
        if throttle_factor < 0.0:
            throttle_factor = -throttle_factor

    if reward_type == 0:
        reward = float(step_reward)
    if reward_type == 1:
        reward = float(step_reward * novelty_weight)
    if reward_type == 2:
        reward = float(step_reward * novelty_weight * smoothness)
    if reward_type == 3:
        reward = float(step_reward * novelty_weight * throttle_factor)
    if reward_type == 8:
        reward = float(step_reward * smoothness)
    if reward_type == 9:
        reward = float(step_reward * throttle_factor)
    if reward_type == 10:
        reward = float(mean_progress_reward)
    if reward_type == 11:
        reward = float(mean_progress_reward * novelty_weight)
    if reward_type == 12:
        reward = float(mean_progress_reward * novelty_weight * smoothness)
    if reward_type == 13:
        reward = float(mean_progress_reward * novelty_weight * throttle_factor)
    if reward_type == 20:
        reward = float(mean_progress)
    if reward_type == 21:
        reward = float(mean_progress * novelty_weight)
    if reward_type == 22:
        reward = float(mean_progress * novelty_weight * smoothness)
    if reward_type == 23:
        reward = float(mean_progress * novelty_weight * throttle_factor)
    if reward_type == 30:
        reward = float(step_progress)
    if reward_type == 31:
        reward = float(step_progress * novelty_weight)
    if reward_type == 32:
        reward = float(step_progress * novelty_weight * smoothness)
    if reward_type == 33:
        reward = float(step_progress * novelty_weight * throttle_factor)
    if reward_type == 38:
        reward = float(step_progress * smoothness)
    if reward_type == 39:
        reward = float(step_progress * throttle_factor)

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for train & test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{step_reward},{reward},{is_finished}"
    )

    return reward
