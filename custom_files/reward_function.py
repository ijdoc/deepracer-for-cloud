import math
import time

TRACK = {
    "name": "caecer_gp",
    "waypoint_count": 231,
    "difficulty": {
        "look-ahead": 3,
        "max": 0.9155687800476376,
        "min": 0.007010262368781583,
    },
    "histogram": {
        "counts": [9, 15, 14, 12, 35, 47, 41, 31, 12, 15],
        "weights": [
            1.0,
            0.5053,
            0.5583,
            0.6908,
            0.0812,
            0.0,
            0.0347,
            0.1222,
            0.6908,
            0.5053,
        ],
        "edges": [
            -0.2513188520107108,
            -0.1999984713513993,
            -0.14867809069208782,
            -0.09735771003277632,
            -0.04603732937346483,
            0.005283051285846663,
            0.056603431945158156,
            0.10792381260446965,
            0.15924419326378114,
            0.21056457392309263,
            0.26188495458240413,
        ],
    },
    "step_progress": {"ymax": 0.625, "ymin": 0.0, "k": -0.015, "x0": 400},
}

# Other globals
LAST_PROGRESS = 0.0
importance_factor = max(TRACK["histogram"]["counts"]) / min(
    TRACK["histogram"]["counts"]
)


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


def get_waypoint_difficulty(i, waypoints, look_ahead=1, max_val=1.0, min_val=0.0):
    """
    Get difficulty at waypoint i, calculated as the normalized amount of direction change.
    """
    next_idx = i
    aggregate_change = get_direction_change(i, waypoints)
    for _ in range(look_ahead - 1):
        next_idx = get_next_distinct_index(next_idx, waypoints)
        aggregate_change += get_direction_change(next_idx, waypoints)
    difficulty = abs(aggregate_change)
    normalized_difficulty = (difficulty - min_val) / (max_val - min_val)
    # Push limits away from 0.5
    weighted_difficulty = sigmoid(
        normalized_difficulty, k=10, x0=0.5, ymin=0.0, ymax=1.0
    )
    return aggregate_change, weighted_difficulty


def get_waypoint_importance(change, histogram):
    """
    Get waypoint weight based on how common the direction change is in the entire track.
    This can be used to give more weight to waypoints that are less likely to occur during
    training, and therefore are more important to learn.
    """
    for j in range(len(histogram["weights"])):
        if change >= histogram["edges"][j] and change < histogram["edges"][j + 1]:
            return histogram["weights"][j]
    return histogram["weights"][j]  # when change is equal to the last edge


def get_target_heading(i, waypoints, delay=1, look_ahead=1, min_val=0.0, max_val=1.0):
    """
    Get heading at waypoint i (in radians)
    """
    for _ in range(delay):
        i = get_prev_distinct_index(i, waypoints)
    direction = get_direction(i, waypoints)
    change, difficulty = get_waypoint_difficulty(
        i, waypoints, look_ahead=look_ahead, min_val=min_val, max_val=max_val
    )
    return direction + (change * 2.5)


def subtract_angles_rad(a, b):
    diff = a - b
    return math.atan2(math.sin(diff), math.cos(diff))


def reward_function(params):
    global LAST_PROGRESS

    this_waypoint = params["closest_waypoints"][0]

    if params["steps"] <= 2:
        # Reset progress at the beginning of the episode
        LAST_PROGRESS = 0.0

    # Get the step progress
    step_progress = params["progress"] - LAST_PROGRESS
    LAST_PROGRESS = params["progress"]

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
            k=TRACK["step_progress"]["k"],
            x0=TRACK["step_progress"]["x0"],
            ymin=TRACK["step_progress"]["ymin"],
            ymax=TRACK["step_progress"]["ymax"],
        )
    else:
        # We are going backwards
        step_reward = -sigmoid(
            -projected_steps,
            k=TRACK["step_progress"]["k"],
            x0=TRACK["step_progress"]["x0"],
            ymin=TRACK["step_progress"]["ymin"],
            ymax=TRACK["step_progress"]["ymax"],
        )

    dir_change, difficulty = get_waypoint_difficulty(
        this_waypoint,
        params["waypoints"],
        look_ahead=TRACK["difficulty"]["look-ahead"],
        min_val=TRACK["difficulty"]["min"],
        max_val=TRACK["difficulty"]["max"],
    )
    importance = get_waypoint_importance(
        get_direction_change(this_waypoint, params["waypoints"]), TRACK["histogram"]
    )
    heading = get_target_heading(
        this_waypoint,
        params["waypoints"],
        delay=5,
        look_ahead=TRACK["difficulty"]["look-ahead"],
        min_val=TRACK["difficulty"]["min"],
        max_val=TRACK["difficulty"]["max"],
    )
    heading_diff = abs(subtract_angles_rad(heading, math.radians(params["heading"])))
    # heading_reward max should be at least the same as step_reward
    heading_reward = math.cos(heading_diff) * step_reward
    importance_weight = (importance * (importance_factor - 1.0)) + 1.0
    reward = float(
        importance_weight
        * ((step_reward * (1.0 - difficulty)) + (heading_reward * difficulty))
    )

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{step_reward},{math.degrees(heading_diff)},{heading_reward},{difficulty},{importance_weight},{reward},{is_finished}"
    )

    return reward
