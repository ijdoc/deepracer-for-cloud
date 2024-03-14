import math
import time

TRACK = {
    "name": "caecer_gp",
    "waypoint_count": 231,
    "difficulty": {"max": 0.7383608251836872, "min": 0.004665479885233692},
    # Cumulative max is ~150@400 steps in our case
    "step_progress": {
        "ymax": 0.5,
        "ymin": 0.0,
        "k": -0.01,
        "x0": 510,
    },
}

# Other globals
LAST_PROGRESS = 0.0
TRIAL_START = 0


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
    return aggregate_change, (difficulty - min_val) / (max_val - min_val)


def get_target_heading(i, waypoints):
    """
    Get heading at waypoint i (in radians)
    """
    direction = get_direction(i, waypoints)
    dir_change = get_direction_change(i, waypoints)
    bias = dir_change * 2.0
    return direction + (bias * 0.6)


def subtract_angles_rad(a, b):
    diff = a - b
    return math.atan2(math.sin(diff), math.cos(diff))


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
    global LAST_PROGRESS
    global TRIAL_START

    this_waypoint = params["closest_waypoints"][0]

    if params["steps"] <= 2:
        # Reset progress at the beginning of the episode
        LAST_PROGRESS = 0.0
        TRIAL_START = this_waypoint

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

    _, difficulty = get_waypoint_difficulty(
        this_waypoint,
        params["waypoints"],
        look_ahead=3,
        max_val=TRACK["difficulty"]["max"],
        min_val=TRACK["difficulty"]["min"],
    )
    heading = get_target_heading(this_waypoint, params["waypoints"])
    heading_diff = abs(subtract_angles_rad(heading, math.radians(params["heading"])))
    heading_reward = sigmoid(
        heading_diff,
        k=-4.0 * math.pi,
        x0=math.pi / 6,  # 30 degrees
        ymin=0.0,
        ymax=4.0 * step_reward,
    )
    reward = float((step_reward * (1.0 - difficulty)) + (heading_reward * difficulty))

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1
        if params["is_offtrack"]:
            if reward > 0.0:
                reward = float(-reward)

    # This trace is needed for test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{projected_steps},{step_reward},{math.degrees(heading_diff)},{heading_reward},{difficulty},{reward},{is_finished}"
    )

    return reward
