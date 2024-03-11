import math
import time

TRACK = {
    "name": "caecer_gp",
    "waypoint_count": 231,
    "importance": {
        "histogram": {
            "weights": [
                0.6377,
                0.3478,
                0.2236,
                0.2754,
                0.2754,
                0.6377,
                0.1107,
                0.3478,
                1.0,
                0.0942,
                0.0489,
                0.0338,
                0.0,
                0.0409,
                0.0217,
                0.058,
                0.0409,
                0.058,
                0.6377,
                0.1848,
                0.3478,
                0.1304,
                1.0,
            ],
            "edges": [
                -0.2513188520107108,
                -0.22900564302840146,
                -0.2066924340460921,
                -0.18437922506378276,
                -0.16206601608147342,
                -0.13975280709916407,
                -0.11743959811685473,
                -0.09512638913454538,
                -0.07281318015223603,
                -0.05049997116992669,
                -0.028186762187617342,
                -0.005873553205307996,
                0.01643965577700135,
                0.03875286475931072,
                0.06106607374162004,
                0.08337928272392936,
                0.10569249170623873,
                0.1280057006885481,
                0.15031890967085743,
                0.17263211865316674,
                0.19494532763547612,
                0.2172585366177855,
                0.2395717456000948,
                0.26188495458240413,
            ],
        }
    },
    "difficulty": {"max": 0.26188495458240413, "min": 0.0},
    # Cumulative max is ~150@400 steps in our case
    "step_progress": {
        "ymax": 0.5,
        "ymin": 0.0,
        "k": -0.01,
        "x0": 510,
    },
    # These arrays can be obtained by running training for two epochs
    "trial_starts": [
        0,
        4,
        14,
        24,
        30,
        36,
        51,
        56,
        60,
        73,
        89,
        98,
        111,
        126,
        148,
        167,
        181,
        193,
        199,
        218,
    ],
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


def get_waypoint_difficulty(i, waypoints):
    """
    Get difficulty at waypoint i, calculated as the normalized amount of direction change.
    """
    difficulty = abs(get_direction_change(i, waypoints))
    return (difficulty - TRACK["difficulty"]["min"]) / (
        TRACK["difficulty"]["max"] - TRACK["difficulty"]["min"]
    )


def get_waypoint_importance(i, waypoints):
    """
    Get waypoint weight based on how common the direction change is in the entire track.
    This can be used to give more weight to waypoints that are less likely to occur during
    training, and therefore are more important to learn.
    """
    histogram = TRACK["importance"]["histogram"]
    for j in range(len(histogram["weights"])):
        change = get_direction_change(i, waypoints)
        if change >= histogram["edges"][j] and change < histogram["edges"][j + 1]:
            return histogram["weights"][j]
    return histogram["weights"][j]  # when change is equal to the last edge


def get_waypoint_batch_rank(i, trial_start, factor):
    """
    Get a waypoint progress rank based on the order in which the agent will encounter it.
    This can be used to give more weight to waypoints that occur later in a trial, and are
    therefore more important to reach.
    """
    starts = TRACK["trial_starts"]
    idx = starts.index(trial_start)
    # Last start limit should be the end of the track
    if idx + 1 == len(starts):
        trial_end = TRACK["waypoint_count"] - 1
    else:
        trial_end = starts[idx + 1]
    if i < trial_start:
        # Assume we crossed the finish line, so loop the count.
        i += trial_end
    batch_progress = (i - trial_start) / (trial_end - trial_start)
    if batch_progress > 1.0:
        rank = 1.0
    else:
        rank = batch_progress
    return 1.0 + (rank * (factor - 1.0))


def get_waypoint_progress_rank(i, trial_start, factor):
    """
    Get a waypoint rank based on the progress of the agent. This can be used to encourage
    the agent to remain in the track, since the reward will be higher for waypoints that
    are closer to the current trial's finish line.
    """
    if i < trial_start:
        # Assume we crossed the finish line, so loop the count.
        i += TRACK["waypoint_count"]
    trial_progress = (i - trial_start) / (TRACK["waypoint_count"] - 1)
    return 1.0 + (trial_progress * (factor - 1.0))


def gaussian(x, a, b, c):
    """_summary_

    Args:
        x (float): the input value
        a (float): height of the curve's peak
        b (float): position of the center of the peak
        c (float): width of the “bell”

    Returns:
        float: the value of the gaussian function at x
    """
    return a * math.exp(-((x - b) ** 2) / (2 * c**2))


def wrapped_bell_curve(x, center=0, width=90):
    """
    Generates a bell-like curve using the cosine function, centered at a specified point,
    with a specified 'width' determining the spread of the bell curve. The curve is adjusted
    to have a maximum of 1.0 and a minimum of 0.0.

    Parameters:
    - x: The input angle in degrees for which to evaluate the bell curve.
    - center: The center of the bell curve on the -180 to 180 degree axis.
    - width: The 'width' of the bell curve, affecting its spread.

    Returns:
    - The value of the bell-like curve at the given input angle, adjusted to range from 0 to 1.
    """
    # Convert angles from degrees to radians
    x_rad = math.radians(x)
    center_rad = math.radians(center)
    width_rad = math.radians(width)

    # Adjust for wrapping
    diff = math.atan2(math.sin(x_rad - center_rad), math.cos(x_rad - center_rad))

    # Ignore values beyond the main cycle
    if diff > width_rad or diff < -width_rad:
        return 0.0

    # Use the cosine function to create a bell-like curve that wraps around
    cosine_value = math.cos((diff * math.pi) / width_rad)

    # Adjust the output to have a range from 0 to 1
    adjusted_value = (cosine_value + 1.0) / 2.0

    return adjusted_value


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

    difficulty = get_waypoint_difficulty(
        this_waypoint,
        params["waypoints"],
    )
    importance = get_waypoint_importance(this_waypoint, params["waypoints"])
    factor = (importance + difficulty) / 2.0
    reward = float(step_reward * (1.0 + (3.0 * factor)))

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1
        if params["is_offtrack"]:
            if reward > 0.0:
                reward = float(-reward)

    # This trace is needed for test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{projected_steps},{step_reward},{importance},{difficulty},{factor},{reward},{is_finished}"
    )

    return reward
