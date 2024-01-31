import math
import time

# Other globals
LAST_PROGRESS = 0.0
# caecer_loop
TRACK = {
    "length": 39.12,
    "difficulty": {
        "min": 0.00044293424911018286,
        "max": 0.1767213283925288,
        "ahead": 4,
    },
    "curves": [
        {"dir": "left", "start": 41, "cross": 47, "end": 60},
        {"dir": "left", "start": 78, "cross": 81, "end": 92},
    ],
}


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
    behind = get_prev_distinct_index(i, waypoints)
    ahead = get_next_distinct_index(i, waypoints)
    this_direction = get_direction(i, waypoints)
    diff0 = this_direction - get_direction(behind, waypoints)
    diff1 = get_direction(ahead, waypoints) - this_direction
    diff0 = math.atan2(math.sin(diff0), math.cos(diff0))
    diff1 = math.atan2(math.sin(diff1), math.cos(diff1))
    return (diff1 + diff0) / 2.0


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


def sigmoid(x, k=3.9, x0=0.6, ymin=0.0, ymax=1.2):
    """Parametrized sigmoid function as seen on:
       https://www.desmos.com/calculator/1c15zoa5b2

    Args:
        x (float): the input value
        k (float): _summary_
        x0 (float): _summary_
        ymin (float): _summary_
        ymax (float): _summary_

    Returns:
        float: the value of the sigmoid function at x
    """
    return ((ymax - ymin) / (1 + math.exp(-k * (x - x0)))) + ymin


def reward_function(params):
    this_waypoint = params["closest_waypoints"][0]

    # projected_steps is the estimated number of steps needed to
    # finish the track, divided by 100, so if the model finishes
    # the track in 200 steps, projected_steps should be ~2.0 on
    # average
    projected_steps = params["steps"] / params["progress"]

    # @350 steps, ~0.184 per step = 64.4 reward
    # @300 steps, ~0.25 per step = 75 reward
    # @250 steps, ~0.36 per step = 90 reward
    # @200 steps, ~0.563 per step = 112.6 reward
    # @150 steps, ~1 per step = 150 reward
    step_reward = 2.25 / (projected_steps**2)

    is_finished = 0
    if params["is_offtrack"]:
        is_finished = 1
        reward = 1e-5
    else:
        reward = float(2.0 * step_reward)

    if params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{params['speed']},{params['steering_angle']},{projected_steps * 100},{step_reward},{reward},{is_finished}"
    )

    return reward
