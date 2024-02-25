import math
import time

# TRACK_NAME = "caecer_loop"
REWARD_TYPE = "sigmoid"
# caecer_loop
MODEL = {"max": 3.8, "min": 1.3}
COACH = {
    # "length": 39.12,
    # "change": {"ahead": 4, "max": 0.7068853135701152, "min": 0.0017717369964407315},
    "curves": [
        {
            "dir": "left",  # direction of the curve
            "break_start": 41,  # waypoint index to start braking
            "steer_start": 42,  # waypoint index to start steering
            "steer_end": 46,  # waypoint index to stop braking & steering
            "apex": 55,  # waypoint index to reach the apex
            "exit": 60,  # waypoint index to exit the curve
            "break_throttle": 0.3,  # throttle is normalized to 0-1
            "min_steer": 10,  # actual degrees
        },
        {
            "dir": "left",
            "break_start": 78,
            "steer_start": 78,
            "steer_end": 80,
            "apex": 87,
            "exit": 92,
            "break_throttle": 0.8,
            "min_steer": 4,
        },
    ],
}

# Other globals
LAST_PROGRESS = 0.0


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

    if params["steps"] <= 2:
        # Reset progress at the beginning of the episode
        LAST_PROGRESS = 0.0

    # Get the step progress
    step_progress = params["progress"] - LAST_PROGRESS
    LAST_PROGRESS = params["progress"]

    this_waypoint = params["closest_waypoints"][0]

    # projected_steps is 0.01 of steps needed to finish the track,
    # so if the model finishes in 200 steps, projected_steps
    # should be 2.0 on average
    if step_progress == 0.0:
        projected_steps = 100  # 10000 actual steps
    else:
        projected_steps = 1 / step_progress

    if step_progress >= 0.0:
        # We reward projected_steps based on each step's progress.
        # The sigmoid saturates outliers to a reward equivalent to the target
        # projected_steps, which is 1.5 (or 150 actual steps) in our case.
        # @350 steps, ~1 reward
        # @300 steps, ~3 reward
        # @250 steps, ~13 reward
        # @200 steps, ~50 reward
        # @150 steps, ~150 reward
        # @100 steps, ~230 reward
        # @50 steps, ~150 reward
        step_reward = sigmoid(projected_steps, k=-3.3, x0=1.25, ymin=0.0, ymax=3.3)
    else:
        # We are going backwards
        step_reward = -sigmoid(-projected_steps, k=-3.3, x0=1.25, ymin=0.0, ymax=3.3)

    reward = 2.0 * step_reward

    for curve in COACH["curves"]:
        # Process curve exceptions
        if this_waypoint >= curve["break_start"] and this_waypoint <= curve["exit"]:
            reward = 1e-5
            throttle_fraction = (params["speed"] - MODEL["min"]) / (
                MODEL["max"] - MODEL["min"]
            )
            if this_waypoint <= curve["steer_end"]:
                # Reward braking ahead of curve
                spread = -50  # ~20% of the throttle range
            else:
                # Reward speeding out of the curve
                spread = 50  # ~20% of the throttle range
            reward += sigmoid(
                throttle_fraction,
                k=spread,
                x0=curve["break_throttle"],
                ymin=0.0,
                ymax=1.0,
            )
            # Reward steering ahead of curve
            if (
                this_waypoint >= curve["steer_start"]
                and this_waypoint <= curve["steer_end"]
            ):
                reward += sigmoid(
                    params["steering_angle"],
                    k=5, # Sigmoid spread is ~2.0 degrees
                    x0=curve["min_steer"],
                    ymin=0.0,
                    ymax=2.0,
                )
            # Reward not steering out of the curve
            elif this_waypoint > curve["steer_end"]:
                reward += sigmoid(
                    params["steering_angle"],
                    k=-5, # Sigmoid spread is ~2.0 degrees
                    x0=3.0,
                    ymin=0.0,
                    ymax=2.0,
                )

            # Reward using full track width. Here we calculate the track position as a
            # range from 0 - 1 where 1 is the outer edge of the turn and 0 is the apex
            if curve["dir"] == "left":
                if params["is_left_of_center"]:
                    # position is 0 to 0.5
                    position = 0.5 - (
                        params["distance_from_center"] / params["track_width"]
                    )
                else:
                    # position is 0.5 to 1
                    position = 0.5 + (
                        params["distance_from_center"] / params["track_width"]
                    )
            else:
                if params["is_left_of_center"]:
                    # position is 0.5 to 1
                    position = 0.5 + (
                        params["distance_from_center"] / params["track_width"]
                    )
                else:
                    # position is 0 to 0.5
                    position = 0.5 - (
                        params["distance_from_center"] / params["track_width"]
                    )

            if this_waypoint <= curve["apex"]:
                # Must go from 1 to 0 (outside to apex)
                expected = (this_waypoint - curve["apex"]) / (
                    curve["break_start"] - curve["apex"]
                )
            else:
                # Must go from 0 to 0.5 (apex to center line)
                expected = (
                    (this_waypoint - curve["apex"]) / (curve["exit"] - curve["apex"])
                ) / 2.0
            reward += 1.0 - abs(position - expected)

    reward = float(reward)

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1
        if params["is_offtrack"]:
            reward /= 1000.0

    # This trace is needed for test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{params['speed']},{params['steering_angle']},{step_progress},{projected_steps * 100},{reward},{is_finished}"
    )

    return reward
