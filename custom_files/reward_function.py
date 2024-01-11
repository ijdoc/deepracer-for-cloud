import math
import time

LAST_PROGRESS = 0.0
TRACKS = {
    "caecer_loop": {"length": 39.12, "min_angle": 0.0, "max_angle": 0.18297208942448917}
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


def reward_function(params):
    global LAST_PROGRESS

    # Reset progress at the beginning
    if params["steps"] <= 2:
        LAST_PROGRESS = 0.0

    # Get difficulty as a number from 0.0 to 1.0 with an
    # added offset to avoid null scores when driving straight
    this_waypoint = params["closest_waypoints"][0]
    difficulty = (
        abs(get_direction_change(this_waypoint, params["waypoints"]))
        / TRACKS["caecer_loop"]["max_angle"]
    ) + 0.1

    # Get the step progress
    step_progress = params["progress"] - LAST_PROGRESS
    LAST_PROGRESS = params["progress"]

    # Encourage good technique at the tightest curve
    if this_waypoint >= 42 and this_waypoint <= 44:
        reward = 1e-5
        if params["steering_angle"] == 0.0:  # go straight
            reward += 1.0
    elif this_waypoint >= 45 and this_waypoint <= 64:
        reward = 1e-5
        if this_waypoint <= 56:
            if params["speed"] <= 1.1:  # go slow
                reward += 1.0
            if params["steering_angle"] != 0.0:  # not straight
                reward += 1.0
        if this_waypoint >= 49:
            if params["is_left_of_center"]:
                reward += 1.0
    else:
        # Weight step progress to favour faster speeds
        weighted_progress = 16 * (step_progress**3)
        reward = difficulty * weighted_progress

    reward = float(reward)

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{step_progress},{difficulty},{reward},{is_finished}"
    )

    return reward
