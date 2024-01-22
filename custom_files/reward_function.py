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

    # Max difficulty is 2.5 and about 5x the min difficulty
    this_waypoint = params["closest_waypoints"][0]
    difficulty = (
        2.0
        * abs(get_direction_change(this_waypoint, params["waypoints"]))
        / TRACKS["caecer_loop"]["max_angle"]
    ) + 0.5

    # Get the step progress
    step_progress = params["progress"] - LAST_PROGRESS
    LAST_PROGRESS = params["progress"]
    # Weight step progress to favour faster speeds
    # weighted_progress = 1.7 * (step_progress**2)
    weighted_progress = 10.0 * (1 - (1 / math.sqrt(1 + (0.5 * (step_progress**2)))))

    coach_factor = 1.0
    is_finished = 0
    if params["is_offtrack"]:
        is_finished = 1
        reward = 1e-5
    else:
        reward = difficulty * weighted_progress
        if this_waypoint >= 45 and this_waypoint <= 63:
            if this_waypoint <= 56:
                if params["speed"] <= 1.3:  # go slow FIXME (depends on model)
                    coach_factor += 1.0
                if params["steering_angle"] != 0.0:  # turn
                    coach_factor += 1.0
            if this_waypoint >= 49:
                if params["is_left_of_center"]:  # keep left
                    coach_factor += 2.0

    reward = float(reward * coach_factor)

    action = -1
    if params["steering_angle"] == -5:
        action = 0
    elif params["steering_angle"] == 0.0:
        action = 1
    elif params["steering_angle"] == 7.5:
        action = 2
    elif params["steering_angle"] == 15 and params["speed"] == 2.6:
        action = 3
    elif params["steering_angle"] == 15 and params["speed"] == 1.3:
        action = 4
    elif params["steering_angle"] == 25:
        action = 5

    if params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{step_progress},{difficulty},{reward},{is_finished},{action}"
    )

    return reward
