import math

LAST_PROGRESS = 0.0
CURVE_LIMITS = {
    "caecer_loop": {"min": 2.220446049250313e-16, "max": 0.25648735747756357}
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
    """
    Progress with track difficulty
    """

    global LAST_PROGRESS
    # look_ahead = 3
    difficulty_constant = 4.0

    # Obtain difficulty
    curve = get_direction_change(params["closest_waypoints"][0], params["waypoints"])
    # Progressing through the hardest curve should be 1 + difficulty_constant
    # times more rewarding than progressing through a straight
    difficulty_factor = difficulty_constant * (
        abs(curve) / CURVE_LIMITS["caecer_loop"]["max"]
    )
    difficulty = 1 + float(difficulty_factor * abs(curve))
    progress = params["progress"]

    # point_ahead = params['closest_waypoints'][1]
    # for _ in range(look_ahead):
    #     point_ahead = get_next_distinct_index(point_ahead, params['waypoints'])

    # Base reward
    step_progress = progress - LAST_PROGRESS
    LAST_PROGRESS = progress

    if step_progress < 0:
        return float(0.00001)

    # Compensate for reduced number of steps by weighting progress with a
    # function that grows faster than f(x) = x after a certain point
    weighted_progress = float((5.0 * step_progress) ** 1.75)
    print(f"MY_TRACE_LOG:{params['steps']},{progress}")

    return float(weighted_progress * difficulty)
