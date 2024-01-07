import math

LAST_PROGRESS = 0.0
CURVE_LIMITS = {"caecer_loop": {"min": 0.0, "max": 0.18297208942448917}}


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

    print(f"MY_TRACE_LOG:{params['steps']},{params['progress']}")
    # look_ahead = 3
    # point_ahead = params['closest_waypoints'][1]
    # for _ in range(look_ahead):
    #     point_ahead = get_next_distinct_index(point_ahead, params['waypoints'])

    if params["steps"] <= 1:
        # Trial started
        LAST_PROGRESS = 0.0
        return 0.0

    # Base reward
    step_progress = params["progress"] - LAST_PROGRESS
    LAST_PROGRESS = params["progress"]

    # Encourage the agent to stay on the right side of the track
    if (
        not params["is_left_of_center"]
        and params["closest_waypoints"][1] >= 40
        and params["closest_waypoints"][1] <= 48
    ):
        step_progress *= 10.0

    # Encourage the agent to brake when approaching the curve
    if (
        params["speed"] <= 1.0
        and params["closest_waypoints"][1] >= 41
        and params["closest_waypoints"][1] <= 48
    ):
        step_progress *= 10.0

    # Encourage the brake straight before the curve
    if (
        params["steering_angle"] == 0.0
        and params["closest_waypoints"][1] >= 41
        and params["closest_waypoints"][1] <= 46
    ):
        step_progress *= 10.0

    bonus = 0.0
    if params["is_offtrack"] or params["progress"] == 100.0:
        bonus = 1000 * params["progress"] / params["steps"]

    # Obtain difficulty
    difficulty_factor = 22.0
    curve = get_direction_change(params["closest_waypoints"][0], params["waypoints"])
    difficulty = 1 + float(difficulty_factor * abs(curve))

    return float((step_progress * difficulty) + bonus)
