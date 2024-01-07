import math

LAST_WAYPOINT = 0.0
CUMULATIVE_REWARD = 0.0
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
    global LAST_WAYPOINT
    global CUMULATIVE_REWARD

    # This trace is needed for test logging
    print(f"MY_TRACE_LOG:{params['steps']},{params['progress']}")

    this_waypoint = params["closest_waypoints"][0]

    # For the first couple steps, just record the waypoint
    if params["steps"] < 2:
        print(f"MY_DEBUG_LOG: Resetting at waypoint {this_waypoint}")
        LAST_WAYPOINT = this_waypoint
        CUMULATIVE_REWARD = 0.0
        return float(0.0)

    # Waypoint hasn't changed, so no reward
    if this_waypoint == LAST_WAYPOINT:
        return float(0.0)

    # Calculate the direction change
    direction_change = abs(get_direction_change(this_waypoint, params["waypoints"]))
    reward = direction_change / CURVE_LIMITS["caecer_loop"]["max"]
    LAST_WAYPOINT = this_waypoint
    CUMULATIVE_REWARD += reward

    bonus = 0.0
    if params["progress"] == 100.0:
        # Reward speed at the finish line
        # ~50% additional reward for 200 steps
        # ~33% additional reward for 300 steps
        bonus = CUMULATIVE_REWARD / (params["steps"] / 100.0)

    if params["is_offtrack"] or params["progress"] == 100.0:
        print(f"MY_DEBUG_LOG: Trial ended with reward {CUMULATIVE_REWARD:0.4f}")

    return float(reward + bonus)
