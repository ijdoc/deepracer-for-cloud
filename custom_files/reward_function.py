import math
import time

LAST_TIME = 0.0
LAST_PERCENT = 0.0
LAST_SPEED = 0.0
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
    global LAST_TIME
    global LAST_PERCENT
    global LAST_SPEED

    # Get difficulty as a number from 0.0 to 1.0
    this_waypoint = params["closest_waypoints"][0]
    difficulty = (
        abs(get_direction_change(this_waypoint, params["waypoints"]))
        / TRACKS["caecer_loop"]["max_angle"]
    )

    this_percent = math.floor(params["progress"])
    if this_percent > LAST_PERCENT:
        # Calculate speed in m/s
        now = time.time()
        LAST_SPEED = 0.01 * TRACKS["caecer_loop"]["length"] / (now - LAST_TIME)
        LAST_PERCENT = this_percent
        LAST_TIME = now

    # Reset speed tracking at the beginning
    if params["steps"] <= 2:
        LAST_SPEED = 0.0
        LAST_TIME = time.time()
        LAST_PERCENT = 0.0

    # Encourage good behavior at the curve
    # factor = 1.0
    # if this_waypoint >= 40 and this_waypoint <= 42:
    #     if not params["is_left_of_center"]:  # correct side of track
    #         factor += 0.25
    # if this_waypoint >= 41 and this_waypoint <= 45:
    #     if params["speed"] <= 1.0:  # breaking before & start of curve
    #         factor += 0.25
    # if this_waypoint >= 41 and this_waypoint <= 43:
    #     if params["steering_angle"] == 0.0:  # not turning
    #         factor += 0.25

    # reward *= factor

    # Bonus reward for completing the track
    bonus = 0.0
    if params["progress"] == 100.0:
        bonus = 14.0 + (4e6 / (params["steps"] ** 2))

    reward = float((difficulty * LAST_SPEED) + bonus)

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1

    # This trace is needed for test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']:0.1f},{LAST_SPEED},{difficulty},{reward},{is_finished}"
    )

    return reward
