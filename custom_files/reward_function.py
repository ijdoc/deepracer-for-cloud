import math

LAST_PROGRESS   = 0.0
# Direction change ranges from -0.40174577800358763 to 0.32913860551437557
# Expected ideal location range: (-0.23477351610260344, 0.21254900557465373)


def get_next_distinct_index(i, waypoints):
    '''
    Get the next distinct waypoint index in a path, after i (skip duplicate waypoints)
    '''
    ahead = i + 1
    if i == len(waypoints) - 1:
        ahead = 0
    if waypoints[i][0] == waypoints[ahead][0] and waypoints[i][1] == waypoints[ahead][1]:
        return get_next_distinct_index(ahead, waypoints)
    return ahead

def get_prev_distinct_index(i, waypoints):
    '''
    Get the previous distinct waypoint index in a path, before i (skip duplicate waypoints)
    '''
    prev = i - 1
    if i == 0:
        prev = len(waypoints) - 1
    if waypoints[i][0] == waypoints[prev][0] and waypoints[i][1] == waypoints[prev][1]:
        return get_prev_distinct_index(prev, waypoints)
    return prev

def get_direction(i, waypoints):
    '''
    Get direction at waypoint i
    '''
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
    '''
    Progress with track difficulty, steer consistency and speed bias bonuses
    '''
    

    global LAST_PROGRESS
    look_ahead = 3

    #Obtain difficulty
    curve      = get_direction_change(params['closest_waypoints'][0], params['waypoints'])
    difficulty = 1 + float(5 * abs(curve))
    progress   = params['progress']
    distance_from_center = params['distance_from_center'] / (params['track_width']/2)
    if params['is_left_of_center']:
        distance_from_center = -distance_from_center

    point_ahead = params['closest_waypoints'][1]
    for _ in range(look_ahead):
        point_ahead = get_next_distinct_index(point_ahead, params['waypoints'])

    loc_factor = (get_direction_change(point_ahead, params['waypoints'])
                  - get_direction_change(get_prev_distinct_index(point_ahead, params['waypoints']), params['waypoints'])) / 0.2
    #Base reward
    step_progress   = progress - LAST_PROGRESS
    # print(f"Curr: {params['closest_waypoints'][0]} Ahead: {point_ahead} Loc:", "{:4.1f}".format(loc_factor))
    print("Error", "{:4.1f}".format(abs(distance_from_center - loc_factor)), "Real:","{:4.1f}".format(distance_from_center), "Ideal:", "{:4.1f}".format(loc_factor))
    # print(f"Pos: {params['closest_waypoints'][0]} Change: {curve}")
    # if abs(curve) < 0.075:
    curve_ahead = get_direction_change(point_ahead, params['waypoints'])
    # print(f"Curve: {-curve} Ahead: {-curve_ahead} Total: {len(params['waypoints'])}")
    # if abs(curve_ahead) < 0.75:
    #     print("NOOP")
    # else:
    #     if curve_ahead < 0:
    #         print ("BE LEFT")
    #     else:
    #         print ("BE RIGHT")
    # print(f"Difficulty: {difficulty} Step: {step_progress}")
    LAST_PROGRESS   = progress

    if step_progress < 0:
        # max = -1.0
        # min = 1.0
        # for point in range(len(params['waypoints'])):
        #     change = get_direction_change(point, params['waypoints']) - get_direction_change(get_prev_distinct_index(point, params['waypoints']), params['waypoints'])
        #     if change > max:
        #         max = change
        #     if change < min:
        #         min = change
        # print(f'Expected location range: ({min}, {max})')
        # Episode reset, no reward
        return float(0.0)

    return float(step_progress * difficulty * (1 - abs(distance_from_center - loc_factor) / 3.0))