import math
import time

TRACK_NAME = "caecer_gp"
STEP_K = -0.0045
STEP_X0 = 0.0
STEP_YMIN = 0.0
STEP_YMAX = 6
DIFFICULTY_FACTOR = 1.0

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


def get_difficulty(i, waypoints, max_val=1.0):
    """
    Get difficulty at waypoint i
    """
    difficulty = abs(get_direction_change(i, waypoints))
    next_idx = get_next_distinct_index(i, waypoints)
    dx = waypoints[next_idx][0] - waypoints[i][0]
    dy = waypoints[next_idx][1] - waypoints[i][1]
    d = math.sqrt(dx**2 + dy**2)
    return (difficulty / d) / max_val


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

    if params["steps"] <= 2:
        # Reset progress at the beginning of the episode
        LAST_PROGRESS = 0.0

    # Get the step progress
    step_progress = params["progress"] - LAST_PROGRESS
    LAST_PROGRESS = params["progress"]

    this_waypoint = params["closest_waypoints"][0]

    difficulty = get_difficulty(
        this_waypoint, params["waypoints"], max_val=1.7394709392167267
    )

    if step_progress == 0.0:
        projected_steps = 10000.0
    else:
        projected_steps = 100.0 / step_progress

    if step_progress >= 0.0:
        # We reward projected_steps based on each step's progress.
        # The sigmoid saturates the reward to a maximum value below the
        # projected_steps, which is ~370@290 steps in our case.
        step_reward = sigmoid(
            projected_steps,
            k=STEP_K,
            x0=STEP_X0,
            ymin=STEP_YMIN,
            ymax=STEP_YMAX,
        )
    else:
        # We are going backwards
        step_reward = -sigmoid(
            -projected_steps,
            k=-STEP_K,
            x0=STEP_X0,
            ymin=STEP_YMIN,
            ymax=STEP_YMAX,
        )

    reward = float(step_reward + (difficulty * DIFFICULTY_FACTOR))

    is_finished = 0
    if params["is_offtrack"] or params["progress"] == 100.0:
        is_finished = 1
        if params["is_offtrack"]:
            reward /= 1000.0

    # This trace is needed for test logging
    print(
        f"MY_TRACE_LOG:{params['steps']},{this_waypoint},{params['progress']},{params['speed']},{params['steering_angle']},{projected_steps},{difficulty},{reward},{is_finished}"
    )

    return reward
