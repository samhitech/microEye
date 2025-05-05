import numpy as np


def map_range(value, args):
    '''
    Map a value from one range to another.

    ... (rest of the map_range function docstring)
    '''
    old_min, old_max, new_min, new_max = args
    return (new_min +
            (new_max - new_min) * (value - old_min) / (old_max - old_min))

def dz_scaled_radial(stick_input, deadzone):
    '''
    Apply a scaled radial transformation with deadzone.

    ... (rest of the dz_scaled_radial function docstring)
    '''
    input_magnitude = np.linalg.norm(stick_input)
    if input_magnitude < deadzone:
        return 0, 0
    else:
        input_normalized = stick_input / input_magnitude
        retval = input_normalized * map_range(
            input_magnitude, (deadzone, 1, 0, 1))
        return retval[0], retval[1]

def dz_sloped_scaled_axial(stick_input, deadzone, n=1):
    '''
    Apply a sloped scaled axial transformation with deadzone.

    ... (rest of the dz_sloped_scaled_axial function docstring)
    '''
    x_val = 0
    y_val = 0
    deadzone_x = deadzone * np.power(abs(stick_input[1]), n)
    deadzone_y = deadzone * np.power(abs(stick_input[0]), n)
    sign = np.sign(stick_input)
    if abs(stick_input[0]) > deadzone_x:
        x_val = sign[0] * map_range(abs(stick_input[0]), (deadzone_x, 1, 0, 1))
    if abs(stick_input[1]) > deadzone_y:
        y_val = sign[1] * map_range(abs(stick_input[1]), (deadzone_y, 1, 0, 1))
    return x_val, y_val

def dz_hybrid(stick_input, deadzone):
    '''
    Apply a hybrid transformation with deadzone.

    ... (rest of the dz_hybrid function docstring)
    '''
    input_magnitude = np.linalg.norm(stick_input)
    if input_magnitude < deadzone:
        return 0, 0

    partial_output = dz_scaled_radial(stick_input, deadzone)

    return partial_output
