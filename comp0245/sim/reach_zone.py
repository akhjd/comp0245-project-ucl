import numpy as np

SHOULDER_X, SHOULDER_Z = 0.204, 0.496  # m, in the base frame
ARM_REACH = 0.985  # m: longest shoulder-to-hand distance (measured with FK)
COMFORT = 0.75  # fraction of full reach to aim for


def parking_distance(z):
    """Horizontal distance from the base centre to the target that the base should park at."""
    r_max = np.sqrt(max(ARM_REACH**2 - (z - SHOULDER_Z) ** 2, 0.0))
    return SHOULDER_X + COMFORT * r_max
