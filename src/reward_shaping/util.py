import numpy as np


def normal_dist_density(x: float, mean: float, sd: float):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density
