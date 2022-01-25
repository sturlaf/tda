import numpy as np
import random

random.seed(100)


def make_matricies(number):
    return [make_matrix(random.random()) for i in range(number)]


def make_matrix(angle):
    return np.array(
        [
            [np.cos(angle), np.sin(angle), -np.cos(angle)],
            [np.sin(angle), 0, -np.sin(angle)],
            [-np.cos(angle), -np.sin(angle), np.cos(angle)],
        ]
    )
