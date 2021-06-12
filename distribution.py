import math
import numpy as np
from enum import Enum

from bounded_pareto_generator import generate_bounded_pareto_random


class Distribution(Enum):
    EXPONENTIAL = 1
    GEOMETRIC = 2
    BINOMIAL_QUARTER = 3
    BINOMIAL_HALF = 4
    PARETO = 5


def get_random_weights(distruibtion, N, pairForPareteo=(1, 1)):
    if distruibtion == Distribution.EXPONENTIAL:
        return exponential_random(N)

    if distruibtion == Distribution.GEOMETRIC:
        return geometrical_random(N)

    if distruibtion == Distribution.BINOMIAL_QUARTER:
        return binomial_random(N, 0.25)

    if distruibtion == Distribution.BINOMIAL_HALF:
        return binomial_random(N, 0.5)

    if distruibtion == Distribution.PARETO:
        return bounded_pareto_random(N, pairForPareteo)


def geometrical_random(N):
    return np.random.geometric(p=1 - 1 / math.e, size=(N, N))  # p=0.63212055882


def exponential_random(N):
    return np.random.exponential(scale=1.0, size=(N, N))


def binomial_random(N, p):
    return np.reshape(np.random.binomial(1, [p] * N * N), (N, N))


def bounded_pareto_random(N, pareto_parameters):
    return generate_bounded_pareto_random(pareto_parameters[0], pareto_parameters[1], N)
