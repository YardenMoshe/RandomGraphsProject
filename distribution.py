import math
import numpy as np
import numba as nb
from enum import Enum


class Distribution(Enum):
    EXPONENTIAL = 1
    GEOMETRIC = 2
    BINOMIAL_QUARTER = 3
    BINOMIAL_HALF = 4
    PARETO = 5


def get_random_weights(distribution, N, pareto_parameters=(1, 1)):
    if distribution == Distribution.EXPONENTIAL:
        return exponential_random(N)

    if distribution == Distribution.GEOMETRIC:
        return geometrical_random(N)

    if distribution == Distribution.BINOMIAL_QUARTER:
        return binomial_random(N, 0.25)

    if distribution == Distribution.BINOMIAL_HALF:
        return binomial_random(N, 0.5)

    if distribution == Distribution.PARETO:
        return bounded_pareto_random(N, pareto_parameters)


def geometrical_random(N):
    return np.random.geometric(p=1 - 1 / math.e, size=(N, N))  # p=0.63212055882


def exponential_random(N):
    return np.random.exponential(scale=1.0, size=(N, N))


def binomial_random(N, p):
    return np.reshape(np.random.binomial(1, [p] * N * N), (N, N))


@nb.njit(['float64[:,::1](float64,int32,int32)'])
def generate_bounded_pareto_random(alpha, max_value, N):
    # according to wikipedia page:
    # https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution
    # section: 3.5.1
    n_square = int(math.pow(N, 2))
    uniform_random_numbers = np.random.random(n_square)[0:n_square]
    bounded_pareto_samples = np.empty(n_square)
    i = 0
    for n in uniform_random_numbers:
        h = math.pow(max_value, alpha)
        bounded_pareto_samples[i] = math.pow(-((n * h - n - h) / h), -1 / alpha)
        i += 1
    return np.reshape(bounded_pareto_samples, (N, N))


def bounded_pareto_random(N, pareto_parameters):
    return generate_bounded_pareto_random(pareto_parameters[0], pareto_parameters[1], N)
