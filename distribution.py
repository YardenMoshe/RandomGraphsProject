import math
import numpy as np
from enum import Enum

from bounded_pareto_generator import generate_bounded_pareto_random

class Distribution(Enum):
    EXPONENTIAL = 1
    GEOMETRIC = 2
    BINOMIAL_FIRST = 3
    BINOMIAL_SECOND = 4
    PARETO = 5

def getRandomWeights(distruibtion, N,pairForPareteo=(1,1)):
    if distruibtion == Distribution.EXPONENTIAL:
        return exponentialRandomFunction(N)

    if distruibtion == Distribution.GEOMETRIC:
        return geometricalRandomFunction(N)

    if distruibtion == Distribution.BINOMIAL_FIRST:
        return BinaryRandomFunction(N, 0.25)

    if distruibtion == Distribution.BINOMIAL_SECOND:
        return BinaryRandomFunction(N, 0.5)

    if distruibtion == Distribution.PARETO:
        return ParetoRandomFunction(pairForPareteo, N)


def geometricalRandomFunction(N):
    return np.random.geometric(p=1 - 1 / math.e, size=(N, N))  # p=0.63212055882

def exponentialRandomFunction(N):
    return np.random.exponential(scale=1.0, size=(N, N))

def BinaryRandomFunction(N, p):
    return np.reshape(np.random.binomial(1, [p] * N*N),(N,N))

def ParetoRandomFunction(paretoPair, N):
    return generate_bounded_pareto_random(paretoPair[0], paretoPair[1], N)