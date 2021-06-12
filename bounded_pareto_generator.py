import math
import numpy as np


def generate_bounded_pareto_random(alpha, H, N):
    # according to wikipedia page: https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution
    # section: 3.5.1

    uValues = np.random.random(N*N)[0:N*N]
    paretoVariables = [generate_single_pareto_variable(u, H, alpha) for u in uValues]
    return np.reshape(paretoVariables, (N, N))


def generate_single_pareto_variable(u, H, alpha):
    h = math.pow(H, alpha)
    up = u * h - u - h
    final = math.pow(-(up / h), -1 / alpha)
    return final