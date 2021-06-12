import math
import numpy as np


def generate_bounded_pareto_random(alpha, max_value, N):
    # according to wikipedia page: https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution
    # section: 3.5.1

    uniform_random_numbers = np.random.random(N * N)[0:N * N]
    bounded_pareto_samples = [generate_single_pareto_variable(number, max_value, alpha) for number in uniform_random_numbers]
    return np.reshape(bounded_pareto_samples, (N, N))


def generate_single_pareto_variable(n, z, alpha):
    h = math.pow(z, alpha)
    semi_compution = n * h - n - h
    final = math.pow(-(semi_compution / h), -1 / alpha)
    return final
