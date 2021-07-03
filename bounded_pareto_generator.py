import math
import numpy as np
import numba as nb

@nb.njit(['float64[:,::1](float64,int32,int32)'])
def generate_bounded_pareto_random(alpha, max_value, N):
    # according to wikipedia page: https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution
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