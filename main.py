import itertools
import time

from distribution import *
from graphs_algorithms import *

all_pareto_values = list(itertools.product([0.5, 1, 1.5, 2],
                                           [100, 10000, 1000000, 100000000]))

N = 10
distribution = Distribution.GEOMETRIC
start = time.time()
weights = get_random_weights(distribution, N)
maximum_paths_matrix = calculate_maximum_paths_matrix(weights)
end = time.time()
print("completed in ", end - start)
