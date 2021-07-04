import itertools
import time

from distribution import *
from graphs_algorithms import *

all_pareto_values = list(itertools.product([0.5, 1, 1.5, 2],
                                           [100, 10000, 1000000, 100000000]))

N = 10000

start = time.time()
weights = get_random_weights(Distribution.EXPONENTIAL, N)
heavy_weights_matrix = calculate_heavy_paths_matrix(weights)
heaviest_path_weight = get_maximum_weight(heavy_weights_matrix)
end = time.time()
print("completed in: %s seconds" % (end - start))


