import itertools
import graphs_files_handler
from distribution import *
from graphs_algorithms import *

all_pareto_values = list(itertools.product([0.5, 1, 1.5, 2],
                                           [100, 10000, 1000000, 100000000]))

N = 10000
distribution = Distribution.EXPONENTIAL

weights = get_random_weights(distribution, N)
maximum_paths_matrix = calculate_maximum_paths_matrix(weights)
max_path = get_maximum_path_weight(maximum_paths_matrix)
