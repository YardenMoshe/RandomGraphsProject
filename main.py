import itertools
import time

from distribution import Distribution, get_random_weights
from heavy_paths_calculator import calculate_heavy_paths_matrix


paretoValues = list(itertools.product([0.5, 1, 1.5, 2], [100, 10000, 1000000, 100000000]))

# example showing how to use single distribution and N value
N = 10000
start = time.time()
weights = get_random_weights(Distribution.PARETO, N,paretoValues[0])
print(weights)
heavy_weights_matrix = calculate_heavy_paths_matrix(weights,N)
heaviest_path_weight = heavy_weights_matrix[N-1][N-1]
end = time.time()
print("completed in: %s seconds" % (end - start))


# Example showing how to use all distributions
# NValues = [10000]
# initial_start = time.time()
# for N in NValues:
#     for dist in Distribution:
#         if dist == Distribution.PARETO:
#             for paretoVal in paretoValues:
#                 weights = get_random_weights(dist, N, paretoVal)
#                 heavy_weights_matrix = calculate_heavy_paths_matrix(weights, N)
#                 heaviest_path_weight = heavy_weights_matrix[N - 1][N - 1]
#         else:
#             weights = get_random_weights(dist, N)
#             heavy_weights_matrix = calculate_heavy_paths_matrix(weights, N)
#             heaviest_path_weight = heavy_weights_matrix[N - 1][N - 1]
# final_end = time.time()
# print("final computing time : %s seconds" % (final_end - initial_start))