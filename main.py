import itertools
import time

from calculate_heavy_path_matrix import HeavyPathCalculator
from distribution import Distribution, get_random_weights

N = 10000
paretoValues = list(itertools.product([0.5, 1, 1.5, 2], [100, 10000, 1000000, 100000000]))

start = time.time()
weights = get_random_weights(Distribution.BINOMIAL_HALF, N)
heavy_weights_matrix = HeavyPathCalculator(weights).calculate_heavy_paths_matrix()
heaviest_path_weight = heavy_weights_matrix[N-1][N-1]
end = time.time()
print("completed in: %s seconds" % (end - start))


#Example showing how to use all distruibtionss
# NValues = [10,100,1000,10000]
# initial_start = time.time()
# for N in NValues:
#     for dist in Distribution:
#         start = time.time()
#         print(dist, N)
#         if dist == Distribution.PARETO:
#             for paretoVal in paretoValues:
#                 weights = get_random_weights(dist, N, paretoVal)
#                 heavy_weights_matrix = HeavyPathCalculator(weights).calculate_heavy_paths_matrix()
#         else:
#             weights = get_random_weights(dist, N)
#             heavy_weights_matrix = HeavyPathCalculator(weights).calculate_heavy_paths_matrix()
#         end=time.time()
#         print("completed in: %s seconds" % (end - start))
# final_end = time.time()
# print("completed in: %s seconds" % (final_end - initial_start))

