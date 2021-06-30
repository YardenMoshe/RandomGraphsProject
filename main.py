import itertools
import math
import time

import numpy as np

from calculate_heavy_path_matrix import HeavyPathCalculator
from distribution import Distribution, get_random_weights

# NValues = [10,100,1000,10000]
N = 100
paretoValues = list(itertools.product([0.5, 1, 1.5, 2], [100, 10000, 1000000, 100000000]))

start = time.time()
weights = get_random_weights(Distribution.BINOMIAL_HALF, N)
heavy_weights_matrix = HeavyPathCalculator(weights).calculate_heavy_paths_matrix()
print(weights)
print(heavy_weights_matrix)
end = time.time()
print("completed in: %s seconds" % (end - start))





# start = time.time()
# for N in NValues:
#     for dist in Distribution:
#         print(dist, N)
#         if dist == Distribution.PARETO:
#             for paretoVal in paretoValues:
#                 print(get_random_weights(dist, N, paretoVal))
#         else:
#             print(get_random_weights(dist, N))
# end = time.time()