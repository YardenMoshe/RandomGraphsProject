import itertools
import math
import time

import numpy as np

from calculate_heavy_path_matrix import HeavyPathCalculator
from distribution import Distribution, get_random_weights


N = 100
paretoValues = list(itertools.product([0.5, 1, 1.5, 2], [100, 10000, 1000000, 100000000]))

#main:
start = time.time()
weights = get_random_weights(Distribution.BINOMIAL_HALF, N)
heavy_weights_matrix = HeavyPathCalculator(weights).calculate_heavy_paths_matrix()
end = time.time()
print("completed in: %s seconds" % (end - start))



#Example showing how to use all distruibtionss
# NValues = [10,100,1000,10000]
# start = time.time()
# for N in NValues:
#     for dist in Distribution:
#         print(dist, N)
#         if dist == Distribution.PARETO:
#             for paretoVal in paretoValues:
#                 weights = get_random_weights(dist, N, paretoVal)
#         else:
#             weights = get_random_weights(dist, N)
# end = time.time()
print("completed in: %s seconds" % (end - start))
