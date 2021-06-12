import itertools
import time

from distribution import Distribution, get_random_weights

# NValues = [10,100,1000,10000]
NValues = [10]
paretoValues = list(itertools.product([0.5, 1, 1.5, 2], [100, 10000, 1000000, 100000000]))

start = time.time()
for N in NValues:
    for dist in Distribution:
        print(dist, N)
        if dist == Distribution.PARETO:
            for paretoVal in paretoValues:
                print(get_random_weights(dist, N, paretoVal))
        else:
            print(get_random_weights(dist, N))
end = time.time()
print("completed in: %s seconds" % (end - start))