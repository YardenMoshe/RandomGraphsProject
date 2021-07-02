import numpy as np
from multiprocessing import Pool, Process

class HeavyPathCalculator:
    def __init__(self, weights):
        self.weights = weights
        self.N = weights.shape.__getitem__(0)


    def calculate_heavy_paths_matrix(self):
        max_weights_matrix = np.copy(self.weights)

        max_weights_matrix[0][0]=self.weights[0][0]
        for x in range(1,self.N):
            max_weights_matrix[0][x]+=max_weights_matrix[0][x-1]

        for y in range(1,self.N):
            max_weights_matrix[y][0]+=max_weights_matrix[y-1][0]

        for y in range(1,self.N):
            for x in range(1,self.N):
                max_weights_matrix[y][x] += max(max_weights_matrix[y-1][x],max_weights_matrix[y][x-1])

        return max_weights_matrix