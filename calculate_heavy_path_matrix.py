import numpy as np


class HeavyPathCalculator:
    def __init__(self, weights):
        self.weights = weights
        self.N = weights.shape.__getitem__(0)
        self.max_weights_matrix = np.zeros((self.N, self.N))

    def max_weights_matrix_value(self, x, y):
        if x < 0 or y < 0:
            return 0
        else:
            return self.max_weights_matrix[y][x]

    def dp(self,y, x):
        return max(self.max_weights_matrix_value(x, y - 1) + self.weights[y][x],
                   (self.max_weights_matrix_value(x - 1, y) + self.weights[y][x]))

    def calculate_heavy_paths_matrix(self):
        for y in range(0, self.N):
            for x in range(0, self.N):
                self.max_weights_matrix[y][x] = self.dp(y, x)
        return self.max_weights_matrix