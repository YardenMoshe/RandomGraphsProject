import numpy as np
import numba as nb


@nb.njit(['int32[:,::1](int32[:,::1],int32)', 'float64[:,::1](float64[:,::1],int64)'])
def calculate_heavy_paths_matrix(input, N):
    output = np.copy(input)
    for x in range(1, N):
        output[0][x] += output[0][x - 1]

    for y in range(1, N):
        output[y][0] += output[y - 1][0]

    for y in range(1, N):
        for x in range(1, N):
            output[y][x] += max(output[y - 1][x], output[y][x - 1])

    return output
