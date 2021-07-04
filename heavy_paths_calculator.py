import numpy as np
import numba as nb


@nb.njit(['int64[:,::1](int64[:,::1],int64)',
          'int32[:,::1](int32[:,::1],int32)',
          'float64[:,::1](float64[:,::1],int64)'])
def calculate_heavy_paths_matrix(input, N):
    output = np.copy(input)
    for s in range(1, N):
        output[0][s] += output[0][s - 1]
        output[s][0] += output[s - 1][0]

    for y in range(1, N):
        for x in range(1, N):
            output[y][x] += max(output[y - 1][x], output[y][x - 1])

    return output


@nb.njit(['int64[:,::1](int64[:,::1],int64)',
          'int32[:,::1](int32[:,::1],int32)',
          'float64[:,::1](float64[:,::1],int64)'])
def calculate_heavy_paths_diamond_matrix(input, N):
    for y in range(0, N):
        for x in range(0, N):
            half_N = N / 2
            if abs(x - y) > half_N and abs(y + x - N) > half_N:
                input[y][x] = 0
    output = calculate_heavy_paths_matrix(input, N)
    for y in range(N):
        for x in range(N):
            half_N = N / 2
            if abs(x - y) > half_N or abs(y + x - N) > half_N:
                output[y][x] = 0
    return output


@nb.njit(['UniTuple(int32, 2)(int32[:,::1],int32)',
          'UniTuple(int64, 2)(int64[:,::1],int64)',
          'UniTuple(int64, 2)(float64[:,::1],int32)'])
def get_maximum_weight_from_matrix(input, N):
    max = 0
    tuple = (0, 0)
    for y in range(0, N):
        for x in range(0, N):
            inp = input[y][x]
            if (inp > max):
                max = inp
                tuple = (y, x)
    return tuple
