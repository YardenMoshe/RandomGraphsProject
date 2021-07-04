import numpy as np
import numba as nb


__all__ = ['calculate_maximum_paths_matrix',
           'calculate_heavy_paths_diamond_matrix',
           'get_maximum_weight_index',
           'get_maximum_path_weight'
           ]




@nb.njit(['int64[:,::1](int64[:,::1])',
          'int32[:,::1](int32[:,::1])',
          'float64[:,::1](float64[:,::1])'])
def calculate_maximum_paths_matrix(input):
    N = len(input)
    output = np.copy(input)
    for s in range(1, N):
        output[0][s] += output[0][s - 1]
        output[s][0] += output[s - 1][0]

    for y in range(1, N):
        for x in range(1, N):
            output[y][x] += max(output[y - 1][x], output[y][x - 1])

    return output


@nb.njit(['int64[:,::1](int64[:,::1])',
          'int32[:,::1](int32[:,::1])',
          'float64[:,::1](float64[:,::1])'])
def calculate_heavy_paths_diamond_matrix(input):
    N = len(input)
    for y in range(0, N):
        for x in range(0, N):
            half_N = N / 2
            if abs(x - y) > half_N and abs(y + x - N) > half_N:
                input[y][x] = 0
    output = calculate_maximum_paths_matrix(input)
    for y in range(N):
        for x in range(N):
            half_N = N / 2
            if abs(x - y) > half_N or abs(y + x - N) > half_N:
                output[y][x] = 0
    return output


@nb.njit(['UniTuple(int32, 3)(int32[:,::1])',
          'UniTuple(int64, 3)(int64[:,::1])',
          'UniTuple(int64, 3)(float64[:,::1])'])
def get_maximum_weight_and_index(input):
    N = len(input)
    max = 0
    tuple = (0, 0, 0)
    for y in range(0, N):
        for x in range(0, N):
            inp = input[y][x]
            if (inp > max):
                max = inp
                tuple = (y, x, max)
    return tuple


def get_maximum_weight_index(input):
    return (get_maximum_weight_and_index(input).__getitem__(0),
            get_maximum_weight_and_index(input).__getitem__(1))


def get_maximum_path_weight(input):
    return get_maximum_weight_and_index(input).__getitem__(2)
