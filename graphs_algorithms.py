import numpy as np
import numba as nb

__all__ = ['calculate_maximum_paths_matrix',
           'calculate_maximum_paths_diamond_matrix',
           'get_heaviest_vertex',
           'get_max_path_value',
           'get_max_path_as_indexes_list',
           'get_heaviest_index_in_max_path'
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


@nb.njit(['void(int64[:,::1])',
          'void(int32[:,::1])',
          'void(float64[:,::1])'])
def zerofy_matrix_in_diamond_shape(input):
    N = len(input)
    half_N = N / 2
    for y in range(0, N):
        for x in range(0, N):
            if abs(x - y) > half_N or abs(y + x - N) > half_N:
                input[y][x] = 0


@nb.njit(['int64[:,::1](int64[:,::1])',
          'int32[:,::1](int32[:,::1])',
          'float64[:,::1](float64[:,::1])'])
def calculate_maximum_paths_diamond_matrix(input):
    zerofy_matrix_in_diamond_shape(input)
    output = calculate_maximum_paths_matrix(input)
    zerofy_matrix_in_diamond_shape(output)
    return output


@nb.njit(['UniTuple(int32, 3)(int32[:,::1])',
          'UniTuple(int64, 3)(int64[:,::1])',
          'UniTuple(float64, 3)(float64[:,::1])'])
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


@nb.njit()
def get_heaviest_index_in_max_path(weights, max_paths_input, is_diamond=False):
    indexes = get_max_path_as_indexes_list(max_paths_input, is_diamond)
    max = 0
    max_index = indexes[0]
    for index in indexes:
        val = weights[index[0]][index[1]]
        if val > max:
            max = val
            max_index = index
    return max_index


@nb.njit(['void(int64[:,::1])',
          'void(int32[:,::1])',
          'void(float64[:,::1])'])
def minux_oneify_matrix_in_diamond_shape(input):
    N = len(input)
    for y in range(0, N):
        for x in range(0, N):
            half_N = N / 2
            if abs(x - y) > half_N or abs(y + x - N) > half_N:
                input[y][x] = -1


def get_heaviest_vertex(input):
    max_weight_and_index = get_maximum_weight_and_index(input)
    return (int(max_weight_and_index.__getitem__(0)),
            int(max_weight_and_index.__getitem__(1)))


def get_max_path_value(input):
    return get_maximum_weight_and_index(input).__getitem__(2)


@nb.njit()
def get_max_path_as_indexes_list(max_paths_input, is_diamond=False):
    list = []
    N = len(max_paths_input)
    #this is a hack but it will work...
    if is_diamond:
        minux_oneify_matrix_in_diamond_shape(max_paths_input)

    max = 0
    current_index=(0,0)
    for y in range(0, N):
        for x in range(0, N):
            inp = max_paths_input[y][x]
            if (inp > max):
                max = inp
                current_index = (y, x)

    list.append(current_index)
    while current_index != (0, 0):
        y = current_index[0]
        x = current_index[1]
        val1 = 0 if (x - 1) < 0 else max_paths_input[y, x - 1]
        val2 = 0 if (y - 1) < 0 else max_paths_input[y - 1, x]
        if (val1,val2) == (-1,-1):
            break
        current_index = (y, x - 1) if (val1 > val2) else (y - 1, x)
        # print(current_index)
        list.append(current_index)
    if is_diamond:
        zerofy_matrix_in_diamond_shape(max_paths_input)
    return list

