import time
import typing
from typing import List
import numpy as np
from matplotlib import pyplot as plt

import config
from FindMinimumOfFunc.algs import find_min
from FindMinimumOfFunc.view_layer import plotsurf
from general_purpose_functions import time_mesuament


def f_(p: List[float]) -> float:
    x_ = p[0]
    y_ = p[1]
    return .5 * (1 - x_) ** 2 + (y_ - x_ ** 2) ** 2 + np.cos(y_ * 5) * np.cos(x_ * 5) * np.exp(-(x_ ** 2 + y_ ** 2))
    # x_ = p[0]
    # y_ = p[1]
    # return np.random.rand()


def space_ro(p1, p2):
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))


def weighted_amount(p0, e_i, epsilon):
    # N = len(points)
    # alpha_vec = np.random.uniform(low=0.0, high=1.0, size=N)
    # alpha_vec = alpha_vec / np.sum(alpha_vec)
    # new_point = [0.0, 0.0]
    # for i in range(N):
    #     tmp = points[i][0]
    #     new_point[0] += alpha_vec[i] * points[i][0]
    #     new_point[1] += alpha_vec[i] * points[i][1]
    # return new_point
    K = len(e_i)
    alpha_vec = np.random.uniform(low=0.0, high=1.0, size=K)
    alpha_vec = alpha_vec / np.sum(alpha_vec)*epsilon
    new_point = [0.0, 0.0]
    new_point[0] = (1 - epsilon) * p0[0]
    new_point[1] = (1 - epsilon) * p0[1]
    for i in range(K):
        new_point[0] += alpha_vec[i] * e_i[i][0]
        new_point[1] += alpha_vec[i] * e_i[i][1]
    return new_point




def is_it_possible_to_crossbreed(size_of_group):
    # get_min_siplex_size_by_size_of_group
    N = size_of_group
    if N >= 2:
        return True
    else:
        return False

if __name__ == '__main__':
    timer = time_mesuament.Timer()
    timer.start()
    np.random.seed(1)
    n = 100
    n_search = 500
    x_grid = np.linspace(-1.0, 1.0, n)
    y_grid = np.linspace(-1.0, 1.0, n)
    x_search = np.random.uniform(-1.0, 1.0, n_search)
    y_search = np.random.uniform(-1.0, 1.0, n_search)

    search_points = [[x_search[i], y_search[i]] for i in range(n_search)]

    N = len(search_points)
    phi_distance_matrix = np.zeros(shape=(N, N))
    print('create conn matrix')
    for i in range(N - 1):
        # print('{}/{}'.format(i, N - 1))
        for j in range(i + 1, N):
            dist_ = np.absolute(f_(search_points[i]) - f_(search_points[j]))
            phi_distance_matrix[i][j] = dist_
            phi_distance_matrix[j][i] = dist_

    matrix_of_dist = np.zeros(shape=(N, N))
    print('create dist matrix')
    for i in range(N - 1):
        # print('{}/{}'.format(i, N - 1))
        for j in range(i + 1, N):
            dist_ = space_ro(search_points[i], search_points[j])
            matrix_of_dist[i][j] = dist_
            matrix_of_dist[j][i] = dist_

    values_on_points = [f_(point) for point in search_points]
    points_group = {
        'points': search_points,
        'values_on_points': values_on_points,
        'matrix_of_dist': matrix_of_dist,
        'matrix_of_cost_dist': phi_distance_matrix,
        'min_value': np.min(values_on_points),
        'is_it_possible_to_crossbreed': is_it_possible_to_crossbreed(len(search_points))
    }

    min_, argmin_ = find_min(
        points_group=points_group,
        func=f_,
        crossing_func=weighted_amount,
        dist_func=space_ro,
        is_it_possible_to_crossbreed_func = is_it_possible_to_crossbreed,
        epsilon=0.9,
        recursion_depth=3,
        num_of_groups_in_end_of_rec_step=10,
        gen_in_each_group=1000,
        debug_plotting={
            'do_plot': True,
            'x_plot': x_grid,
            'y_plot': y_grid,
            'dir_to_save_figs': config.debug_min_search_base_path
        })
    timer.stop()
    print(timer.get_execution_time())
