import os
import time
from typing import List, Dict, Any, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, cm
from sklearn.cluster import OPTICS, DBSCAN, AgglomerativeClustering

from FileSystem.general_purpose_functions import mkdir_if_not_exists
from FindMinimumOfFunc.view_layer import plot_rect
from SearchAlg.genetic_alg_general_functions import plot_hist_of_list


def get_min_samples_by_N_for_values_clustering(N):
    min_samples = -1
    if N <= 5:
        min_samples = 3
    if 5 < N <= 1000:
        min_samples = 5
    if 1000 < N <= 1500:
        min_samples = 9
    if N > 1500:
        min_samples = N / 100
    return int(min_samples)

def get_min_samples_by_N_for_space_clustering(N):
    min_samples = -1
    if N <= 5:
        min_samples = 2
    if 5 < N <= 20:
        min_samples = 4
    if 20 < N<=100:
        min_samples = 5
    if 100 < N <= 1000:
        min_samples = 6
    if 1000 < N <= 1500:
        min_samples = 7
    if N > 1500:
        min_samples = N / 100
    return int(min_samples)



def clustering_vertices(distance_matrix, min_samples):
    clustering_alg = OPTICS(min_samples=min_samples,
                            metric='precomputed')
    clustering = clustering_alg.fit(distance_matrix)
    return clustering.labels_

def cluster_by_hand(cost_values):
    # тут могут потеряться точки из-за percentile
    x = cost_values
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    min_x = np.min(x)
    max_x = np.max(x)
    num_of_bins = int((max_x-min_x)/bin_width)
    # plot_hist_of_list(cost_values)
    clusters = {}
    remaining_indexes = [ind for ind in range(len(cost_values))]
    for i in range(num_of_bins):
        cluster_labels = []
        v_segment = [min_x+i*bin_width, min_x+(i+1)*bin_width]
        for unseen_object_index in remaining_indexes:
            unseen_object = cost_values[unseen_object_index]
            if v_segment[0] <= unseen_object <= v_segment[1]:
                cluster_labels.append(unseen_object_index)
        clusters.update({i: cluster_labels})
        remaining_indexes = np.setdiff1d(remaining_indexes, cluster_labels)
    # вернуть [номер кластера для i го объекта из cost values, номер кластера для i+1 го объекта из cost values,...]
    output = []
    for object_index in range(len(cost_values)):
        for key_ in clusters:
            if object_index in clusters[key_]:
               output.append(key_)
    return output


def hierarchical_clustering(distance_matrix, num_of_clusters):
    clustering_alg = AgglomerativeClustering(n_clusters=num_of_clusters,
                                             metric='precomputed',
                                             linkage='average')
    clustering = clustering_alg.fit(distance_matrix)
    return clustering.labels_


def clustering_vertices_in_space(distance_matrix, min_samples, eps):

    # eps = np.percentile(distance_matrix, 1)

    clustering_alg = DBSCAN(min_samples=min_samples,
                            eps=eps,
                            metric='precomputed')
    # clustering_alg = OPTICS(min_samples=min_samples,
    #                         metric='precomputed')
    clustering = clustering_alg.fit(distance_matrix)
    return clustering.labels_


# def clustering_vertices_Hierarh(distance_matrix, n_clusters):
#
#     clustering_alg = AgglomerativeClustering(n_clusters=n_clusters,
#                                              metric='precomputed',
#                                              linkage='average',
#                                              # connectivity=conn_matrix
#                                              )
#     clustering = clustering_alg.fit(distance_matrix)
#     return clustering.labels_


def step_1(points_group, is_it_possible_to_crossbreed_func) -> Tuple[List[Dict[str, Any]], int]:
    # points_group = {
    #     'points': points_,
    #     'values_on_points': values_on_points_,
    #     'matrix_of_dist': matrix_of_dist_,
    #     'matrix_of_cost_dist': matrix_of_c_dist_,
    #     'min_value': np.min(values_on_points_),
    #     'indexes_in_the_ancestor': indexes_of_points
    # }
    points = points_group['points']
    N = len(points)
    phi_distance_matrix = points_group['matrix_of_cost_dist']
    # lables_of_points = clustering_vertices(distance_matrix=phi_distance_matrix, min_samples=get_min_samples_by_N_for_values_clustering(N))
    lables_of_points = cluster_by_hand(cost_values=points_group['values_on_points'])
    number_of_lost_points_cost_clustering = 0
    # получить все кластеры, отбросив точки, которые в кластеры не попали
    clusters = {}
    for unique_label in np.unique(lables_of_points):
        if unique_label == -1:
            continue
        clusters.update({unique_label: []})  # indexes of points

    k_ = 0
    for label_, elem_ in zip(lables_of_points, points):
        if label_ == -1:
            number_of_lost_points_cost_clustering += 1
            k_ += 1
            continue
        clusters[label_].append(k_)
        k_ += 1

    values_on_points = points_group['values_on_points']
    # подсчет min phi в кластере
    cluster_min_phi_key = []
    cluster_min_phi_value = []
    for key_ in clusters:
        indexes_of_points = clusters[key_]
        values = [values_on_points[index_of_point] for index_of_point in indexes_of_points]
        min_ = np.min(values)
        cluster_min_phi_key.append(key_)
        cluster_min_phi_value.append(min_)
    # сортировка, возврат лучших
    indexes_of_best = np.argsort(cluster_min_phi_value)
    best_list = []
    # max_n = min(len(indexes_of_best), the_size_of_the_winners_list)
    max_n = len(indexes_of_best)
    for i in range(max_n):
        index_of_best = indexes_of_best[i]
        best_list.append(cluster_min_phi_key[index_of_best])

    m_dists = points_group['matrix_of_dist']
    m_c_dists = points_group['matrix_of_cost_dist']
    output = []

    # indexes_in_the_ancestor_ = points_group['indexes_in_the_ancestor']
    for key_ in clusters:
        if key_ in best_list:
            indexes_of_points = clusters[key_]
            points_ = [points[i] for i in indexes_of_points]
            values_on_points_ = [values_on_points[i] for i in indexes_of_points]
            n_ = len(points_)
            matrix_of_dist_ = np.zeros(shape=(n_, n_))
            matrix_of_c_dist_ = np.zeros(shape=(n_, n_))
            for i in range(n_ - 1):
                for j in range(i + 1, n_):
                    i_index = indexes_of_points[i]
                    j_index = indexes_of_points[j]
                    dist_ = m_dists[i_index][j_index]
                    cost_dist_ = m_c_dists[i_index][j_index]
                    matrix_of_dist_[i][j] = dist_
                    matrix_of_dist_[j][i] = dist_
                    matrix_of_c_dist_[i][j] = cost_dist_
                    matrix_of_c_dist_[j][i] = cost_dist_
            points_group = {
                'points': points_,
                'values_on_points': values_on_points_,
                'matrix_of_dist': matrix_of_dist_,
                'matrix_of_cost_dist': matrix_of_c_dist_,
                'min_value': np.min(values_on_points_),
                'max_value':np.max(values_on_points_),
                'indexes_in_the_ancestor': indexes_of_points,
                'is_it_possible_to_crossbreed': is_it_possible_to_crossbreed_func(len(points_))
            }
            output.append(points_group)

    return output, number_of_lost_points_cost_clustering


def step_2(points_groups, is_it_possible_to_crossbreed_func, eps):
    # points_group = {
    #     'points': points_,
    #     'values_on_points': values_on_points_,
    #     'matrix_of_dist': matrix_of_dist_,
    #     'matrix_of_cost_dist': matrix_of_c_dist_,
    #     'min_value': np.min(values_on_points_),
    #     'indexes_in_the_ancestor': indexes_of_points
    # }
    # all_N = get_groups_capacity(points_groups)
    number_of_lost_points_space_clustering = 0
    output = []
    unclustered_groups = []
    for i in range(len(points_groups)):

        points_group = points_groups[i]
        indexes_in_the_ancestor_ = points_group['indexes_in_the_ancestor']
        # print('clustering by space {}/{}'.format(i, len(points_groups)))
        points = points_group['points']
        N = len(points)
        space_distance_matrix = points_group['matrix_of_dist']
        # print(N)
        lables_of_points = clustering_vertices_in_space(distance_matrix=space_distance_matrix,
                                               min_samples=get_min_samples_by_N_for_space_clustering(N),
                                                eps=eps)
        clusters = {}
        for unique_label in np.unique(lables_of_points):
            if unique_label == -1:
                continue
            clusters.update({unique_label: []})  # indexes of points
        #
        # if len(clusters) == 1:
        #     print('warning')
        unclustered_points_indexes = []
        k_ = 0
        for label_, elem_ in zip(lables_of_points, points):
            if label_ == -1:
                number_of_lost_points_space_clustering += 1
                unclustered_points_indexes.append(k_)
                k_ += 1
                continue
            clusters[label_].append(k_)
            k_ += 1

        values_on_points = points_group['values_on_points']
        m_dists = points_group['matrix_of_dist']
        m_c_dists = points_group['matrix_of_cost_dist']

        #split by space
        for key_ in clusters:
            indexes_of_points = clusters[key_]
            points_ = [points[i] for i in indexes_of_points]
            values_on_points_ = [values_on_points[i] for i in indexes_of_points]
            n_ = len(points_)
            matrix_of_dist_ = np.zeros(shape=(n_, n_))
            matrix_of_c_dist_ = np.zeros(shape=(n_, n_))
            for i in range(n_ - 1):
                for j in range(i + 1, n_):
                    i_index = indexes_of_points[i]
                    j_index = indexes_of_points[j]
                    dist_ = m_dists[i_index][j_index]
                    cost_dist_ = m_c_dists[i_index][j_index]
                    matrix_of_dist_[i][j] = dist_
                    matrix_of_dist_[j][i] = dist_
                    matrix_of_c_dist_[i][j] = cost_dist_
                    matrix_of_c_dist_[j][i] = cost_dist_
            points_group = {
                'points': points_,
                'values_on_points': values_on_points_,
                'matrix_of_dist': matrix_of_dist_,
                'matrix_of_cost_dist': matrix_of_c_dist_,
                'min_value': np.min(values_on_points_),
                'max_value':np.max(values_on_points_),
                'indexes_in_the_ancestor': [indexes_in_the_ancestor_[ind] for ind in indexes_of_points],
                'is_it_possible_to_crossbreed': is_it_possible_to_crossbreed_func(len(points_))
            }
            output.append(points_group)


        points_ = [points[i] for i in unclustered_points_indexes]
        values_ = [values_on_points[i] for i in unclustered_points_indexes]
        n_ = len(points_)
        if n_ == 0:
            continue
        matrix_of_dist_ = np.zeros(shape=(n_, n_))
        matrix_of_c_dist_ = np.zeros(shape=(n_, n_))
        for i in range(n_ - 1):
            for j in range(i + 1, n_):
                i_index = unclustered_points_indexes[i]
                j_index = unclustered_points_indexes[j]
                dist_ = m_dists[i_index][j_index]
                cost_dist_ = m_c_dists[i_index][j_index]
                matrix_of_dist_[i][j] = dist_
                matrix_of_dist_[j][i] = dist_
                matrix_of_c_dist_[i][j] = cost_dist_
                matrix_of_c_dist_[j][i] = cost_dist_
        unclustered_group = {
            'points': points_,
            'values_on_points':values_,
            'matrix_of_dist': matrix_of_dist_,
            'matrix_of_cost_dist': matrix_of_c_dist_,
            'min_value': np.min(values_),
            'max_value': np.max(values_),
            'indexes_in_the_ancestor': [indexes_in_the_ancestor_[ind] for ind in unclustered_points_indexes],
            'is_it_possible_to_crossbreed': is_it_possible_to_crossbreed_func(len(points_))
        }
        unclustered_groups.append(unclustered_group)

    return output, unclustered_groups, number_of_lost_points_space_clustering


def get_best_groups(points_groups, size_of_top):
    number_of_groups = len(points_groups)
    groups_values = []
    for i in range(number_of_groups):
        # требуется модификация
        points_group = points_groups[i]
        points = points_group['points']
        N = len(points)
        min_v = points_group['min_value']
        groups_values.append(min_v)

    args_of_min_values = np.argsort(groups_values)
    max_index = min(size_of_top, number_of_groups)
    output = [points_groups[args_of_min_values[i]] for i in range(max_index)]
    return output


def gen_new_points_in_group(points_group,
                            number_of_new_samples,
                            func,
                            crossing_func,
                            dist_func,
                            is_it_possible_to_crossbreed_func
                            ):
    values_on_points = points_group['values_on_points']
    m_dists = points_group['matrix_of_dist']
    m_c_dists = points_group['matrix_of_cost_dist']
    points = points_group['points']

    # gen
    n_new = number_of_new_samples
    new_points = []
    print('gen new points')
    for i in range(n_new):
        # print('{}/{}'.format(i, n_new - 1))
        new_points.append(crossing_func(points))
    # new values
    new_values = [func(point) for point in new_points]

    n_old = len(points)
    N = n_old + n_new
    # compute and join
    phi_distance_matrix = np.zeros(shape=(N, N))
    print('create conn matrix')
    for i in range(N - 1):
        # print('{}/{}'.format(i, N - 1))
        for j in range(i + 1, N):
            dist_ = 0.0
            if i < n_old and j < n_old:
                dist_ = m_c_dists[i][j]
            if i < n_old and j >= n_old:
                dist_ = np.absolute(values_on_points[i] - func(new_points[j - n_old]))
            if i >= n_old and j < n_old:
                dist_ = np.absolute(func(new_points[i - n_old]) - values_on_points[j])
            if i >= n_old and j >= n_old:
                dist_ = np.absolute(func(new_points[i - n_old]) - func(new_points[j - n_old]))
            if dist_ == 0.0:
                print('error')
                raise SystemExit
            phi_distance_matrix[i][j] = dist_
            phi_distance_matrix[j][i] = dist_

    matrix_of_dist = np.zeros(shape=(N, N))
    print('create dist matrix')
    for i in range(N - 1):
        # print('{}/{}'.format(i, N - 1))
        for j in range(i + 1, N):
            dist_ = 0.0
            if i < n_old and j < n_old:
                dist_ = m_dists[i][j]
            if i < n_old and j >= n_old:
                dist_ = dist_func(points[i], new_points[j - n_old])
            if i >= n_old and j < n_old:
                dist_ = dist_func(new_points[i - n_old], points[j])
            if i >= n_old and j >= n_old:
                dist_ = dist_func(new_points[i - n_old], new_points[j - n_old])
            matrix_of_dist[i][j] = dist_
            matrix_of_dist[j][i] = dist_
    points_= points + new_points
    points_group_new = {
        'points': points_,
        'values_on_points': values_on_points + new_values,
        'matrix_of_dist': matrix_of_dist,
        'matrix_of_cost_dist': phi_distance_matrix,
        'min_value': np.min([points_group['min_value'], np.min(new_values)]),
        'max_value': np.max([points_group['max_value'], np.min(new_values)]),
        'is_it_possible_to_crossbreed_func':is_it_possible_to_crossbreed_func(len(points_))
    }
    return points_group_new


def get_best_point(points_group):
    points = points_group['points']
    values = points_group['values_on_points']
    argmin_ = np.argmin(values)
    min_ = values[argmin_]
    return min_, points[argmin_]


def init_debug_plotting(debug_plotting, func):
    plt.rcParams["figure.figsize"] = [14, 7]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    axs = fig.add_subplot(111)
    x_vec = debug_plotting['x_plot']
    y_vec = debug_plotting['y_plot']
    x = []
    y = []
    highs = []
    rects_info = []
    print('calc debug surf')
    for j in range(len(x_vec) - 1):
        for k in range(len(y_vec) - 1):
            x_1 = x_vec[j]
            x_2 = x_vec[j + 1]
            y_1 = y_vec[k]
            y_2 = y_vec[k + 1]
            x.append((x_2 + x_1) / 2)
            y.append((y_2 + y_1) / 2)
            rects_info.append([x_1, x_2, y_1, y_2])
            point_ = [(x_2 + x_1) / 2, (y_2 + y_1) / 2]
            highs.append(func(point_))
    norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
    m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    for i in range(len(rects_info)):
        plot_rect(axs, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],
                  m.to_rgba(highs[i]))
    print('')
    argmin_debug = np.argmin(highs)
    print('debug min value = {} x= {}, y={}'.format(highs[argmin_debug], x[argmin_debug], y[argmin_debug]))

    fig.colorbar(m, ax=axs)
    axs.set_xlim([np.min(x_vec), np.max(x_vec)])
    axs.set_ylim([np.min(y_vec), np.max(y_vec)])
    axs.set_xlabel(r"$x$")
    axs.set_ylabel(r'$y$')
    axs.set_title(r'$\varphi(x,y)$')
    return fig, axs


def plt_points_group(points_group, axs,
                     plot_edges=False,
                     plot_vertexes=True,
                     vertexes_color='#E41919',
                     edges_color='#0A0303'):
    # points_group = {
    #     'points': points_,
    #     'values_on_points': values_on_points_,
    #     'matrix_of_dist': matrix_of_dist_,
    #     'matrix_of_cost_dist': matrix_of_c_dist_,
    #     'min_value': np.min(values_on_points_)
    # }
    group_ = points_group
    points = group_['points']

    colors = [vertexes_color for i in range(len(points))]
    x_ = [points[i][0] for i in range(len(points))]
    y_ = [points[i][1] for i in range(len(points))]
    vertexes_obj = 0.0
    edges_objs = []

    if plot_vertexes:
        vertexes_obj = axs.scatter(x_, y_, c=colors)
    if plot_edges:
        number_of_points = len(points)
        for i in range(number_of_points):
            for j in range(number_of_points):
                if i != j:
                    x_line = [points[i][0], points[j][0]]
                    y_line = [points[i][1], points[j][1]]
                    edge_ = axs.plot(x_line, y_line, color=edges_color, marker='', linewidth=2, alpha=0.1)
                    edges_objs.append(edge_)
    axs.set_title(r'$generate \: random \: points$')
    return vertexes_obj, edges_objs


def plot_points_groups(points_groups, axs,
                       plot_edges=False,
                       plot_vertexes=True,
                       force_color=False,
                       color=''
                       ):
    vertex_list = []
    edges_list = []
    n_ = len(points_groups)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_ - 1)
    m = cm.ScalarMappable(norm=norm, cmap=plt.cm.gist_ncar)
    if not force_color:
        for i in range(n_):
            group = points_groups[i]
            vertexes_obj, edges_objs = plt_points_group(
                points_group=group,
                axs=axs,
                plot_edges=plot_edges,
                plot_vertexes=plot_vertexes,
                vertexes_color=m.to_rgba(i),
                edges_color=m.to_rgba(i))
            vertex_list.append(vertexes_obj)
            edges_list.append(edges_objs)
    else:
        for i in range(n_):
            group = points_groups[i]
            vertexes_obj, edges_objs = plt_points_group(
                points_group=group,
                axs=axs,
                plot_edges=plot_edges,
                plot_vertexes=plot_vertexes,
                vertexes_color=color,
                edges_color=color)
            vertex_list.append(vertexes_obj)
            edges_list.append(edges_objs)
    return vertex_list, edges_list


def clear_axs(vertex_list, edges_list):
    for i in range(len(vertex_list)):
        vertexes = vertex_list[i]
        if vertexes == 0.0:
            continue
        vertexes.set_visible(False)
    for i in range(len(edges_list)):
        edjes_of_group = edges_list[i]
        for j in range(len(edjes_of_group)):
            edge_ = edjes_of_group[j][0]
            edge_.set_visible(False)


def get_eps(distance_matrix):
    N = distance_matrix.shape[0]
    size_ = int((N**2-N)/2)
    all_distanses = np.zeros(shape=(size_,))
    k_ = 0
    for i in range(N-1):
        for j in range(i+1,N):
            all_distanses[k_] = distance_matrix[i][j]
            k_+=1
    eps = np.percentile(all_distanses, 1)
    return eps

def sort_objects_by_values(points_group):
    values = points_group['values_on_points']
    indexes_of_sorted_objects = np.argsort(values)
    corr_values = []
    for i in range(len(indexes_of_sorted_objects)):
        corr_values.append(values[indexes_of_sorted_objects[i]])
    return indexes_of_sorted_objects, corr_values

def sorted_minus_groups(indexes_of_sorted_objects:List[int],
                        corr_values_1:List[float],
                        list_of_groups_indxs:List[List[int]],
                        mins_in_gropus:List[float]):

    min_of_groups = np.min(mins_in_gropus)
    output = []
    for i in range(len(indexes_of_sorted_objects)):
        index_of_sorted_object = indexes_of_sorted_objects[i]
        value_of_sorted_object = corr_values_1[i]
        # если найдется значение в осортированных, лучшее чем в во всех группах
        # и это значение не в группах, то записать

        # is in groups
        is_in_groups = True
        for group_index in range(len(list_of_groups_indxs)):
            group_indexes = list_of_groups_indxs[group_index]
            if index_of_sorted_object not in group_indexes:
                is_in_groups = False
                break
        if is_in_groups==False:
            if value_of_sorted_object < min_of_groups:
                output.append(index_of_sorted_object)

    return output

def make_best_of_the_best_group(points_group, points_groups2, is_it_possible_to_crossbreed_func):
    all_sorted_points_indexes, corr_values_1 = sort_objects_by_values(points_group)
    indexes_in_space_clusters = [el['indexes_in_the_ancestor'] for el in points_groups2]
    best_of_the_best_indexes = sorted_minus_groups(
        indexes_of_sorted_objects=all_sorted_points_indexes,
        corr_values_1=corr_values_1,
        list_of_groups_indxs=indexes_in_space_clusters,
        mins_in_gropus=[group['min_value'] for group in points_groups2]
    )
    N_best = len(best_of_the_best_indexes)
    print('best of the best size {}'.format(len(best_of_the_best_indexes)))
    if N_best > 0:
        best_points = [points_group['points'][index_of_point] for index_of_point in best_of_the_best_indexes]
        m_d_best = np.zeros(shape=(N_best, N_best))
        m_cd_best = np.zeros(shape=(N_best, N_best))
        # points_dist_matrix = points_group['matrix_of_dist']
        for i in range(N_best - 1):
            for j in range(i + 1, N_best):
                dist_ = points_group['matrix_of_dist'][best_of_the_best_indexes[i]][best_of_the_best_indexes[j]]
                m_d_best[i][j] = dist_
                m_d_best[j][i] = dist_

        for i in range(N_best - 1):
            for j in range(i + 1, N_best):
                dist_ = points_group['matrix_of_cost_dist'][best_of_the_best_indexes[i]][best_of_the_best_indexes[j]]
                m_cd_best[i][j] = dist_
                m_cd_best[j][i] = dist_
        points_values = points_group['values_on_points']

        best_of_the_best_group = {
            'points': best_points,
            'values_on_points': [points_values[index_of_point] for index_of_point in best_of_the_best_indexes],
            'matrix_of_dist': m_d_best,
            'matrix_of_cost_dist': m_cd_best,
            'min_value': np.min(corr_values_1),
            'max_value': np.max(corr_values_1),
            'is_it_possible_to_crossbreed_func': is_it_possible_to_crossbreed_func(len(best_points))

        }
        return N_best, best_of_the_best_group
    else:
        return N_best, {}


def get_number_of_clusters(N_of_objects, simpex_size):
    remainder = N_of_objects % simpex_size
    if remainder == 0:
        return N_of_objects//simpex_size
    else:
        return N_of_objects//simpex_size + 1

def get_dist_to_group_by_space(all_d_m, index_of_point,group):
    group_indxs = group['indexes_in_the_ancestor']
    dists= []
    for index_of_point_in_group in group_indxs:
        dists.append(all_d_m[index_of_point][index_of_point_in_group])
    dist_ = np.min(dists)
    return dist_

def find_all_possible_local_minimums_maximums_and_unknown(all_points, all_d_m, all_values, clustered_groups, unclustered_groups,
                                                          is_it_possible_to_crossbreed_func):

    # dim_of_set = np.max(all_d_m)

    all_unclustered_indexes = []
    for unclustered_group in unclustered_groups:
        indxs = unclustered_group['indexes_in_the_ancestor']
        for indx in indxs:
            all_unclustered_indexes.append(indx)
    # фиксируем неразмеченную точку и классифицируем ее
    unclustered_poins_info = []
    unknown_indexes = []
    for unclustered_indx in all_unclustered_indexes:
        # получаем расстояния от фиксированной точки до всех выделенных кластеров
        dists_to_sets = []
        for i in range(len(clustered_groups)):
            clustered_group = clustered_groups[i]
            dists_to_sets.append(get_dist_to_group_by_space(all_d_m, unclustered_indx, clustered_group))
        k = 3
        nearest_clusters_indexes = np.argsort(dists_to_sets)[:k]
        point_info = {
            'index_of_point_in_the_ancestor':unclustered_indx
        }
        less_then_in_clusters = True
        for nearest_cluster_index in nearest_clusters_indexes:
            cluster = clustered_groups[nearest_cluster_index]
            # проверяем, меньше ли всех, значений в кластере текущая точка
            if all_values[unclustered_indx] >= cluster['min_value']:
                less_then_in_clusters = False
                break

        greater_then_in_cluster = True
        for nearest_cluster_index in nearest_clusters_indexes:
            cluster = clustered_groups[nearest_cluster_index]
            # проверяем, меньше ли всех, значений в кластере текущая точка
            if all_values[unclustered_indx] <= cluster['max_value']:
                greater_then_in_cluster = False
                break

        if less_then_in_clusters:
            point_info.update({'is_local_minimum': True})
            point_info.update({'is_local_maximum': False})
            point_info.update({'ref_clusters': nearest_clusters_indexes})
        elif greater_then_in_cluster:
            point_info.update({'is_local_minimum': False})
            point_info.update({'is_local_maximum': True})
            point_info.update({'ref_clusters': nearest_clusters_indexes})
        else:
            point_info.update({'is_local_minimum': False})
            point_info.update({'is_local_maximum': False})
            point_info.update({'ref_clusters': nearest_clusters_indexes})
            unknown_indexes.append(unclustered_indx)
        unclustered_poins_info.append(point_info)

    local_minima = []
    local_maxima = []

    # уникальные k опорных кластеров(те которые мы смогли выделить)
    k_ref_clusters = {}
    for point_info in unclustered_poins_info:
        is_local_minimum = point_info['is_local_minimum']
        is_local_maximum = point_info['is_local_maximum']
        ref_clusters = point_info['ref_clusters']
        k_ref_clusters.update({str(np.sort(ref_clusters)):
        {
            'minima': [],
            'maxima': []
        }})
    for point_info in unclustered_poins_info:
        is_local_minimum = point_info['is_local_minimum']
        is_local_maximum = point_info['is_local_maximum']
        ref_clusters = point_info['ref_clusters']
        point_index = point_info['index_of_point_in_the_ancestor']
        if is_local_minimum:
            k_ref_clusters[str(np.sort(ref_clusters))]['minima'].append(point_index)
        elif is_local_maximum:
            k_ref_clusters[str(np.sort(ref_clusters))]['maxima'].append(point_index)

    for k_cluster_key in k_ref_clusters:
        k_cluster = k_ref_clusters[k_cluster_key]

        if len(k_cluster['minima']) != 0:
            points_= [all_points[ind] for ind in k_cluster['minima']]
            values_ = [all_values[ind] for ind in k_cluster['minima']]
            new_group = {
                'points': points_,
                'values_on_points': values_,
                'min_value': np.min(values_),
                'max_value': np.max(values_),
                'is_it_possible_to_crossbreed_func': is_it_possible_to_crossbreed_func(len(points_))
            }
            local_minima.append(new_group)

        if len(k_cluster['maxima']) != 0:
            points_= [all_points[ind] for ind in k_cluster['maxima']]
            values_ = [all_values[ind] for ind in k_cluster['maxima']]
            new_group = {
                'points': points_,
                'values_on_points': values_,
                'min_value': np.min(values_),
                'max_value': np.max(values_),
                'is_it_possible_to_crossbreed_func': is_it_possible_to_crossbreed_func(len(points_))
            }
            local_maxima.append(new_group)

    return local_minima, local_maxima, unknown_indexes

# def make_a_simplices_of_groups(group_of_clusters, simpex_size):
#     group = group_of_clusters
#     N = len(group)
#     num_of_clusters = get_number_of_clusters(N, simpex_size)
#     simplexes = []
#     points = group['points']
#     m_d = group['matrix_of_dist']
#     m_cd = group['matrix_of_cost_dist']
#     values_on_points = group['values_on_points']
#
#     output = []
#
#     N = len(points)
#     # print(N)
#     lables_of_points = hierarchical_clustering(distance_matrix=m_d, num_of_clusters=num_of_clusters)
#     clusters = {}
#     for unique_label in np.unique(lables_of_points):
#         if unique_label == -1:
#             continue
#         clusters.update({unique_label: []})  # indexes of points
#
#     k_ = 0
#     for label_, elem_ in zip(lables_of_points, points):
#         if label_ == -1:
#             k_ += 1
#             continue
#         clusters[label_].append(k_)
#         k_ += 1
#
#     values_on_points = points_group['values_on_points']
#     m_dists = points_group['matrix_of_dist']
#     m_c_dists = points_group['matrix_of_cost_dist']
#
#     #split by space
#     for key_ in clusters:
#         indexes_of_points = clusters[key_]
#         points_ = [points[i] for i in indexes_of_points]
#         values_on_points_ = [values_on_points[i] for i in indexes_of_points]
#         n_ = len(points_)
#         matrix_of_dist_ = np.zeros(shape=(n_, n_))
#         matrix_of_c_dist_ = np.zeros(shape=(n_, n_))
#         for i in range(n_ - 1):
#             for j in range(i + 1, n_):
#                 i_index = indexes_of_points[i]
#                 j_index = indexes_of_points[j]
#                 dist_ = m_dists[i_index][j_index]
#                 cost_dist_ = m_c_dists[i_index][j_index]
#                 matrix_of_dist_[i][j] = dist_
#                 matrix_of_dist_[j][i] = dist_
#                 matrix_of_c_dist_[i][j] = cost_dist_
#                 matrix_of_c_dist_[j][i] = cost_dist_
#         points_group = {
#             'points': points_,
#             'values_on_points': values_on_points_,
#             'matrix_of_dist': matrix_of_dist_,
#             'matrix_of_cost_dist': matrix_of_c_dist_,
#             'min_value': np.min(values_on_points_),
#             'indexes_in_the_ancestor': [indexes_in_the_ancestor_[ind] for ind in indexes_of_points],
#             'is_it_possible_to_crossbreed': is_it_possible_to_crossbreed_func(len(points_))
#         }
#         output.append(points_group)
#

def get_degree_of_knowledge(base_capacity, unknown_capacity):
    return (1.0-unknown_capacity/base_capacity)*100

def get_groups_capacity(groups):
    len_ = 0
    for group in groups:
        len_+=len(group['points'])
    return len_

# def get_smallest_elements_using_recursion(groups, number_of_elements):
#     current_group
#     argsort_ = np.argsort(groups['values_on_points'])
#     argsort_len_ = len(argsort_)


def find_min(points_group,
             func,
             crossing_func,
             dist_func,
             is_it_possible_to_crossbreed_func,
             epsilon,
             max_simplex_size=5,
             recursion_depth=3,
             num_of_groups_in_end_of_rec_step=3,
             gen_in_each_group=1000,
             debug_plotting={}
             ):
    # алгоритм Нелдера-Мида



    # random search where the best
    # points_ = points_group['points']
    # values_ = points_group['values_on_points']
    # argsort_ = np.argsort(values_)
    # current_best_index = argsort_[0]
    #
    # print('start min = {} x = {} y ={}'.format(values_[current_best_index], points_[current_best_index][0], points_[current_best_index][1]))
    # # среди отсортированных по значению выбрать первые k ближайших по пространству
    # K = 10
    # d_m_ = points_group['matrix_of_dist']
    # distances_argsort = np.argsort(d_m_[current_best_index])
    # simplex_indexes = distances_argsort[1:K]
    #
    # new_generation = {
    #     'base_point': points_[current_best_index],
    #     'other_points': [points_[simplex_index] for simplex_index in simplex_indexes]
    # }
    # N = 1000
    # new_items = []
    # their_values = []
    # for i in range(N):
    #     p_new = crossing_func(new_generation['base_point'], new_generation['other_points'], epsilon)
    #     new_items.append(p_new)
    #     p_value = func(p_new)
    #     their_values.append(p_value)
    # debug_new_items = {
    #     'points': new_items
    # }
    # debug_base_point = {
    #     'points': [points_[current_best_index]]
    # }
    # debug_simplex_points ={
    #     'points': [points_[simplex_index] for simplex_index in simplex_indexes]
    # }
    # axs = 0.0
    # fig = 0.0
    # path_to_plotting = debug_plotting['dir_to_save_figs']
    # mkdir_if_not_exists(path_to_plotting)
    # if debug_plotting['do_plot']:
    #     fig, axs = init_debug_plotting(debug_plotting, func)
    #     fig.savefig(os.path.join(path_to_plotting, 'step_0.png'))
    # if debug_plotting['do_plot']:
    #     vertex_list1, edges_list1 = plot_points_groups(points_groups=[debug_new_items],
    #                                                  axs=axs,
    #                                                  plot_edges=False,
    #                                                  plot_vertexes=True,
    #                                                  force_color=True,
    #                                                  color='#000000')
    #     vertex_list2, edges_list2 = plot_points_groups(points_groups=[debug_base_point],
    #                                                  axs=axs,
    #                                                  plot_edges=False,
    #                                                  plot_vertexes=True,
    #                                                  force_color=True,
    #                                                  color='#880808')
    #     vertex_list3, edges_list3 = plot_points_groups(points_groups=[debug_simplex_points],
    #                                                  axs=axs,
    #                                                  plot_edges=False,
    #                                                  plot_vertexes=True,
    #                                                  force_color=True,
    #                                                  color='#ffff00')
    #
    #     axs.set_title('generate points where the best')
    #     fig.savefig(os.path.join(path_to_plotting, 'random_eps_search.png'))
    #     clear_axs(vertex_list=vertex_list1, edges_list=edges_list1)
    #     clear_axs(vertex_list=vertex_list2, edges_list=edges_list2)
    #     clear_axs(vertex_list=vertex_list3, edges_list=edges_list3)
    #
    # argmin_pos = np.argsort(their_values)[0]
    # argmin_=  new_items[argmin_pos]
    # min_  = their_values[argmin_pos]
    # print('step min = {} x = {} y ={}'.format(min_,argmin_[0],argmin_[1]))
    # return min_,argmin_

























    # print(len(points_group['points']))
    # # debug_plotting = {
    # #     'do_plot': True,
    # #     'x_plot': x_grid,
    # #     'y_plot': y_grid,
    # #     'dir_to_save_figs': config.debug_min_search_base_path
    # # }
    #
    # # all_sorted_points_indexes, corr_values_1 = sort_objects_by_values(points_group)
    # # from SearchAlg.genetic_alg_general_functions import plot_hist_of_list
    # # plot_hist_of_list(corr_values_1)
    #
    # axs = 0.0
    # fig = 0.0
    # path_to_plotting = debug_plotting['dir_to_save_figs']
    # mkdir_if_not_exists(path_to_plotting)
    # if debug_plotting['do_plot']:
    #     fig, axs = init_debug_plotting(debug_plotting, func)
    #     fig.savefig(os.path.join(path_to_plotting, 'step_0.png'))
    # if debug_plotting['do_plot']:
    #     vertexes_obj, edges_objs = plt_points_group(points_group, axs, plot_edges=False, plot_vertexes=True)
    #     fig.savefig(os.path.join(path_to_plotting, 'step_1.png'))
    #     clear_axs(vertex_list=[vertexes_obj], edges_list=[edges_objs])
    #
    # print('start clustering by cost')
    # points_groups1, number_of_lost_points_cost_clustering = step_1(points_group, is_it_possible_to_crossbreed_func)
    # groups_1_cap = get_groups_capacity(points_groups1)
    # print('number of clusters by cost clustering = {}'.format(len(points_groups1)))
    # print('lost in clusters splitting by cost = {}'.format(number_of_lost_points_cost_clustering))
    # if debug_plotting['do_plot']:
    #     vertex_list, edges_list = plot_points_groups(points_groups=points_groups1,
    #                                                  axs=axs,
    #                                                  plot_edges=False,
    #                                                  plot_vertexes=True)
    #     axs.set_title('value of func clustering')
    #     fig.savefig(os.path.join(path_to_plotting, 'step_2.png'))
    #     clear_axs(vertex_list=vertex_list, edges_list=edges_list)
    # # raise SystemExit
    #
    #
    # print('start clustering by space')
    # eps = get_eps(points_group['matrix_of_dist'])
    # points_groups2, unclustered_groups, number_of_lost_points_space_clustering = step_2(points_groups1, is_it_possible_to_crossbreed_func,eps=eps)
    # print('number of clusters by space clustering = {}'.format(len(points_groups2)))
    # print('lost in clusters splitting by space = {}'.format(number_of_lost_points_space_clustering))
    #
    #
    #
    #
    # # выделить все локальные минимумы и максимумы
    # # выделить все локальные максимумы
    # local_minima, local_maxima, unknown_indexes = find_all_possible_local_minimums_maximums_and_unknown(
    #     all_points=points_group['points'],
    #     all_d_m=points_group['matrix_of_dist'],
    #     all_values=points_group['values_on_points'],
    #     clustered_groups=points_groups2,
    #     unclustered_groups=unclustered_groups,
    #     is_it_possible_to_crossbreed_func=is_it_possible_to_crossbreed_func)
    # print('capacity:')
    # print('local_minima groups {}'.format(get_groups_capacity(local_minima)))
    # print('local_maxima groups {}'.format(get_groups_capacity(local_maxima)))
    # print('number of unknown points {}'.format(len(unknown_indexes)))
    #
    # DOK = get_degree_of_knowledge(base_capacity=len(points_group['points']),
    #                               unknown_capacity = len(unknown_indexes))
    # print('####################################')
    # print('DOK {}'.format(str(DOK)[:3]))
    # print('####################################')
    # if debug_plotting['do_plot']:
    #     vertex_list1, edges_list1 = plot_points_groups(points_groups=points_groups2,
    #                                                  axs=axs,
    #                                                  plot_edges=True,
    #                                                  plot_vertexes=True
    #                                                  )
    #     vertex_list2, edges_list2 = plot_points_groups(points_groups=local_minima,
    #                                                  axs=axs,
    #                                                  plot_edges=False,
    #                                                  plot_vertexes=False,
    #                                                  force_color=True,
    #                                                  color='#000000')
    #     # vertex_list3, edges_list3 = plot_points_groups(points_groups=local_maxima,
    #     #                                              axs=axs,
    #     #                                              plot_edges=True,
    #     #                                              plot_vertexes=True,
    #     #                                             force_color=True,
    #     #                                             color='#FFA07A')
    #     axs.set_title('clustering points in space')
    #     fig.savefig(os.path.join(path_to_plotting, 'step_3.png'))
    #     clear_axs(vertex_list=vertex_list1, edges_list=edges_list1)
    #     clear_axs(vertex_list=vertex_list2, edges_list=edges_list2)
    #     # clear_axs(vertex_list=vertex_list3, edges_list=edges_list3)
    #
    #
    # # raise SystemExit
    #
    # # print('start construct best of the best cluster')
    # # N_best, best_of_the_best_group = make_best_of_the_best_group(points_group=points_group,
    # #                                                              points_groups2=points_groups2,
    # #                                                              is_it_possible_to_crossbreed_func = is_it_possible_to_crossbreed_func)
    #
    # # if debug_plotting['do_plot']:
    # #     vertex_list, edges_list = plot_points_groups(points_groups=[best_of_the_best_group],
    # #                                                  axs=axs,
    # #                                                  plot_edges=False,
    # #                                                  plot_vertexes=True)
    # #     axs.set_title('value of func clustering')
    # #     fig.savefig(os.path.join(path_to_plotting, 'best_of_the_best.png'))
    # #     clear_axs(vertex_list=vertex_list, edges_list=edges_list)
    #
    #
    # # объединить группы в симплексы(построть мосты между разорванными графами)
    #
    #
    # # return 9999, [-9999, 9999]
    # all_groups = local_minima + points_groups2
    # # all_groups = points_groups2
    # points_groups3 = get_best_groups(all_groups, size_of_top=10)




    # points_groups3.append(best_of_the_best_group) #adding best of the best group


    # ###########################points group3 содержит best of the best cluster

    # if debug_plotting['do_plot']:
    #     vertex_list, edges_list = plot_points_groups(points_groups=points_groups3,
    #                                                  axs=axs,
    #                                                  plot_edges=False,
    #                                                  plot_vertexes=True)
    #     axs.set_title('get top {}'.format(num_of_groups_in_end_of_rec_step))
    #     fig.savefig(os.path.join(path_to_plotting, 'step_4.png'))
    #     clear_axs(vertex_list=vertex_list, edges_list=edges_list)
    # points_group_i = gen_new_points_in_group(points_groups3[-2], 10,
    #                                          func=func,
    #                                          crossing_func=crossing_func,
    #                                          dist_func=dist_func,
    #                                          is_it_possible_to_crossbreed_func=is_it_possible_to_crossbreed_func
    #                                          )
    # if debug_plotting['do_plot']:
    #     vertexes_obj, edges_objs = plt_points_group(points_group_i, axs, plot_edges=False, plot_vertexes=True)
    #     axs.set_title('parent is {} cluster. generate new points and do new recursion step. '.format(0))
    #     fig.savefig(os.path.join(path_to_plotting, 'step_5.png'))
    #     clear_axs(vertex_list=[vertexes_obj], edges_list=[edges_objs])
    #
    # min_, argmin_ = get_best_point(points_group_i)
    # print('random search min_={},argmin={}'.format(min_, argmin_))
    # return min_, argmin_



