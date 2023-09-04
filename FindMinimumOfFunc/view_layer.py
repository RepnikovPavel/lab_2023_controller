import random

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import seaborn as sns
from typing import Dict, Any, List

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

import scipy.cluster.hierarchy as hcluster
import scipy.spatial.distance as ssd

from SearchAlg.genetic_alg_general_functions import plot_hist_of_list


def plot_rect(ax, x_1, x_2, y_1, y_2, color):
    h_1 = x_2 - x_1
    h_2 = y_2 - y_1
    rect = matplotlib.patches.Rectangle((x_1, y_1), h_1, h_2, color=color)
    ax.add_patch(rect)


def plot_graph2D(points: Dict[str, Any], adj_matrix: np.array, axs, plot_vertexes=True, plot_edges=True):
    x_ = points['x']
    y_ = points['y']
    if plot_vertexes:
        axs.scatter(x_, y_, color='r')
    if plot_edges:
        N = adj_matrix.shape[0]
        for i in range(N-1):
            print('{}/{}'.format(i, N-1))
            for j in range(i+1, N):
                if adj_matrix[i][j] > 0.0:
                    x_line = [x_[i], x_[j]]
                    y_line = [y_[i], y_[j]]
                    axs.plot(x_line,y_line,color='black', marker='',linewidth=2, alpha=0.2)


def plot_clusters(groups: List[Dict[str, Any]], list_of_clusters_lables: List[Any], axs, plot_edges=True):
    # group_ = {
    #     'points': local_groups[group_index],
    #     'cost_function_at_a_given_scale': cluster_median_phi[cost_cluster_key],
    #     'belonging_to_a_cluster_by_cost': cost_cluster_key
    # }
    labels_of_clusters = list_of_clusters_lables
    norm = matplotlib.colors.Normalize(vmin=min(labels_of_clusters), vmax=max(labels_of_clusters))
    m = cm.ScalarMappable(norm=norm, cmap=plt.cm.gist_ncar)

    all_colors = []
    unique_labels = np.unique(labels_of_clusters)
    for i in range(len(unique_labels)):
        all_colors.append(m.to_rgba(unique_labels[i]))
    # random.shuffle(all_colors)

    label_color={}
    tmp_index = 0
    for lable_ in unique_labels:
        label_color.update({lable_:all_colors[tmp_index]})
        tmp_index+=1

    # plot connections

    for i in range(len(groups)):
        group_ = groups[i]
        label_for_color = group_['belonging_to_a_cluster_by_cost']
        points = group_['points']
        number_of_points = len(points)
        # if number_of_points>2:
        #     print(1)

        colors = [label_color[label_for_color] for i in range(len(points))]
        x_ = [points[i][0] for i in range(len(points))]
        y_ = [points[i][1] for i in range(len(points))]
        scat = axs.scatter(x_, y_, c=colors)
        if plot_edges:
            number_of_points = len(points)

            for i in range(number_of_points):
                for j in range(number_of_points):
                    if i!=j:
                        x_line = [points[i][0], points[j][0]]
                        y_line = [points[i][1], points[j][1]]
                        axs.plot(x_line, y_line, color=label_color[label_for_color], marker='', linewidth=2, alpha=0.1)



    # N = len(np.unique(labels_of_clusters))
    # print('number_of_clusters {}'.format(N))
    #
    # norm = matplotlib.colors.Normalize(vmin=min(labels_of_clusters), vmax=max(labels_of_clusters))
    # m = cm.ScalarMappable(norm=norm, cmap=plt.cm.gist_ncar)
    #
    # all_colors = []
    # unique_labels = np.unique(labels_of_clusters)
    # for i in range(len(unique_labels)):
    #     all_colors.append(m.to_rgba(unique_labels[i]))
    # # random.shuffle(all_colors)
    #
    # label_color={}
    # tmp_index = 0
    # for lable_ in unique_labels:
    #     label_color.update({lable_:all_colors[tmp_index]})
    #     tmp_index+=1
    #
    # if plot_edges:
    #     number_of_points= len(x_)
    #     for i in range(number_of_points):
    #         for j in range(number_of_points):
    #             if i!=j:
    #                 if labels_of_clusters[i] == labels_of_clusters[j]:
    #                     x_line = [x_[i], x_[j]]
    #                     y_line = [y_[i], y_[j]]
    #                     axs.plot(x_line, y_line, color=label_color[labels_of_clusters[i]] ,marker='',linewidth=2, alpha=0.1)
    #
    #
    # colors = [label_color[labels_of_clusters[i]] for i in range(len(labels_of_clusters))]
    # scat = axs.scatter(x_, y_, c=colors)


def clustering_vertices_Hierarh(distance_matrix, n_clusters):

    clustering_alg = AgglomerativeClustering(n_clusters=n_clusters,
                                             metric='precomputed',
                                             linkage='average',
                                             # connectivity=conn_matrix
                                             )
    clustering = clustering_alg.fit(distance_matrix)
    return clustering.labels_


def clustering_vertices(distance_matrix, min_samples):
    # clustering_alg = DBSCAN(eps=eps, min_samples=min_samples,
    #                         metric='precomputed')
    clustering_alg = OPTICS(min_samples=min_samples,
                            metric='precomputed')
    clustering = clustering_alg.fit(distance_matrix)
    return clustering.labels_

# def clustering_vertices(points, metric, threshold):
#     cluster_lables = hcluster.fclusterdata(points, t=threshold, criterion="distance", metric=metric)
#     return cluster_lables
#

def space_distance(point_1, point_2):
    x_1 = point_1[0]
    y_1= point_1[1]

    x_2 = point_2[0]
    y_2 =point_2[1]

    return np.sqrt(np.square(x_1-x_2)+np.square(y_1-y_2))

def L_distance(point_1, point_2):
    return 1/np.absolute(point_1-point_2)

def get_min_samples_by_N(N):
    if N <= 5:
        return 3
    if 5 < N <= 1000:
        min_samples = 5
    if 1000<N<=1500:
        min_samples = 9
    if N>1500:
        min_samples = N / 100
    return int(min_samples)

def get_n_of_gloups_by_n_of_objects(N, max_group_size):
    result = N // max_group_size
    remains = N % max_group_size
    if remains == 0:
        return result
    else:
        return result+1


def plotsurf(x_vec, y_vec, func_):


    plt.rcParams["figure.figsize"] = [14, 7]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    axs = fig.add_subplot(111)

    rects_info = []

    x = []
    y = []
    highs = []

    for j in range(len(x_vec)-1):
        for k in range(len(y_vec)-1):
            x_1 = x_vec[j]
            x_2 = x_vec[j+1]
            y_1 = y_vec[k]
            y_2 = y_vec[k+1]
            x.append((x_2 + x_1) / 2)
            y.append((y_2 + y_1) / 2)
            rects_info.append([x_1, x_2, y_1, y_2])
            point_ = [(x_2 + x_1) / 2, (y_2 + y_1) / 2]
            highs.append(func_(point_))


    # x_ = search_points['x']
    # y_ = search_points['y']
    #
    # distrs_ = [[x_[i], y_[i]] for i in range(len(x_))]
    #
    # f_ = []
    # for j in range(len(x_)):
    #     f_.append(func_(x_[j], y_[j]))
    #
    # N = len(x_)
    #
    # phi_distance_matrix = np.zeros(shape=(N, N))
    # print('create conn matrix')
    # # all_f_dist= []
    # for i in range(N-1):
    #     print('{}/{}'.format(i, N-1))
    #     for j in range(i+1, N):
    #         dist_ = np.absolute(f_[i]-f_[j])
    #         phi_distance_matrix[i][j] = dist_
    #         phi_distance_matrix[j][i] = dist_
    #         # all_f_dist.append(dist_)
    # # plot_hist_of_list(all_f_dist)
    # # plot_hist_of_list(f_)
    # # pers = np.percentile(all_f_dist, 1)
    # # print('pers {}'.format(pers))
    # # points_= [[x_[i], y_[i]] for i in range(len(x_))]
    # # lables_of_points = clustering_vertices(points=points_,metric=space_distance,threshold=0.15)
    # # посчитали номер кластера для каждой точки относительно значений функции
    # lables_of_points = clustering_vertices(distance_matrix=phi_distance_matrix, min_samples=get_min_samples_by_N(N))
    #
    #
    # number_of_lost_points_cost_clustering = 0
    # # получить все кластеры, отбросив точки, которые в кластеры не попали
    # clusters = {}
    # for unique_label in np.unique(lables_of_points):
    #     if unique_label == -1:
    #         continue
    #     clusters.update({unique_label:[]})
    #
    # for label_, elem_ in zip(lables_of_points, distrs_):
    #     if label_ == -1:
    #         number_of_lost_points_cost_clustering += 1
    #         continue
    #     clusters[label_].append(elem_)
    #
    # # подсчет min phi в кластере
    # cluster_median_phi = {}
    # for key_ in clusters:
    #     min_ = np.min(clusters[key_])
    #     cluster_median_phi.update({key_: min_})
    #
    # # кластеризация по пространству  в каждом кластере, созданом по cost
    #
    # clusters_of_space = {}
    #
    # number_of_clusters_in_cost_clastering= len(clusters)
    # number_of_lost_points_space_clustering = 0
    #
    # for key_ in clusters:
    #     print('clustering by space {}/{}'.format(key_,number_of_clusters_in_cost_clastering))
    #     # cluster_ = clusters[key_]
    #     distrs_in_fixed_cluster = clusters[key_]
    #     n_ = len(distrs_in_fixed_cluster)
    #     space_distance_matrix = np.zeros(shape=(n_, n_))
    #     # print('create adj matrix')
    #     for i in range(n_ - 1):
    #         # print('{}/{}'.format(i, n_ - 1))
    #         for j in range(i + 1, n_):
    #             ro_12 = np.sqrt(np.square(distrs_in_fixed_cluster[i][0] - distrs_in_fixed_cluster[j][0]) + np.square(distrs_in_fixed_cluster[i][1] - distrs_in_fixed_cluster[j][1]))
    #             space_distance_matrix[i][j] = ro_12
    #             space_distance_matrix[j][i] = ro_12
    #
    #     clusters_in_cluster = {}
    #     lables_of_points_in_fixed_cluster = clustering_vertices(distance_matrix=space_distance_matrix,
    #                                                             min_samples=get_min_samples_by_N(n_))
    #     for unique_label in np.unique(lables_of_points_in_fixed_cluster):
    #         if unique_label == -1:
    #             continue
    #         clusters_in_cluster.update({unique_label: []})
    #
    #     for label_, elem_ in zip(lables_of_points_in_fixed_cluster, distrs_in_fixed_cluster):
    #         if label_ == -1:
    #             number_of_lost_points_space_clustering += 1
    #             continue
    #         clusters_in_cluster[label_].append(elem_)
    #
    #     clusters_of_space.update({key_: clusters_in_cluster})
    #
    # # создаем тройки или четверки из распределений
    # all_groups = []
    # for cost_cluster_key in clusters_of_space:
    #     for space_cluster_key in clusters_of_space[cost_cluster_key]:
    #         distrs_for_make_groups = clusters_of_space[cost_cluster_key][space_cluster_key]
    #         n_ = len(distrs_for_make_groups)
    #
    #         # либо рекурсивно запускаем процедуру либо делаем группы################################################
    #
    #         space_distance_matrix = np.zeros(shape=(n_, n_))
    #         # print('create adj matrix')
    #         for i in range(n_ - 1):
    #             # print('{}/{}'.format(i, n_ - 1))
    #             for j in range(i + 1, n_):
    #                 ro_12 = np.sqrt(
    #                     np.square(distrs_for_make_groups[i][0] - distrs_for_make_groups[j][0]) + np.square(
    #                         distrs_for_make_groups[i][1] - distrs_for_make_groups[j][1]))
    #                 space_distance_matrix[i][j] = ro_12
    #                 space_distance_matrix[j][i] = ro_12
    #
    #         # connectivity = np.zeros(shape=(n_, n_))
    #         # # print('create adj matrix')
    #         # for i in range(n_ - 1):
    #         #     # print('{}/{}'.format(i, n_ - 1))
    #         #     for j in range(i + 1, n_):
    #         #         X_i = distrs_for_make_groups[i]
    #         #         X_j = distrs_for_make_groups[j]
    #         #         f_i = func_(X_i[0], X_i[1])
    #         #         f_j = func_(X_j[0], X_j[1])
    #         #         degree_of_connectivity = 1/np.absolute(f_i - f_j)
    #         #         # if degree_of_connectivity>0.0:
    #         #         #
    #         #         # dist_ = np.absolute(f_i - f_j)
    #         #         connectivity[i][j] = 1.0
    #         #         connectivity[j][i] = 1.0
    #         n_groups_ = get_n_of_gloups_by_n_of_objects(n_, max_group_size=3)
    #
    #         local_groups = {}
    #         indexes_of_group = clustering_vertices_Hierarh(
    #                                     distance_matrix=space_distance_matrix,
    #                                     n_clusters=n_groups_)
    #         all_indexes = np.unique(indexes_of_group)
    #         for ind in all_indexes:
    #             if ind==-1:
    #                 print(1)
    #
    #         for index_of_group in indexes_of_group:
    #             local_groups.update({index_of_group:[]})
    #
    #         for i in range(len(distrs_for_make_groups)):
    #             el = distrs_for_make_groups[i]
    #             local_groups[indexes_of_group[i]].append(el)
    #
    #         # append local group to all_groups
    #         for group_index in local_groups:
    #             group_ = {
    #                 'points': local_groups[group_index],
    #                 'cost_function_at_a_given_scale': cluster_median_phi[cost_cluster_key],
    #                 'belonging_to_a_cluster_by_cost': cost_cluster_key
    #             }
    #             all_groups.append(group_)


    norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
    m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    tmp_len = len(rects_info)
    for i in range(len(rects_info)):
        # print("\r interation {} of {}".format(i, tmp_len), end='')
        plot_rect(axs, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],
                  m.to_rgba(highs[i]))
    print('')

    plt.colorbar(m, ax=axs)
    axs.set_xlim([-1.0, 1.0])
    axs.set_ylim([-1.0, 1.0])
    axs.set_xlabel(r"$x$")
    axs.set_ylabel(r'$y$')
    plt.title(r'$\varphi(x,y)$')

    # axs.scatter(x_, y_, color='r')
    # plot_graph2D(search_points, adj_matrix, axs,
    #              plot_vertexes=True,
    #              plot_edges=False)
    # plot_clusters(all_groups, list(clusters.keys()), axs, plot_edges=True)

    argmin_ = np.argmin(highs)
    print('min value = {} x= {}, y={}'.format(highs[argmin_], x[argmin_], y[argmin_]))

    print("plot response surface done")
    return axs