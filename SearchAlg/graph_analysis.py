import torch
import os
import config
from SearchAlg.genetic_alg_general_functions import plot_hist_of_list
import numpy as np

if __name__ == '__main__':
    graph_ = torch.load(os.path.join(config.graph_with_projections_to_Phi_base_dir, 'projections_to_Phi.txt'))
    projections_to_Phi = graph_['projections_to_Phi']
    adj_matrix = graph_['adj_matrix']
    get_index_of_vertex_by_key_ = graph_['get_index_of_vertex_by_key_']
    get_key_by_index_of_vertex = graph_['get_key_by_index_of_vertex']
    N = adj_matrix.shape[0]
    all_distances = []
    for i in range(N-1):
        for j in range(i+1, N):
            all_distances.append(adj_matrix[i][j])
    # гистограмма отличается от гистограммы для трехмерных точек, равномерно распределенных
    print(np.percentile(all_distances, q=1))
    print(np.mean(all_distances))
    print(np.median(all_distances))
    plot_hist_of_list(all_distances)


    # 3d uniform distances
    # linear_rands1 = np.random.uniform(low=0, high=1.0, size=1000)
    # linear_rands2 = np.random.uniform(low=0, high=1.0, size=1000)
    # linear_rands3 = np.random.uniform(low=0, high=1.0, size=1000)
    # dstn = []
    # for i in range(linear_rands1.shape[0]-1):
    #     print(i)
    #     for j in range(i+1, linear_rands1.shape[0]):
    #         d1 = np.square(linear_rands1[i]-linear_rands1[j])
    #         d2 = np.square(linear_rands2[i]-linear_rands2[j])
    #         d3 = np.square(linear_rands3[i] - linear_rands3[j])
    #         ro_12 = np.sqrt(d1+d2+d3)
    #         dstn.append(ro_12)
    # plot_hist_of_list(dstn)
