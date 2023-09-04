import torch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network

from Alg.solving_algorithm import ModelGenerator
from CustomModels.my_models import Integrator, ro_Distrib4D
from FileSystem.general_purpose_functions import *
from FileSystem.storage import SimResultsManager, PStorage
import config

from SearchAlg.genetic_alg_general_functions import Loss, plot_hist_of_list

import scipy

from general_purpose_functions import time_mesuament
from general_purpose_functions.tensor_to_numpy import list_of_tensors_to_list_of_numpy


if __name__ == '__main__':
    timer = time_mesuament.Timer()
    timer.start()
    # шаг под вопросом: получить распредления внутри выпуклой оболочки, и тем самым создать новые кластеры

    # загружаем список распределений на которых нужно построить граф
    # этот список может содерджать любые распределения.
    # в том числе список распределений на i-м шаге сходимости к локальному минимуму

    # пусть нам дан список какихто расперделений. нам не важно каких.
    # посчитаем с помощью расстояния веса графа
    # создадим граф, сохраним его.

    # решим задачу кластеризации
    # найдем те кластера в которых значения Cost меньше всего
    # рекурсивно повторим процедуру
    shared_data = ModelGenerator(rules=config.rules,
                                 cache_dir=config.models_gen['pointwise_approx']['dir'],
                                 clear_cache=False).shared_data
    shared_integration_supports = Integrator(dir_=config.integrator_dir,
                                             shared_data=shared_data,
                                             clear_cache=False).shared_integration_supports

    a_shared = list_of_tensors_to_list_of_numpy(shared_data['ahfcoeff_list'][0])

    # load all p
    all_p = {}
    for i in range(16):
        index_of_exe = str(i)
        pstorage_i = PStorage(dir_=config.p_storage_base_dir,
                              index_of_exe=str(index_of_exe),
                              clear_cache=False)
        n_of_p = pstorage_i.get_number_of_elements()

        for j in range(n_of_p):
            element_path = pstorage_i.get_path_by_index(j)
            p_xi_eta_gamma = pstorage_i.load_by_index(j)
            all_p.update({element_path: p_xi_eta_gamma})

    # p_1 = pstorage.load_by_index(23)
    # p_2 = pstorage.load_by_index(22)
    # timer = time_mesuament.Timer()
    # timer.start()
    # for i in range(10000):
    #     print(i/(10000)*100)
    #     print(ro_Distrib4D(p_1, p_2, a_shared=a_shared))

    get_index_of_vertex_by_key_= {}
    get_key_by_index_of_vertex = {}
    # G = nx.Graph()
    k_ = 0
    for key_ in all_p:
        # G.add_node(k_, UTM=key_)
        get_index_of_vertex_by_key_.update({key_: k_})
        get_key_by_index_of_vertex.update({k_: key_})
        k_ += 1

    N = len(all_p)
    adj_matrix = np.zeros(shape=(N, N))
    for i in range(N-1):
        print('{}/{}'.format(i, N-1))
        timer.start_time_period()
        p_1 = all_p[get_key_by_index_of_vertex[i]]
        for j in range(i+1, N):

            p_2 = all_p[get_key_by_index_of_vertex[j]]
            ro_12 = ro_Distrib4D(p_1, p_2, a_shared=a_shared)
            adj_matrix[i][j] = ro_12
            adj_matrix[j][i] = ro_12
        print('elapsed time {}'.format(timer.stop_time_period()))

    graph_ = {
        'projections_to_Phi': all_p,
        'adj_matrix': adj_matrix,
        'get_index_of_vertex_by_key_': get_index_of_vertex_by_key_,
        'get_key_by_index_of_vertex': get_key_by_index_of_vertex
    }
    mkdir_if_not_exists(config.graph_with_projections_to_Phi_base_dir)
    torch.save(graph_, os.path.join(config.graph_with_projections_to_Phi_base_dir, 'projections_to_Phi.txt'))

    # G.add_nodes_from([
    #     (4, {"path": "dir_1"}),
    #     (5, {"path": "dir_1"}),
    # ])
    # G.add_edge(1, 2, weight = 1.0)
    # G.add_edge(4,5, weight=2.0)
    # net= Network(notebook=True)
    # net.from_nx(G)
    # net.show('example.html')
    # print(G.nodes[4])
    # print(G[4])
    # nx.draw(G)
    # plt.show()
    timer.stop()
    print(timer.get_execution_time())
