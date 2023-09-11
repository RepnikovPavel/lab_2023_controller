import torch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# from pyvis.network import Network

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
    mg = ModelGenerator(rules=config.rules,
                                cache_dir=config.Phi_cache_dir,
                                clear_cache=False)
    N = 1000
    all_p = []
    for i in range(N):
        all_p.append(torch.load(os.path.join(mg.cache_dir, 'distrib4D_{}.txt'.format(i))))

    adj_matrix = np.zeros(shape=(N, N))
    k_ = 0
    a_, h, f, coeff_list  = mg.shared_data['ahfcoeff_list']
    a_ = [el.numpy() for el in a_]
    for i in range(N-1):
        p_i = all_p[i]
        for j in range(i+1, N):
            p_j = all_p[j]
            ro_12 = ro_Distrib4D(p_i, p_j, a_shared=a_)
            adj_matrix[i][j] = ro_12
            adj_matrix[j][i] = ro_12
            k_ += 1
            print('\r{}/{}\t{}%'.format(k_+1, int(N*(N-1)/2),(k_+1)/(N*(N-1)/2)*100),end='')
    print('')
    torch.save(adj_matrix, config.Phi_adj_matrix)
    timer.stop()
    print(timer.get_execution_time()/60)
