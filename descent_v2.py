import torch
import config
import numpy as np
import matplotlib.pyplot as plt
import os
from Alg.solving_algorithm import ModelGenerator
from CustomModels.my_models import Integrator
from CustomModels.my_models import weighted_amount
from aml.plotting import *
from Losses.Losses import *
from sklearn import decomposition
from tqdm import tqdm
from scipy.spatial import ConvexHull
N = 1000

mg = ModelGenerator(rules=config.rules,
                            cache_dir=config.Phi_cache_dir,
                            clear_cache=False)
vectors = torch.load(config.Phi_vector_representation)
all_p = [torch.load(os.path.join(mg.cache_dir, 'distrib4D_{}.txt'.format(i))) for i in range(N)]
all_v = torch.load(os.path.join(config.task_dir, 'L2_for_Phi.txt'))
dim = len(vectors[0])
# vectors = vectors - np.mean(vectors,axis = 0)


mg = ModelGenerator(rules=config.rules,
                            cache_dir=config.Phi_cache_dir,
                            clear_cache=False)
shared_integration_supports = Integrator(dir_=config.integrator_dir,
                                        shared_data=mg.shared_data,
                                        clear_cache=True).shared_integration_supports

K = 10
p_n = all_p[np.random.randint(0,len(all_p))]
hull = ConvexHull(vectors)
for simplex in hull.simplices:
    
for ITER in range(10):
    indexes = np.random.randint(0,len(all_p),K)
    # choose supports points
    p_j_set = [all_p[el] for el in indexes]

    # solve K one dimensional optimization problems and choose best solution
    # for j in range(K):
    #     p_j = p_j_set[j]
    #     eps_vec = np.linspace(0.0,1.0,10)
    #     l_vec = np.zeros(shape=(10,))
    #     for q in range(len(eps_vec)):
    #         eps_ = eps_vec[q]
    #         p_tilde = weighted_amount(list_of_distributions=[p_n, p_j],
    #                         alpha_list=np.asarray([eps_, 1.0-eps_]))
    #         l_vec[q] = get_L2_Distrib4D(p_tilde, shared_integration_supports)
    #     plt.plot(eps_vec, l_vec)
    #     plt.show()

