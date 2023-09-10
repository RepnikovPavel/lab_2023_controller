from Alg.solving_algorithm import ModelGenerator
from FileSystem.storage import PStorage
from CustomModels.my_models import DistrMaker
from CustomModels.my_models import Integrator
import config
import sys
from typing import List
from general_purpose_functions import time_mesuament
import torch
import os
import pathos
from aml.batching import *
from copy import deepcopy
# from pathos.multiprocessing import ProcessingPool
import multiprocess as mp

def gen_new_distributions(index_of_process:int,indexes:List[int],model_generator:ModelGenerator):
    N = len(indexes)
    timer = time_mesuament.Timer()
    timer.start()
    shared_data = model_generator.shared_data
    cache_dir = model_generator.cache_dir
    n = 1
    S_=  0.0
    for i in range(N):
        timer.start_time_period()
        # learn existing rules
        z_list_0 = model_generator.CreateModelFromScratch(is_approx=True,gr_steps = 2000,lr=0.01,print_tmp_cons_and_loss=False)
        p_0 = DistrMaker.make_distrib4D(x_={
            'z_list': z_list_0,
            'np_omega': shared_data['np_omega']
        })
        index_of_element = indexes[i]
        torch.save(p_0, os.path.join(cache_dir, 'distrib4D_{}.txt'.format(index_of_element)))
        time_of_iter = timer.stop_time_period()
        S_ += time_of_iter
        estimation_of_all_time = (S_/n)*N
        time_left = estimation_of_all_time -  timer.get_time_from_start()
        if index_of_process == 0:
            print('time left {} m'.format(time_left / 60))
        n += 1 
    timer.stop()
    time_for_all = timer.get_execution_time()
    print('the process has completed its work with time {} m'.format( time_for_all/ 60))

if __name__ == '__main__':
    N = 1000
    NUMBER_OF_PROCESSORS = 16
    mg = ModelGenerator(rules=config.rules,
                                cache_dir=config.Phi_cache_dir,
                                clear_cache=True)
    all_indexes = np.arange(start=0,stop=N)
    batches, rest = make_batches(vector_of_numbers=all_indexes,
                            num_of_batches=NUMBER_OF_PROCESSORS)
    timer = time_mesuament.Timer()
    timer.start()
    processes = []
    for i in range(NUMBER_OF_PROCESSORS):
        # try_step(p_best,L_best,batches[i],all_distribs,gr_step_indx,shared_output,i)
        p = mp.Process(target=gen_new_distributions, args=(i, batches[i], deepcopy(mg)))
        processes.append(p)
        p.start()

    if len(rest) > 0:
        rest_p = mp.Process(target=gen_new_distributions, args=(NUMBER_OF_PROCESSORS, rest, deepcopy(mg)))
        processes.append(rest_p)
        rest_p.start()

    for i in range(NUMBER_OF_PROCESSORS):
        processes[i].join()
    timer.stop()
    print('total time {} m'.format(timer.get_execution_time()/60))

    
# index_of_exe = sys.argv[1]
# # index_of_exe ='test'
# N = 200
# print('exe {} approximate execution time {} m'.format(index_of_exe, 16 * N * 40 / 60 / 7))
# timer = time_mesuament.Timer()
# timer.start()

# shared_data = ModelGenerator(rules=config.rules,
#                              cache_dir=config.models_gen['pointwise_approx']['dir'],
#                              clear_cache=False).shared_data
# shared_integration_supports = Integrator(dir_=config.integrator_dir,
#                                          shared_data=shared_data,
#                                          clear_cache=False).shared_integration_supports
# gen_config = config.models_gen['pointwise_approx']

# pstorage = PStorage(dir_=config.p_storage_base_dir,
#                     index_of_exe=index_of_exe,
#                     clear_cache=False)

# mg = ModelGenerator(rules=config.rules,
#                     cache_dir=gen_config['dir'],
#                     clear_cache=False)

# for i in range(N):
#     zlist = mg.CreateModelFromScratch()
#     p_xi_eta_gamma = DistrMaker.make_distrib4D(x_={
#         'z_list': zlist,
#         'np_omega': shared_data['np_omega']
#     })
#     pstorage.append(p_xi_eta_gamma)

# timer.stop()
# print('the process {} has completed its work with time {} m'.format(index_of_exe, timer.get_execution_time() / 60))
