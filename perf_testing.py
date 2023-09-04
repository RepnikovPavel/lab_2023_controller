import time
import numpy as np
import torch
from SearchAlg.genetic_alg_general_functions import get_L
from Alg.solving_algorithm import ModelGenerator
from CustomModels.my_models import Integrator
import config
from general import get_sim_results
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function

import cProfile
import pstats

if __name__ == '__main__':
    shared_data = ModelGenerator(rules=config.rules,
                                 cache_dir=config.models_gen['pointwise_approx']['dir'],
                                 clear_cache=False).shared_data
    shared_integration_supports = Integrator(dir_=config.integrator_dir,
                                             shared_data=shared_data,
                                             clear_cache=False).shared_integration_supports

    p_xi_eta_gamma = torch.load(config.random_decent_best_filepath)
    p_xi_eta_gamma.cast_to_float32()

    # with cProfile.Profile() as profile:
    condition_of_break = np.asarray([
        config.theta_range,
        config.omega_range,
        [-9999.0, 9999.0],
        [-9999.0, 9999.0]
    ],dtype=np.float32)
    start_time = time.time()
    T = config.phys_sim_params['t_end']
    loss = get_L(p_xi_eta_gamma, shared_integration_supports, T, condition_of_break)
    stop_time = time.time()
    print(stop_time-start_time)

    # pr_res = pstats.Stats(profile)
    # pr_res.sort_stats(pstats.SortKey.TIME)
    # pr_res.print_stats()
    # pr_res.dump_stats('profile_results.prof')


    #cd C:\Users\User\PycharmProjects\lab_2023
    #C:\Python3\Python_3_10_6\venvs\lab\Scripts\python.exe -m cProfile perf_testing.py
    #C:\Python3\Python_3_10_6\venvs\lab\Scripts\tuna.exe profile_results.prof
