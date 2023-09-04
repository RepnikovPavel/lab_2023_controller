import os
import time

import numpy as np
import torch

import config
from WinTermHelpers.query_helpers import *
from FileSystem.general_purpose_functions import *
from Alg.solving_algorithm import ModelGenerator
from FileSystem.storage import PStorage, SimResultsManager
from CustomModels.my_models import DistrMaker
from CustomModels.my_models import Integrator
import config
from general import plot_policy_function, plot_trajectories, get_sim_results
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function
import sys
from general_purpose_functions import time_mesuament

import operator

from SearchAlg.genetic_alg_general_functions import GetPhysicalLoss, get_L

if __name__ == '__main__':
    index_of_exe = sys.argv[1]
    # index_of_exe = '1'
    timer = time_mesuament.Timer()
    timer.start()

    shared_data = ModelGenerator(rules=config.rules,
                                 cache_dir=config.models_gen['pointwise_approx']['dir'],
                                 clear_cache=False).shared_data
    shared_integration_supports = Integrator(dir_=config.integrator_dir,
                                             shared_data=shared_data,
                                             clear_cache=False).shared_integration_supports

    # simulation
    sim_storage = SimResultsManager(base_dir_=config.p_sim_results_base_dir, index_of_exe=index_of_exe,
                                    clear_cache=True)

    pstorage_i = PStorage(dir_=config.p_storage_base_dir,
                          index_of_exe=str(index_of_exe),
                          clear_cache=False)
    n_of_p = pstorage_i.get_number_of_elements()
    current_elapsed_time = 0.0
    T = config.phys_sim_params['t_end']
    for j in range(n_of_p):
        timer.start_time_period()
        element_path = pstorage_i.get_path_by_index(j)
        # print('path ={}'.format(element_path))
        p_xi_eta_gamma = pstorage_i.load_by_index(j)
        p_func = Integrator.make_policy_function(p_xi_eta_gamma=p_xi_eta_gamma,
                                                 new_omega_list=shared_integration_supports['new_omega_list'],
                                                 list_for_integrate=shared_integration_supports['list_for_integrate'],
                                                 projection_to_x_y_info=shared_integration_supports[
                                                     'projection_to_x_y_info'],
                                                 shared_Rects=shared_integration_supports['Rects'],
                                                 shared_Grids=shared_integration_supports['Grids'],
                                                 print_time_of_this_func=False
                                                 )
        condition_of_break = np.asarray([
            config.theta_range,
            config.omega_range,
            [-9999.0, 9999.0],
            [-9999.0, 9999.0]
        ])
        psi = make_psi(policy_func=p_func,
                       translators_units_of_measurement=config.translators_units_of_measurement)

        simulation = make_simulation_for_one_policy_function(
            psi=psi,
            phys_sim_params=config.phys_sim_params,
            condition_of_break=condition_of_break,
            object_params=config.phys_params
        )
        results_ = get_sim_results(simulation=simulation,
                                   phys_sim_params=config.phys_sim_params,
                                   plot_tr_params=config.plot_trajectories_params,
                                   units_translators=config.translators_units_of_measurement)
        n_good = results_['n_good']
        n_bad = results_['n_bad']
        n_early = results_['n_early']
        n_all = n_good + n_bad + n_early

        # print('physical loss {}'.format(GetPhysicalLoss(results_,T)))
        # print('get L {}'.format(get_L(p_xi_eta_gamma, shared_integration_supports, T, condition_of_break)))
        # raise SystemExit
        sim_storage.write_results(source_distrib_path=element_path, results=results_)
        current_elapsed_time += timer.stop_time_period()
        print(
            'element {}/{} n_good {}/{} n_bad {}/{} n_early {}/{} elapsed time {} sek\t'.format(j,
                                                                                    n_of_p - 1,
                                                                                    n_good,
                                                                                    n_all,
                                                                                    n_bad,
                                                                                    n_all,
                                                                                    n_early,
                                                                                    n_all,
                                                                                    current_elapsed_time)
        )



    timer.stop()
    print('exe has completed its work with time {} m'.format(timer.get_execution_time() / 60))
