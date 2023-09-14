import os
import pprint
import time

import numpy as np
import torch

import config
from SearchAlg.genetic_alg_general_functions import get_L, GetPhysicalLoss

from WinTermHelpers.query_helpers import *
from FileSystem.general_purpose_functions import *
from Alg.solving_algorithm import ModelGenerator
from FileSystem.storage import PStorage, SimResultsManager
from CustomModels.my_models import DistrMaker
from CustomModels.my_models import Integrator,renormolize_distribution
import config
from general import plot_policy_function, plot_trajectories,get_sim_results,plot_policy_function_with_trajectories
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function
import sys
from general_purpose_functions import time_mesuament
from inc_random_descent import insert_noise_to_p
import matplotlib.pyplot as plt


from_th_in_si_to_th_in_volt = config.translators_units_of_measurement['from_th_in_si_to_th_in_volt']
from_omega_in_si_to_omega_in_volt = config.translators_units_of_measurement['from_omega_in_si_to_omega_in_volt']
def apply_transofrm_of_units(x_,transform_):
    o_ = []
    for i in range(len(x_)):
        o_i=[]
        for j in range(len(x_[i])):
            o_i.append(transform_(x_[i][j]))
        o_.append(np.array(o_i))
    return o_

if __name__ == '__main__':
    # shared_data = ModelGenerator(rules=config.rules,
    #                              cache_dir=config.models_gen['pointwise_approx']['dir'],
    #                              clear_cache=False).shared_data
    # shared_integration_supports = Integrator(dir_=config.integrator_dir,
    #                                          shared_data=shared_data,
    #                                          clear_cache=False).shared_integration_supports

    # condition_of_break = np.asarray([
    #     config.theta_range,
    #     config.omega_range,
    #     [-9999.0, 9999.0],
    #     [-9999.0, 9999.0]
    # ])
    #
    #
    # timer = time_mesuament.Timer()
    # timer.start()
    # # загрузить все раcпределения
    # all_p = {}
    # for i in range(16):
    #     index_of_exe = str(i)
    #     pstorage_i = PStorage(dir_=config.p_storage_base_dir,
    #                           index_of_exe=str(index_of_exe),
    #                           clear_cache=False)
    #     n_of_p = pstorage_i.get_number_of_elements()
    #
    #     for j in range(n_of_p):
    #         element_path = pstorage_i.get_path_by_index(j)
    #         p_xi_eta_gamma = pstorage_i.load_by_index(j)
    #         all_p.update({element_path: p_xi_eta_gamma})
    # # загрузить значения L на этих распределениях
    # results_ = {}
    # dirs = os.listdir(config.p_sim_results_base_dir)
    # for i in range(len(dirs)):
    #     if not dirs[i].isnumeric():
    #         # print(dirs[i])
    #         continue
    #     sim_storage = SimResultsManager(base_dir_=config.p_sim_results_base_dir, index_of_exe=dirs[i],
    #                                     clear_cache=False)
    #     results_i = sim_storage.get_sim_results()
    #     for j in range(len(results_i)):
    #         result = results_i[j]
    #         key = result['source_path']
    #         results_.update({key: result['results']})
    # all_names = []
    # all_results = []
    # all_values = []
    # T = config.phys_sim_params['t_end']
    # for key in results_:
    #     # key is source path
    #     all_names.append(key)
    #     all_results.append(results_[key])
    #     all_values.append(GetPhysicalLoss(sim_results_=results_[key], T=T))
    # argmin = np.argmin(all_values)
    # print(all_values[argmin])
    # # plot_hist_of_list(all_values)
    # # проблема с различными результатами для одних и тех же распредлений
    #
    # # выбрать лучшее, начать случайный спуск
    # best_name = all_names[argmin]
    # L_best = all_values[argmin]
    # p_xi_eta_gamma = all_p[best_name]

    mg=  ModelGenerator(rules=config.incomplete_rules,
                                 cache_dir=config.Phi_cache_dir,
                                 clear_cache=False)
    shared_data = mg.shared_data
    shared_integration_supports = Integrator(dir_=config.Phi_cache_dir,
                                             shared_data=shared_data,
                                             clear_cache=False).shared_integration_supports

    # p_xi_eta_gamma = torch.load(config.inc_p0)
    # insert_noise_to_p(p_xi_eta_gamma,mask_of_noisy_rules=config.mask_for_rules_for_noise,shared_data=shared_data)

    # p_xi_eta_gamma = torch.load(config.inc_random_dec_file_path)
    # p_xi_eta_gamma = torch.load(config.Phi_descent_best_p_path)
    N=1000
    all_p = [torch.load(os.path.join(mg.cache_dir, 'distrib4D_{}.txt'.format(i))) for i in range(N)]
    nr = len(all_p[0].z_list)
    uniform_distrib_of_rules= np.ones(shape=(nr,))/nr
    a, h, f, coeff_list = mg.shared_data['ahfcoeff_list']
    for i in range(N):
        all_p[i] = renormolize_distribution(all_p[i],[a[el].detach().numpy() for el in range(len(a))],uniform_distrib_of_rules)
    p_xi_eta_gamma = all_p[0]

    # p_xi_eta_gamma = torch.load(config.inc_p0)
    # indexes_to_modify = np.concatenate([np.arange(start = 13, stop=35, dtype=np.uint32),
    #                                    np.array([2,9,37,44,3,10,38,45,4,11,39,46],dtype=np.uint32)])
    # p_xi_eta_gamma.modify_z_in_rule(indexes_to_modify)
    p_func = Integrator.make_policy_function(p_xi_eta_gamma=p_xi_eta_gamma,
                                             new_omega_list=shared_integration_supports['new_omega_list'],
                                             list_for_integrate=shared_integration_supports['list_for_integrate'],
                                             projection_to_x_y_info=shared_integration_supports[
                                                 'projection_to_x_y_info'],
                                             shared_Rects=shared_integration_supports['Rects'],
                                             shared_Grids=shared_integration_supports['Grids'],
                                             print_time_of_this_func=False
                                              )

    axs = plot_policy_function(mode_of_plot='map',
                         filepath_to_save_response_surface='',
                         p_func=p_func,
                         Grids=shared_integration_supports['Grids'],
                         block_canvas=False
                         )
    condition_of_break = np.asarray([
        config.theta_range,
        config.omega_range,
        [-9999.0, 9999.0],
        [-9999.0, 9999.0]
    ])
    psi = make_psi(policy_func=p_func,
                   translators_units_of_measurement=config.translators_units_of_measurement)

    sym_params_ =  {'x_1_range': [-0.9, 0.9], 'x_2_range': [-0.9,  0.9],  'n_x_for_sim': 20, 'n_y_for_sim': 20}
    simulation = make_simulation_for_one_policy_function(
        psi=psi,
        phys_sim_params=config.phys_sim_params,
        condition_of_break=condition_of_break,
        object_params=config.phys_params,
        use_an_early_stop=False
    )

    # results_ = get_sim_results(simulation=simulation,
    #                            phys_sim_params=config.phys_sim_params,
    #                            plot_tr_params=config.plot_trajectories_params,
    #                            units_translators=config.translators_units_of_measurement)
    T = config.phys_sim_params['t_end']
    # loss = get_L(p_xi_eta_gamma,shared_integration_supports,T,condition_of_break,sym_params_)
    # pprint.pprint(results_)
    # print('loss {}'.format(loss))


    results_ = get_sim_results(simulation=simulation,
                            phys_sim_params=config.phys_sim_params,
                            plot_tr_params=sym_params_,
                            units_translators=config.translators_units_of_measurement)




    plot_trajectories(simulation=simulation,
                      phys_sim_params=config.phys_sim_params,
                      plot_tr_params=sym_params_,
                      units_translators=config.translators_units_of_measurement,
                      make_animation=False)
    plt.show()
    # start_time = time.time()
    #
    # stop_time =  time.time()
    # print(stop_time-start_time)