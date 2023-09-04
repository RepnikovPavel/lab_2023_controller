import os

import numpy as np
import torch

import config
from WinTermHelpers.query_helpers import *
from FileSystem.general_purpose_functions import *
from Alg.solving_algorithm import ModelGenerator
from FileSystem.storage import ZStorage
from CustomModels.my_models import DistrMaker
from CustomModels.my_models import Integrator
import config
from general import plot_policy_function, plot_trajectories
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function

if __name__ == '__main__':
    # 'init_db' 'delete_db'
    # mode_of_pr ='init_db'
    # if mode_of_pr == 'init_db':
    #     os.system("sqlite3 {0} {1}".format(
    #         WinPath(config.SQLite3_alg_db_path),
    #         WinString('.read '+WinPath(config.SQLite3_init_script))
    #     ))
    # if mode_of_pr =='delete_db':
    #     delete_file_if_exists(config.SQLite3_alg_db_path)

    ############################## create p \in \Phi ###########################################

    # create distrib
    gen_config = config.models_gen['pointwise_approx']
    mg = ModelGenerator(rules=config.rules,
                        cache_dir=gen_config['dir'],
                        clear_cache=False)
    # zlist = mg.CreateModelFromScratch()
    zstorage = ZStorage(dir_=config.z_storage_dir,
                        clear_cache=False)
    # zstorage.append(zlist)

    ############################### make policy function #####################################

    # zstorage = ZStorage(config.z_storage_dir)
    # zlist = zstorage.laod_by_index(index=2)
    #
    # shared_data = ModelGenerator(rules=config.rules,
    #                              cache_dir=config.models_gen['pointwise_approx']['dir'],
    #                              clear_cache=False).shared_data
    # shared_integration_supports = Integrator(dir_=config.integrator_dir,
    #                                          shared_data=shared_data,
    #                                          clear_cache=False).shared_integration_supports
    #
    # p_xi_eta_gamma = DistrMaker.make_distrib4D(x_={
    #     'z_list': zlist,
    #     'np_omega': shared_data['np_omega']
    # })
    # p_func = Integrator.make_policy_function(p_xi_eta_gamma=p_xi_eta_gamma,
    #                                          new_omega_list=shared_integration_supports['new_omega_list'],
    #                                          list_for_integrate=shared_integration_supports['list_for_integrate'],
    #                                          projection_to_x_y_info=shared_integration_supports[
    #                                              'projection_to_x_y_info'],
    #                                          shared_Rects=shared_integration_supports['Rects'],
    #                                          shared_Grids=shared_integration_supports['Grids'],
    #                                          print_time_of_this_func=True
    #                                          )
    # condition_of_break = np.asarray([
    #     config.theta_range,
    #     config.omega_range,
    #     [-9999.0, 9999.0],
    #     [-9999.0, 9999.0]
    # ])
    # psi = make_psi(policy_func=p_func,
    #                translators_units_of_measurement=config.translators_units_of_measurement)
    #
    # simulation = make_simulation_for_one_policy_function(
    #     psi=p_func,
    #     phys_sim_params=config.phys_sim_params,
    #     condition_of_break=condition_of_break,
    #     object_params=config.phys_params
    # )
    # plot_trajectories(simulation=simulation,
    #                   phys_sim_params=config.phys_sim_params,
    #                   plot_tr_params=config.plot_trajectories_params,
    #                   units_translators=config.translators_units_of_measurement)
