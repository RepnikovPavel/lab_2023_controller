import numpy as np
import torch 
from numba import jit
from CustomModels.my_models import *
from Alg.solving_algorithm import ModelGenerator
import config
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function
from general import WMA

@jit(nopython = True)
def minusReward(solution:np.array,rho_max:float,tau:float,t_end:float)->float:
    theta_vec = solution[:,0]
    omega_vec = solution[:,1]
    T_ = len(theta_vec)
    rho_i = np.sqrt(np.square(theta_vec) + np.square(omega_vec))
    r_vec = np.zeros(shape=(T_-1,),dtype=np.float32)
    for t in range(1, T_):
        r_vec[t-1] = 1.0/(rho_i[t]/rho_max*np.absolute(t*tau/t_end)+0.001)
    r_vec =  r_vec*T_
    return -r_vec


def get_L_Distrib4D(p_: Distrib4D, shared_integration_supports):
    
    p_func = Integrator.make_policy_function(p_xi_eta_gamma=p_,
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
        object_params=config.phys_params, use_an_early_stop=False,action_check=False
    )
    results_ = get_sim_results(simulation=simulation,
                               phys_sim_params=config.phys_sim_params,
                               plot_tr_params=config.descent_start_points_params,
                               units_translators=config.translators_units_of_measurement)
    times = results_['times']
    solutions = results_['solutions']
    Losses = np.zeros(shape=(len(solutions),))
    tau = config.phys_sim_params['tau']
    rho_max = np.sqrt(config.fuzzy_inf_params['th_max']**2 + config.fuzzy_inf_params['omega_max']**2)
    for i in range(len(solutions)):
        s_i = solutions[i]
        t_end = times[i]
        L_i = np.sum(minusReward(s_i,rho_max,tau,t_end))
        Losses[i] = L_i
    total_loss = np.mean(Losses)
    return total_loss 


def get_L2_Distrib4D(p_: Distrib4D, shared_integration_supports):
    
    p_func = Integrator.make_policy_function(p_xi_eta_gamma=p_,
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
        object_params=config.phys_params, use_an_early_stop=False,action_check=False
    )
    results_ = get_sim_results(simulation=simulation,
                               phys_sim_params=config.phys_sim_params,
                               plot_tr_params=config.descent_start_points_params,
                               units_translators=config.translators_units_of_measurement)
    times = results_['times']
    solutions = results_['solutions']
    Losses = np.zeros(shape=(len(solutions),))
    tau = config.phys_sim_params['tau']
    rho_max = np.sqrt(config.fuzzy_inf_params['th_max']**2 + config.fuzzy_inf_params['omega_max']**2)
    for i in range(len(solutions)):
        s_i = solutions[i]
        t_end = times[i]
        # L_i = np.absolute(s_i[-1,2]- s_i[0,2])
        # L_i = np.sum(minusReward(s_i,rho_max,tau,t_end))
        velocity_ = s_i[-10:,3]
        mean_last_coord_velocity =  np.absolute(WMA(velocity_))
        Losses[i] = mean_last_coord_velocity
    total_loss = np.mean(Losses)
    return total_loss