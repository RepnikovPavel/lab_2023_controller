from typing import List
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import generalized_box_iou_loss

import config
from CustomModels.my_models import Integrator
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function
from general import get_sim_results


def Loss(v_T: List[float], codes_of_sim: List[int]):
    if len(codes_of_sim) == 0:
        print('in Loss len(codes_of_sim)==0')
        raise SystemExit
    else:

        # num_of_bad = np.sum(codes_of_sim)
        # num_of_good = len(v_T)-num_of_bad
        # good_mask = 1-np.asarray(codes_of_sim)
        # v_i_vec = np.absolute(v_T)
        # v_max = 1.0
        # if num_of_good == 0:
        #     return np.nan
        # loss = np.sum(v_i_vec/np.abs(v_max-v_i_vec)*good_mask)/num_of_good

        # num_of_bad = np.sum(codes_of_sim)
        # num_of_good = len(v_T)-num_of_bad
        # good_mask = 1-np.asarray(codes_of_sim)
        # v_i_vec = np.absolute(v_T)
        #
        # if num_of_good == 0:
        #     return np.infty
        # loss = np.sum(v_i_vec*good_mask)/num_of_good
        # return loss
        n = len(v_T)
        num_of_bad = np.sum(codes_of_sim)
        num_of_good = n-num_of_bad
        loss = 1/(1+np.exp(-num_of_bad/n))
        return loss



def PhysicalLoss(x_end, v_end, a_end, t_end, T):

    # TODO: решить проблему с разными порядками величин
    x_p = x_end + v_end * (T - t_end) + a_end * (T - t_end) ** 2 / 2
    v_p = v_end + a_end * (T - t_end)

    lx = np.abs(x_p - 0.0)
    lv = np.abs(v_p - 0.0)
    #10 is scale factor
    # lv = np.abs(v_p-0.0)+np.maximum(0.0, np.abs(v_p-0.0)*(t_end-T/2))

    # lv=0.0
    # la = np.maximum(0.0, np.abs(a_end-0.0)*(t_end-T/2))
    # la = 0.0
    la = np.abs(a_end - 0.0)
    L_ = lx + lv + la
    # L_ = lv

    return L_

def AGGPhysicalLoss(losses):
    return np.mean(losses)


tr_ = config.translators_units_of_measurement
omega_from_v_to_si = tr_['from_omega_in_volt_to_omega_in_si']
a_om = omega_from_v_to_si(-0.1)
b_om = omega_from_v_to_si(0.1)
def GetPhysicalLoss(sim_results_,T):
    x_end_vec = sim_results_['last_y']
    v_end_vec = sim_results_['last_v']
    # a_end_vec = sim_results_['last_a']
    t_end_vec = sim_results_['times']
    th_end_vec = sim_results_['last_th']
    omega_end_vec = sim_results_['last_omega']
    all_omega = sim_results_['all_omega']
    all_th = sim_results_['all_th']
    codes = sim_results_['markers']
    # all_alpha_vec = []
    # all_alpha_loss = []
    # for i in range(len(all_th)):
    #     th_vec = all_th[i]
    #     omega_vec = all_omega[i]
    #     f_vec = []
    #     rho_vec = []
    #     for j in range(len(all_th[i])-1):
    #         p_i_1 = np.array([th_vec[j+1],omega_vec[j+1]]) 
    #         p_i = np.array([th_vec[j],omega_vec[j]])
    #         diff = p_i_1-p_i
    #         f_vec.append(diff/np.linalg.norm(diff))
    #         rho_vec.append(p_i/np.linalg.norm(p_i))
    #     alpha_vec = []
    #     alpha_loss_vec = []
    #     for j in range(len(f_vec)):
    #         alpha_vec.append(np.arccos(np.dot(f_vec[j],rho_vec[j])))
    #         alpha_loss_vec.append(np.pi-alpha_vec[j])
    #     all_alpha_vec.append(alpha_vec)
    #     all_alpha_loss.append(alpha_loss_vec)
        # plt.plot(alpha_vec)
        # plt.show()

    # from_v_in_volt_to_v_in_si = translators_units_of_measurement['from_v_in_volt_to_v_in_si']
    # from_th_in_si_to_th_in_volt = translators_units_of_measurement['from_th_in_si_to_th_in_volt']
    # from_omega_in_si_to_omega_in_volt = translators_units_of_measurement['from_omega_in_si_to_omega_in_volt']


    # l_vec = np.zeros(shape=(len(all_th),))
    # for i in range(len(all_th)):
    #     # if codes[i] == 1:
    #     th_end = all_th[i][-1]
    #     omega_end = all_omega[i][-1]
    #     l_vec[i] = np.sqrt(np.square(th_end) + np.square(omega_end))
    # L = np.mean(l_vec)

    L = 0.0
    # alpha_vecs = []
    l_vec = np.zeros(shape=(len(all_th),))
    for i in range(len(all_th)):
        # if codes[i] == 1:
        th_vec = all_th[i]
        omega_vec = all_omega[i]
        # print(th_vec[0])
        # if a_om<omega_vec[0] < b_om:
        #     continue
        f_vec = np.zeros(shape=(len(th_vec)-1, 2))
        rho_vec = np.zeros(shape=(len(th_vec)-1, 2))
        for j in range(len(th_vec)-1):
            p_i_1 = np.array([th_vec[j+1],omega_vec[j+1]]) 
            p_i = np.array([th_vec[j],omega_vec[j]])
            diff = p_i_1-p_i
            f_vec[j]=(diff/np.linalg.norm(diff))
            rho_vec[j]=(p_i/np.linalg.norm(p_i))
        # alpha_vec = np.zeros(shape=(len(f_vec)-1,))
        alpha_loss_vec = np.zeros(shape=(len(f_vec),))
        for j in range(len(f_vec)):
            # alpha_vec.append(np.arccos(np.dot(f_vec[j],rho_vec[j])))
            alpha_loss_vec[j] = (np.pi-np.arccos(np.dot(f_vec[j],rho_vec[j])))
        # alpha_vecs.append(alpha_loss_vec)
        # l_with_nan = np.sum(np.square(alpha_loss_vec)) 
        # l_i = np.nanmean((np.square(alpha_loss_vec)))
        l_i = (np.square(alpha_loss_vec[-1]))
        l_vec[i] = l_i 
        # if np.isnan(l_i):
        #     tmp_ = np.where(alpha_loss_vec == np.nan)
        #     print(1)
        # print(l_i)
        L += l_i
    L = np.percentile(l_vec,70)
            # all_alpha_vec.append(alpha_vec)
            # all_alpha_loss.append(alpha_loss_vec)

    # fig,axs = plt.subplots(1,1)
    # for i in range(len(all_th)):
    #     axs.plot(alpha_vecs[i])
    # plt.show()
    
    # fig,axs = plt.subplots(1,1)
    # for i in range(len(all_th)):
    #     axs.plot(all_th[i],all_omega[i])
    # plt.show()

    # 1 - bad
    # 0 - end of sim without falling
    # 2 - early stopping
    # losses = np.zeros(shape=(len(x_end_vec),))
    # good_losses = []
    # bad_a = []
    # ctr_ = 0
    # n_good= 0
    # n_bad = 0
    # for i in range(len(x_end_vec)):
    #     if codes[i] != 1:
    #         good_losses.append(PhysicalLoss(x_end_vec[i], v_end_vec[i], a_end_vec[i], t_end_vec[i], T))
    #         n_good += 1
    #     else:
    #         bad_a.append(a_end_vec[i])
    #     ctr_ += 1
    # L = 0.0
    # # if n_good == 0:
    # #     L = ((ctr_-n_good)/(ctr_))*(1.0/np.mean(np.absolute(bad_a)))
    # # else:
    # #     L = np.maximum((np.sum(good_losses)/n_good),1.0)*((ctr_-n_good)/(ctr_))*(1.0/np.mean(np.absolute(bad_a)))
    # L = ((ctr_-n_good)/(ctr_))
    # L = 0
    # for i in range(len(codes)):
    #     code_ = codes[i]
    #     if code_ == 1:
    #         L+=1
    # L = np.sqrt(np.sum(np.square(th_end_vec)+ np.square(omega_end_vec)))



    return L

def GetLossForPlot(sim_results_,T):
    x_end_vec = sim_results_['last_y']
    v_end_vec = sim_results_['last_v']
    # a_end_vec = sim_results_['last_a']
    t_end_vec = sim_results_['times']
    th_end_vec = sim_results_['last_th']
    omega_end_vec = sim_results_['last_omega']
    all_omega = sim_results_['all_omega']
    all_th = sim_results_['all_th']
    codes = sim_results_['markers']

    L = 0.0
    alpha_vecs = []
    loss_per_trajectory = []
    for i in range(len(all_th)):
        # if codes[i] == 1:
        th_vec = all_th[i]
        omega_vec = all_omega[i]
        # print(th_vec[0])
        # if a_om<omega_vec[0] < b_om:
        #     continue
        f_vec = np.zeros(shape=(len(th_vec)-1, 2))
        rho_vec = np.zeros(shape=(len(th_vec)-1, 2))
        for j in range(len(th_vec)-1):
            p_i_1 = np.array([th_vec[j+1],omega_vec[j+1]]) 
            p_i = np.array([th_vec[j],omega_vec[j]])
            diff = p_i_1-p_i
            f_vec[j]=(diff/np.linalg.norm(diff))
            rho_vec[j]=(p_i/np.linalg.norm(p_i))
        # alpha_vec = np.zeros(shape=(len(f_vec)-1,))
        alpha_loss_vec = np.zeros(shape=(len(f_vec),))
        for j in range(len(f_vec)):
            # alpha_vec.append(np.arccos(np.dot(f_vec[j],rho_vec[j])))
            alpha_loss_vec[j] = (np.pi-np.arccos(np.dot(f_vec[j],rho_vec[j])))
        alpha_vecs.append(alpha_loss_vec)
        # l_with_nan = np.sum(np.square(alpha_loss_vec)) 
        l_i = np.nanmean(np.square(alpha_loss_vec)) 
        # if np.isnan(l_i):
        #     tmp_ = np.where(alpha_loss_vec == np.nan)
        #     print(1)
        # print(l_i)
        loss_per_trajectory.append(l_i)
        L += l_i

    
    # fig,axs = plt.subplots(1,1)
    # for i in range(len(all_th)):
    #     axs.plot(alpha_vecs[i])
    # plt.show()

    return alpha_vecs

def get_L(  p_new,
            shared_integration_supports,
            T,
            condition_of_break,
            sym_params_
          ):
    # проверить полученную точку

    p_func = Integrator.make_policy_function(p_xi_eta_gamma=p_new,
                                             new_omega_list=shared_integration_supports['new_omega_list'],
                                             list_for_integrate=shared_integration_supports['list_for_integrate'],
                                             projection_to_x_y_info=shared_integration_supports[
                                                 'projection_to_x_y_info'],
                                             shared_Rects=shared_integration_supports['Rects'],
                                             shared_Grids=shared_integration_supports['Grids'],
                                             print_time_of_this_func=False
                                             )

    psi = make_psi(policy_func=p_func,
                   translators_units_of_measurement=config.translators_units_of_measurement)

    simulation = make_simulation_for_one_policy_function(
        psi=psi,
        phys_sim_params=config.phys_sim_params,
        condition_of_break=condition_of_break,
        object_params=config.phys_params, use_an_early_stop=True
    )

    results_ = get_sim_results(simulation=simulation,
                               phys_sim_params=config.phys_sim_params,
                               plot_tr_params=sym_params_,
                               units_translators=config.translators_units_of_measurement)

    L_of_p_new = GetPhysicalLoss(sim_results_=results_, T=T)

    return L_of_p_new



def plot_hist_of_list(x_:List[float]):
    fig, ax = plt.subplots()

    x = []
    for i in range(len(x_)):
        if np.isnan(x_[i]):
            continue
        else:
            x.append(x_[i])
    x = np.asarray(x)
    q25, q75 = np.percentile(x, [25, 75])
    q99 = np.percentile(x, 99)
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((np.max(x) - np.min(x)) / bin_width)
    # bins = 1000
    print('num of bins = {}'.format(bins))
    ax.hist(x, bins=bins)
    plt.show(block=True)