import matplotlib
from matplotlib import cm
from torchvision.ops import box_iou
import copy
import pprint
import warnings
from general import plot_empty_rect
from general import plot_rect
import torch
import os
import config
from Alg.solving_algorithm import ModelGenerator
from CustomModels import my_models
from CustomModels.my_models import ro_Distrib4D
from FileSystem.storage import SimResultsManager, PStorage
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function
from general import get_sim_results
from general_purpose_functions import time_mesuament
from SearchAlg.genetic_alg_general_functions import Loss, plot_hist_of_list, GetPhysicalLoss, get_L ,GetLossForPlot
import numpy as np
from CustomModels.my_models import weighted_amount, Integrator, DistrMaker
from general import plot_vec

import matplotlib.pyplot as plt
import base64
from io import BytesIO
import multiprocessing as mp
from typing import Tuple, List

from general import compute_gauss
from general import plot_policy_function_with_trajectories,plot_policy_function

warnings.filterwarnings('ignore')


def calc_eps(adj_matrix):
    all_distances = []
    N = len(adj_matrix)
    for i in range(N - 1):
        for j in range(i + 1, N):
            all_distances.append(adj_matrix[i][j])
    eps = np.percentile(all_distances, q=1)
    return eps


def plot_loss_along_the_line(p_1, p_2, T, condition_of_break):
    N = 20
    eps_vec = np.linspace(start=0.0, stop=1.0, num=N)
    loss_vec = []
    # best_p, best_v = get_better_p(p_1, p_2, T)
    # print(best_v)
    for i in range(len(eps_vec)):
        print('{}/{}'.format(i + 1, len(eps_vec)))
        eps_ = eps_vec[i]
        p_new = weighted_amount(list_of_distributions=[p_1, p_2],
                                alpha_list=np.asarray([1 - eps_, eps_]))
        L_of_p_new = get_L(p_new, shared_integration_supports, T, condition_of_break)
        loss_vec.append(L_of_p_new)
    print(loss_vec)
    plot_vec(x=eps_vec, y=loss_vec, title='', block=True)


step_to_eps = {
    0:0.8,
    1:0.7,
    2:0.6,
    3:0.5,
    4:0.4,
    5:0.3,
    6:0.2,
    7:0.1
}

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


mg = ModelGenerator(rules=config.incomplete_rules,
                cache_dir=config.incomplete_rules_cache_dir,
                clear_cache=False)
shared_data = mg.shared_data
shared_integration_supports = Integrator(dir_=config.incomplete_rules_integrator_dir,
                                            shared_data=shared_data,
                                            clear_cache=False).shared_integration_supports

def get_better_p(p_1, L_p_1, p_2, des_step, p_ref_index, shared_integration_supports, T, condition_of_break, eps_,sym_params_):

    # d_p1_p2 =
    # ro_vec =
    # eps_vec = np.concatenate([np.linspace(start=0.1, stop=0.9, num=N), np.linspace(start=0.01, stop=0.1, num=N),np.linspace(start=0.001, stop=0.01, num=N)])
    # eps_vec = np.concatenate([np.linspace(start=0.01, stop=0.1, num=N), np.linspace(start=0.001, stop=0.009, num=N), np.array([0.82])])
    N = 10
    # eps_vec = np.logspace(start=-N, stop=0 ,num=N+1, endpoint=True, base=10.0, dtype=None, axis=0)[:-1]
    # print(eps_vec)
    # eps_vec = [0.1,0.4,0.8]
    eps_vec = [0.8]
    # eps_vec = [0.82]
    # if des_step > 7:
    #     # eps_vec = [0.001, 0.1, 0.2, 0.4,0.8]
    #     eps_vec = [eps_]
    # else:
    #     eps_vec= [step_to_eps[des_step]]

    loss_vec = []
    p_vec = []
    # eps_vec = [eps_, eps_/10.0, min(eps_*10, 0.82)]
    for i in range(len(eps_vec)):
        eps_i = eps_vec[i]
        # print('{}/{}'.format(i+1, len(eps_vec)))
        # a, h, f, coeff_list = shared_data['ahfcoeff_list']
        # print('dist {}'.format(ro_Distrib4D(p_1,p_2,a_shared=[el.cpu().detach().numpy() for el in a])))
        p_new = weighted_amount(list_of_distributions=[p_1, p_2],
                                alpha_list=np.asarray([1 - eps_i, eps_i]))
        
        # p_func = Integrator.make_policy_function(p_xi_eta_gamma=p_new,
        #                                     new_omega_list=shared_integration_supports['new_omega_list'],
        #                                     list_for_integrate=shared_integration_supports['list_for_integrate'],
        #                                     projection_to_x_y_info=shared_integration_supports[
        #                                         'projection_to_x_y_info'],
        #                                     shared_Rects=shared_integration_supports['Rects'],
        #                                     shared_Grids=shared_integration_supports['Grids'],
        #                                     print_time_of_this_func=False
        #                                     )

        # axs_ = plot_policy_function(mode_of_plot='map',
        #                  filepath_to_save_response_surface='',
        #                  p_func=p_func,
        #                  Grids=shared_integration_supports['Grids'],
        #                  block_canvas=False
        #                  )
        
        # plt.show()
        L_of_p_new = get_L(p_new, shared_integration_supports, T, condition_of_break,sym_params_)

        # if i==0 and (L_of_p_new > L_p_1):
        #     return p_1, L_p_1
        loss_vec.append(L_of_p_new)
        p_vec.append(p_new)
    # print(loss_vec)

    # plt.plot(loss_vec)
    # plt.show()

    # plot_vec(x=eps_vec, y=loss_vec, title='', block=True)

    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(eps_vec, loss_vec)
    # ax.set_title('step {} p ref index {} current best {}'.format(des_step, p_ref_index, L_p_1))
    # ax.set_xlabel(r'$\varepsilon$')
    # ax.set_ylabel(r'$\mathcal{L}$')
    # ax.set_xscale('log')
    #
    # tmpfile = BytesIO()
    # fig.savefig(tmpfile, format='png')
    # encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    #
    # html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    #
    # with open(config.descent_html, 'a') as f:
    #     f.write(html)
    # plt.close(fig)
    best_pos = np.argmin(loss_vec)
    return p_vec[best_pos], loss_vec[best_pos]

def plot_and_save_polisy(dir,filename, 
                          distr,condition_of_break,sym_params_,shared_integration_supports):
    if not os.path.exists(dir):
        os.makedirs(dir)
    p_func = Integrator.make_policy_function(p_xi_eta_gamma=distr,
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
    
    loss_vec = GetLossForPlot(results_,T=config.phys_sim_params['t_end'])

    fig_ = plot_policy_function_with_trajectories(mode_of_plot='map',
                    filepath_to_save_response_surface='',
                    p_func=p_func,
                    Grids=shared_integration_supports['Grids'],
                    all_theta=apply_transofrm_of_units(results_['all_th'],from_th_in_si_to_th_in_volt),
                    all_omega=apply_transofrm_of_units(results_['all_omega'],from_omega_in_si_to_omega_in_volt),
                    loss_vec=loss_vec
                    )
    tmpfile = BytesIO()
    fig_.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    
    with open(os.path.join(dir,filename), 'a') as f:
        f.write(html)
    plt.close(fig_)


def make_batches(vector_of_numbers: np.array, num_of_batches: int) -> Tuple[np.array, np.array]:
    N = len(vector_of_numbers)
    batch_size = N // num_of_batches
    batches = np.zeros(shape=(num_of_batches, batch_size), dtype=np.intc)
    for i in range(num_of_batches):
        batches[i] = vector_of_numbers[i * batch_size:(i + 1) * batch_size]
    rest = vector_of_numbers[num_of_batches * batch_size:]
    return batches, rest


def try_step(current_best_object, current_best_value,
             batch_of_indexes, shared_objects,
             des_step,
             conn, index_of_process,
             shared_integration_supports, T, condition_of_break,
             eps_,sym_params_):
    k_ = 1
    output_best = current_best_object
    output_best_value = current_best_value
    for index_of_ref_object in batch_of_indexes:
        # if k_>10:
            # break
        # if(index_of_process==0):
        #     print('process {} current object {}'.format(index_of_process, k_))
        ref_obj = shared_objects[index_of_ref_object]
        p_new, L_of_p_new = get_better_p(p_1=current_best_object, L_p_1=current_best_value,
                                         p_2=ref_obj, des_step=des_step, p_ref_index=index_of_ref_object,
                                         shared_integration_supports=shared_integration_supports, T=T,
                                         condition_of_break=condition_of_break,
                                         eps_=eps_,sym_params_=sym_params_)
        # print(L_of_p_new)
        if L_of_p_new < output_best_value:
            print('new best found. value {}'.format(L_of_p_new))
            output_best = p_new
            output_best_value = L_of_p_new
            # break
        k_ += 1
    conn.send([output_best, output_best_value])
    conn.close()


def insert_noise_to_p(p: my_models.Distrib4D, mask_of_noisy_rules: np.array,
                      shared_data) -> None:
    if len(mask_of_noisy_rules) == 0:
        return 
    n_r = len(p.z_list)
    a, h, f, coeff_list = shared_data['ahfcoeff_list']

    rand_seed = ModelGenerator.make_random_random_seed_for_start_z_point(
        num_of_rules=n_r,
        a_i_j=shared_data['a_i_j'],
        b_i_j=shared_data['b_i_j'],
        is_approx=False
    )
    init_z = ModelGenerator.make_init_z(
        num_of_rules=n_r,
        distrib_of_rules=shared_data['distrib_of_rules'],
        a_i_j=shared_data['a_i_j'],
        b_i_j=shared_data['b_i_j'],
        random_seed=rand_seed,
        x_i_vec=shared_data['x_i_vec'],
        y_i_vec=shared_data['y_i_vec'],
        z_i_vec=shared_data['z_i_vec'],
        razb_tensor_a=a)

    for idx in mask_of_noisy_rules:
        # z_i = p.z_list[idx]
        # shape = z_i.shape
        # a , _ = shared_data['ahfcoeff_list']

        # not_normed_z = compute_gauss(x_from_torch, y_from_torch, z_from_torch,
        #               mu_x[i],
        #               sigma_x[i],
        #               mu_y[i],
        #               sigma_y[i],
        #               mu_z[i],
        #               sigma_z[i])
        # norm = torch.sum(not_normed_z * razb_tensor_a[i])
        # to_insert = not_normed_z / norm

        # to_insert = np.random.uniform(low=0.0, high=1.0, size=shape)
        # to_insert = to_insert/np.sum(to_insert)
        p.z_list[idx] = init_z[idx].cpu().detach().numpy()  # insert noise

def from_indexes_of_rules_to_indexes_of_list(indxs:list,rules:dict):
    from_to_ = {}
    i_ = 0
    for k_ in rules.keys():
        from_to_.update({k_:i_})
        i_ +=1
    return [from_to_[el] for el in indxs]

def do_they_intersect(rect1, rect2):
    x_a = rect1[0][0]
    x_b = rect1[0][1]
    y_a = rect1[1][0]
    y_b = rect1[1][1]
    x_a_tilde = rect2[0][0]
    x_b_tilde = rect2[0][1]
    y_a_tilde = rect2[1][0]
    y_b_tilde = rect2[1][1]
    return ((x_a <= x_a_tilde < x_b) or (x_a < x_b_tilde <=x_b)) and ((y_a <= y_a_tilde < y_b) or (y_a < y_b_tilde <=y_b))

if __name__ == '__main__':

    NUMBER_OF_PROCESSORS = 8
    
    EPOCH = 1
    gradien_steps = 1
    per_processor_crossovers = 20
    number_of_cyclic_passes = 2
    
    from_th_in_volt_to_th_in_si = config.translators_units_of_measurement['from_th_in_volt_to_th_in_si']
    from_omega_in_volt_to_omega_in_si = config.translators_units_of_measurement['from_omega_in_volt_to_omega_in_si']



    list_of_rects = shared_data['list_of_rect']
    fictitious_rules_indxs = from_indexes_of_rules_to_indexes_of_list(
        [50,51,52,53,54,55,56,57,58,59,60,61],config.incomplete_rules_dict
    )
    rects_of_fictitious_rules = {ind:list_of_rects[ind] for ind in fictitious_rules_indxs}
    areas_of_interests_sym_params = []
    conditions_of_breaks = []
    masks_of_rules = []
    IOU_treshhold = 0.3
    for ind_of_rule in rects_of_fictitious_rules.keys():
        rect_of_rule = rects_of_fictitious_rules[ind_of_rule]
        # какие прямоугольники наложились на текущий?
        list_of_intersected_rules = [ind_of_rule]
        iou_vec = [1.0]
        for another_ind in rects_of_fictitious_rules.keys():
            if ind_of_rule == another_ind:
                continue
            rect2_ = rects_of_fictitious_rules[another_ind]
            if do_they_intersect(rect_of_rule,rects_of_fictitious_rules[another_ind]):
                # x1 y1 x2 y2
                to_t_1 = np.expand_dims(np.array([rect_of_rule[0][0],rect_of_rule[1][0],rect_of_rule[0][1],rect_of_rule[1][1]]),axis=0)
                to_t_2 = np.expand_dims(np.array([rect2_[0][0],rect2_[1][0],rect2_[0][1],rect2_[1][1]]),axis=0)
                iou_ = box_iou(boxes1=torch.tensor(to_t_1),
                        boxes2=torch.tensor(to_t_2))
                iou_vec.append(iou_.cpu().detach().numpy()[0][0])
                list_of_intersected_rules.append(another_ind)
                # if iou_ > IOU_treshhold:
                #     list_of_intersected_rules.append(another_ind)
        argsort_ = np.flip(np.argsort(iou_vec))
        sorted_by_score_rects_indexes = []
        for i in range(len(argsort_)):
            if iou_vec[i] > 0.3:
                sorted_by_score_rects_indexes.append(list_of_intersected_rules[argsort_[i]])

        delta_x =rect_of_rule[0][1]- rect_of_rule[0][0]
        delta_y = rect_of_rule[1][1]- rect_of_rule[1][0]
        # must be in SI units of mesuarments
        conditions_of_breaks.append(
             np.asarray([
                [from_th_in_volt_to_th_in_si(rect_of_rule[0][0]), from_th_in_volt_to_th_in_si(rect_of_rule[0][1])],
                [from_omega_in_volt_to_omega_in_si(rect_of_rule[1][0]), from_omega_in_volt_to_omega_in_si(rect_of_rule[1][1])],
                [-9999.0, 9999.0],
                [-9999.0, 9999.0]
            ], dtype=np.float32)
        )
        areas_of_interests_sym_params.append(
            {'x_1_range': [rect_of_rule[0][0]+delta_x*0.1, rect_of_rule[0][1]-delta_x*0.1], 
                'x_2_range': [rect_of_rule[1][0]+delta_y*0.1, rect_of_rule[1][1]-delta_y*0.1],
            'n_x_for_sim': 5, 'n_y_for_sim': 5},
        )
        masks_of_rules.append(
            sorted_by_score_rects_indexes
        )
        # visualize intersected rules with IOU > IOU_treshold
        fig,axs = plt.subplots(1,1)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(sorted_by_score_rects_indexes)-1)
        m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        i_=0
        rect_ = rects_of_fictitious_rules[ind_of_rule]
        plot_rect(axs,x_1=rect_[0][0],x_2=rect_[0][1],y_1=rect_[1][0],y_2=rect_[1][1],color=(1,0,0,1))
        for ind_of_r in sorted_by_score_rects_indexes:
            rect_ = rects_of_fictitious_rules[ind_of_r]
            plot_empty_rect(axs,x_1=rect_[0][0],x_2=rect_[0][1],y_1=rect_[1][0],y_2=rect_[1][1],color=m.to_rgba(i_))
            i_ +=1 
        axs.set_title(str(sorted_by_score_rects_indexes))
        plt.show()

    # areas_of_interests_sym_params = [
    #     # imagine like this 
    #     # {'x_1_range': [-1/3, 1/3], 'x_2_range': [-1,  0],   'n_x_for_sim': 20, 'n_y_for_sim': 20},
    #     # {'x_1_range': [-1/3, 1/3], 'x_2_range': [-2/3, 2/3],  'n_x_for_sim': 20, 'n_y_for_sim': 20}
    #     # {'x_1_range': [-1/3, 1/3], 'x_2_range': [0,  1],  'n_x_for_sim': 20, 'n_y_for_sim': 20},
    #     # but make like this  X_{i} \in X_{i+1} 
    #     {'x_1_range': [-0.9, 0.9], 'x_2_range': [-0.9,0.9],   'n_x_for_sim': 10, 'n_y_for_sim': 10},
    #     {'x_1_range': [-0.9, 0.9], 'x_2_range': [-0.9,0.9],   'n_x_for_sim': 10, 'n_y_for_sim': 10},
    #     {'x_1_range': [-0.9, 0.9], 'x_2_range': [-0.9,0.9],   'n_x_for_sim': 10, 'n_y_for_sim': 10},
    # ]
    # masks_of_rules = [


    #     from_indexes_of_rules_to_indexes_of_list([58,59,60,61], config.incomplete_rules_dict),

    #     from_indexes_of_rules_to_indexes_of_list([54,55,56,57], config.incomplete_rules_dict),

    #     from_indexes_of_rules_to_indexes_of_list([50,51,52,53], config.incomplete_rules_dict)

    #     # #                                      th ZE w NS ZE PS
    #     # from_indexes_of_rules_to_indexes_of_list([52,53,54], config.incomplete_rules_dict),
    #     # #                                      th NS w NS ZE PS
    #     # from_indexes_of_rules_to_indexes_of_list([59,60,61], config.incomplete_rules_dict),
    #     # #                                      th PS w NS ZE PS
    #     # from_indexes_of_rules_to_indexes_of_list([66,67,68], config.incomplete_rules_dict),
    #     # #                                      th NS,ZE,PS w PM
    #     # from_indexes_of_rules_to_indexes_of_list([55,62,69], config.incomplete_rules_dict),
    #     # #                                      th NS,ZE,PS w NM
    #     # from_indexes_of_rules_to_indexes_of_list([51,58,65], config.incomplete_rules_dict),
    #     # #                                      th NS,ZE,PS w NB
    #     # from_indexes_of_rules_to_indexes_of_list([50,57,64], config.incomplete_rules_dict),
    #     # #                                      th NS,ZE,PS w PB
    #     # from_indexes_of_rules_to_indexes_of_list([56,63,70], config.incomplete_rules_dict)
    # ]

    

    if os.path.exists(config.descent_html):
        os.remove(config.descent_html)
    if os.path.exists('./descentplot') and (os.path.exists(os.path.join('./descentplot','steps.html'))):
        os.remove(os.path.join('./descentplot','steps.html'))

    # add grid/grids for empty rules - done
    # learn existing rules - done
    # add empty rules with noise



    # condition_of_break = np.asarray([
    #     config.theta_range,
    #     config.omega_range,
    #     [-9999.0, 9999.0],
    #     [-9999.0, 9999.0]
    # ], dtype=np.float32)

    # # learn existing rules
    # z_list_0 = mg.CreateModelFromScratch()
    # p_0 = DistrMaker.make_distrib4D(x_={
    #     'z_list': z_list_0,
    #     'np_omega': shared_data['np_omega']
    # })
    # insert_noise_to_p(p_0, mask_of_noisy_rules=indexes_of_fictitious_rules)
    # # save base distr
    # torch.save(p_0, config.inc_p0)
    # raise  SystemExit

    p0 = torch.load(config.inc_p0)
    p0.cast_to_float32()

    timer = time_mesuament.Timer()
    timer.start()
    T = config.phys_sim_params['t_end']
    p_best = p0
    loss_vec = []

    number_of_crosses = NUMBER_OF_PROCESSORS * per_processor_crossovers
    # eps = calc_eps(adj_matrix)
    search_eps = 0.82
    nums_of_starts = []
    num_of_early_stops = 0
    
    already_oprimized_areas = []
    for epoch in range(EPOCH):
        print('\t\t\t\t\t\t\t\tepoch {}/{}'.format(epoch,EPOCH-1))
        for area_index in range(len(masks_of_rules)):
            condition_of_break = conditions_of_breaks[area_index]
            # в новой области потерю нужно пересчитать, т.к. решается уже совсем другая оптимизационная задача
            sym_params_ = areas_of_interests_sym_params[area_index]
            L_best = get_L(p_best, shared_integration_supports, T, condition_of_break,sym_params_)
            loss_vec.append(L_best)
            print('\t\t\t\t\t\tarea {}/{}. current best = {}'.format(area_index,len(masks_of_rules)-1,L_best))
            sorted_by_iou_indexes_of_rules = np.setdiff1d(masks_of_rules[area_index],already_oprimized_areas)
            for pass_index in range(number_of_cyclic_passes):
                print('\t\t\t\tcyclic pass {}/{}'.format(pass_index,number_of_cyclic_passes-1))
                # rules_per_step = min(4,len(sorted_by_iou_indexes_of_rules))
                rules_per_step = len(sorted_by_iou_indexes_of_rules)
                # number_of_steps_in_one_pass = len(sorted_by_iou_indexes_of_rules)-rules_per_step+1
                number_of_steps_in_one_pass = 1
                for step_index in range(number_of_steps_in_one_pass):
                    timer.start_time_period()
                    indexes_of_fictitious_rules = sorted_by_iou_indexes_of_rules[step_index:step_index+rules_per_step]
                    print('step in pass {}/{}'.format(step_index,number_of_steps_in_one_pass-1))
                    for gr_step_indx in range(gradien_steps):
                        # print('\t\tgrstep {}/{}'.format(gr_step_indx,gradien_steps-1))
                        eps_ = 0.1
                        p_copy = [copy.deepcopy(p_best) for i in range(number_of_crosses)]
                        for i in range(len(p_copy)):
                            insert_noise_to_p(p_copy[i], indexes_of_fictitious_rules,
                                            shared_data)
                        all_distribs = p_copy
                        all_indexes = [i for i in range(len(p_copy))]
                        # make batches for each process
                        batches, rest = make_batches(vector_of_numbers=all_indexes,
                                                    num_of_batches=NUMBER_OF_PROCESSORS)
                        # try_step(p_best,L_best,batches[0],all_distribs,gr_step_indx,None,0,shared_integration_supports, T,condition_of_break)
                        # raise SystemExit
                        processes = []
                        parent_coons = []
                        output_of_processes = []

                        for i in range(NUMBER_OF_PROCESSORS):
                            # try_step(p_best,L_best,batches[i],all_distribs,gr_step_indx,shared_output,i)
                            parent_conn, child_conn = mp.Pipe()
                            parent_coons.append(parent_conn)
                            p = mp.Process(target=try_step, args=(
                            p_best, L_best, batches[i], all_distribs, gr_step_indx, child_conn, i, shared_integration_supports, T,
                            condition_of_break, eps_,sym_params_))
                            processes.append(p)
                            p.start()
                        if len(rest) > 0:
                            parent_conn, child_conn = mp.Pipe()
                            parent_coons.append(parent_conn)
                            rest_p = mp.Process(target=try_step, args=(
                            p_best, L_best, rest, all_distribs, gr_step_indx, child_conn, NUMBER_OF_PROCESSORS,
                            shared_integration_supports, T, condition_of_break, eps_,sym_params_))
                            processes.append(rest_p)
                            rest_p.start()
                        for i in range(NUMBER_OF_PROCESSORS):
                            output_of_process = parent_coons[i].recv()
                            output_of_processes.append(output_of_process)
                            processes[i].join()
                        # pprint.pprint(output_of_processes)
                        all_processes_values = []
                        for i in range(len(output_of_processes)):
                            all_processes_values.append(output_of_processes[i][1])
                        argmin_ = np.argmin(all_processes_values)
                        p_best, L_best = output_of_processes[argmin_]
                        loss_vec.append(L_best)
                    torch.save(p_best, config.inc_random_dec_file_path)
                    # print and save current best surf
                    plot_and_save_polisy(dir='./descentplot',filename='steps.html',distr=p_best,
                                        condition_of_break=condition_of_break,sym_params_=sym_params_,shared_integration_supports=shared_integration_supports)


                    time_per_iter = timer.stop_time_period()
                    print('\ttime per step in pass {}'.format(time_per_iter))
            already_oprimized_areas.append(sorted_by_iou_indexes_of_rules[0])

    timer.stop()
    print('{} sek'.format(timer.get_execution_time()))
    print('start min {}'.format(loss_vec[0]))
    print('best min {}'.format(min(loss_vec)))
    plt.plot(loss_vec)
    plt.show()
    # plot_vec(x=np.arange(len(loss_vec)), y=loss_vec, title=r'$\mathcal{L}$', block=False)
    # plt.show()
