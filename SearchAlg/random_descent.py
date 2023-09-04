import pprint

import torch
import os
import config
from Alg.solving_algorithm import ModelGenerator
from FileSystem.storage import SimResultsManager, PStorage
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function
from general import get_sim_results
from general_purpose_functions import time_mesuament
from SearchAlg.genetic_alg_general_functions import Loss, plot_hist_of_list, GetPhysicalLoss, get_L
import numpy as np
from CustomModels.my_models import weighted_amount, Integrator
from general import plot_vec

import matplotlib.pyplot as plt
import base64
from io import BytesIO
import multiprocessing as mp
from typing import Tuple,List


def calc_eps(adj_matrix):
    all_distances = []
    N = len(adj_matrix)
    for i in range(N-1):
        for j in range(i+1,N):
            all_distances.append(adj_matrix[i][j])
    eps = np.percentile(all_distances, q=1)
    return eps


def plot_loss_along_the_line(p_1,p_2, T,condition_of_break):
    N = 20
    eps_vec = np.linspace(start=0.0, stop=1.0,num=N)
    loss_vec  = []
    # best_p, best_v = get_better_p(p_1, p_2, T)
    # print(best_v)
    for i in range(len(eps_vec)):
        print('{}/{}'.format(i+1, len(eps_vec)))
        eps_ = eps_vec[i]
        p_new = weighted_amount(list_of_distributions=[p_1, p_2],
                                alpha_list=np.asarray([1 - eps_, eps_]))
        L_of_p_new = get_L(p_new, shared_integration_supports, T,condition_of_break)
        loss_vec.append(L_of_p_new)
    print(loss_vec)
    plot_vec(x=eps_vec,y=loss_vec,title='',block=True)


def get_better_p(p_1, L_p_1, p_2, des_step, p_ref_index, shared_integration_supports, T,condition_of_break):
    N = 3
    # d_p1_p2 =
    # ro_vec =
    # eps_vec = np.concatenate([np.linspace(start=0.1, stop=0.9, num=N), np.linspace(start=0.01, stop=0.1, num=N),np.linspace(start=0.001, stop=0.01, num=N)])
    # eps_vec = np.concatenate([np.linspace(start=0.01, stop=0.1, num=N), np.linspace(start=0.001, stop=0.009, num=N), np.array([0.82])])
    eps_vec= np.array([0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.8])
    loss_vec = []
    p_vec = []
    for i in range(len(eps_vec)):
        # print('{}/{}'.format(i+1, len(eps_vec)))
        eps_ = eps_vec[i]
        p_new = weighted_amount(list_of_distributions=[p_1, p_2],
                                alpha_list=np.asarray([1 - eps_, eps_]))
        L_of_p_new = get_L(p_new, shared_integration_supports, T,condition_of_break)
        # if i==0 and (L_of_p_new > L_p_1):
        #     return p_1, L_p_1
        loss_vec.append(L_of_p_new)
        p_vec.append(p_new)
    # plot_vec(x=eps_vec, y=loss_vec, title='', block=True)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(eps_vec, loss_vec)
    ax.set_title('step {} p ref index {} current best {}'.format(des_step, p_ref_index,L_p_1))
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$\mathcal{L}$')

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    with open(config.descent_html, 'a') as f:
        f.write(html)
    plt.close(fig)
    best_pos = np.argmin(loss_vec)
    return p_vec[best_pos], loss_vec[best_pos]


def make_batches(vector_of_numbers: np.array, num_of_batches: int) -> Tuple[np.array, np.array]:
    N = len(vector_of_numbers)
    batch_size = N // num_of_batches
    batches = np.zeros(shape=(num_of_batches, batch_size), dtype=np.intc)
    for i in range(num_of_batches):
        batches[i] = vector_of_numbers[i*batch_size:(i+1)*batch_size]
    rest = vector_of_numbers[num_of_batches*batch_size:]
    return batches, rest

def try_step(current_best_object, current_best_value,
             batch_of_indexes, shared_objects,
             des_step,
             conn, index_of_process,
             shared_integration_supports, T,condition_of_break):
    k_ = 1
    output_best = current_best_object
    output_best_value = current_best_value
    for index_of_ref_object in batch_of_indexes:
        if k_>50:
            break
        print('process {} ref object {}'.format(index_of_process, k_))
        ref_obj = shared_objects[index_of_ref_object]
        p_new, L_of_p_new = get_better_p(p_1=current_best_object, L_p_1=current_best_value,
                                         p_2=ref_obj, des_step=des_step, p_ref_index=index_of_ref_object,
                                         shared_integration_supports=shared_integration_supports, T=T,condition_of_break=condition_of_break)
        if L_of_p_new < output_best_value:
            print('new best found in process {}'.format(index_of_process))
            output_best = p_new
            output_best_value = L_of_p_new
        k_+=1
    conn.send([output_best, output_best_value])
    conn.close()




if __name__ == '__main__':

    NUMBER_OF_PROCESSORS = 16

    indexes_to_modify = np.concatenate([np.arange(start = 13, stop=35, dtype=np.uint32),
                                       np.array([2,9,37,44,3,10,38,45,4,11,39,46],dtype=np.uint32)])

    if os.path.exists(config.descent_html):
        os.remove(config.descent_html)

    shared_data = ModelGenerator(rules=config.rules,
                                 cache_dir=config.models_gen['pointwise_approx']['dir'],
                                 clear_cache=False).shared_data
    shared_integration_supports = Integrator(dir_=config.integrator_dir,
                                             shared_data=shared_data,
                                             clear_cache=False).shared_integration_supports

    condition_of_break = np.asarray([
        config.theta_range,
        config.omega_range,
        [-9999.0, 9999.0],
        [-9999.0, 9999.0]
    ], dtype=np.float32)


    timer = time_mesuament.Timer()
    timer.start()
    # загрузить все раcпределения
    all_p = {}
    for i in range(16):
        index_of_exe = str(i)
        pstorage_i = PStorage(dir_=config.p_storage_base_dir,
                              index_of_exe=str(index_of_exe),
                              clear_cache=False)
        n_of_p = pstorage_i.get_number_of_elements()

        for j in range(n_of_p):
            element_path = pstorage_i.get_path_by_index(j)
            p_xi_eta_gamma = pstorage_i.load_by_index(j)
            p_xi_eta_gamma.modify_z_in_rule(indexes_to_modify)
            p_xi_eta_gamma.cast_to_float32()
            all_p.update({element_path: p_xi_eta_gamma})
    # загрузить значения L на этих распределениях
    results_ = {}
    dirs = os.listdir(config.p_sim_results_base_dir)
    for i in range(len(dirs)):
        if not dirs[i].isnumeric():
            # print(dirs[i])
            continue
        sim_storage = SimResultsManager(base_dir_=config.p_sim_results_base_dir, index_of_exe=dirs[i],
                                        clear_cache=False)
        results_i = sim_storage.get_sim_results()
        for j in range(len(results_i)):
            result = results_i[j]
            key = result['source_path']
            results_.update({key: result['results']})
    all_names = []
    all_results = []
    all_values = []
    T = config.phys_sim_params['t_end']
    for key in results_:
        # key is source path
        all_names.append(key)
        all_results.append(results_[key])
        all_values.append(GetPhysicalLoss(sim_results_=results_[key], T=T))
    argmin = np.argmin(all_values)
    print(all_values[argmin])
    # plot_hist_of_list(all_values)
    # проблема с различными результатами для одних и тех же распредлений

    # выбрать лучшее, начать случайный спуск
    best_name = all_names[argmin]
    L_best = all_values[argmin]
    p_best = all_p[best_name]
    # p_best = torch.load('C:/Users/User/Desktop/lab2023\p_storage\\13\\distrib_75.txt')
    print('auto min L {}'.format(L_best))
    # p_best = torch.load('C:/Users/User/Desktop/lab2023\p_storage\\3\\distrib_184.txt')
    L_best = get_L(p_best, shared_integration_supports, T, condition_of_break)
    print(best_name)
    print('current best = {}'.format(L_best))
    loss_vec = [L_best]

    all_distribs = list(all_p.values())
    gradien_steps = 100
    # eps = calc_eps(adj_matrix)
    search_eps = 0.82
    nums_of_starts = []
    num_of_early_stops = 0
    for gr_step_indx in range(gradien_steps):
        timer.start_time_period()
        all_indexes = np.arange(start=0, stop=len(all_distribs))
        np.random.shuffle(all_indexes)
        num_of_starts = len(all_distribs)
        has_a_full_pass_been_made = False
        # make batches for each process
        batches, rest = make_batches(vector_of_numbers=all_indexes,
                                     num_of_batches=NUMBER_OF_PROCESSORS)
        processes = []
        parent_coons = []
        output_of_processes = []

        for i in range(NUMBER_OF_PROCESSORS):
            # try_step(p_best,L_best,batches[i],all_distribs,gr_step_indx,shared_output,i)
            parent_conn, child_conn = mp.Pipe()
            parent_coons.append(parent_conn)
            p = mp.Process(target=try_step, args=(p_best,L_best,batches[i],all_distribs,gr_step_indx,child_conn,i,shared_integration_supports, T,condition_of_break))
            processes.append(p)
            p.start()
        if len(rest)>0:
            parent_conn, child_conn = mp.Pipe()
            parent_coons.append(parent_conn)
            rest_p = mp.Process(target=try_step, args=(p_best,L_best,rest,all_distribs,gr_step_indx,child_conn,NUMBER_OF_PROCESSORS,shared_integration_supports, T,condition_of_break))
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
        print('current L_best {}'.format(L_best))
        torch.save(p_best, config.random_decent_best_filepath)

        time_per_iter = timer.stop_time_period()
        rem_time = (gradien_steps-gr_step_indx-1)*time_per_iter

        print('step {}/{} elapsed time {} s remaining time {} m'.format(gr_step_indx,
                                                                        gradien_steps,
                                                                        str(time_per_iter)[:7],
                                                                        str(rem_time/60.0)[:5])
              )



    timer.stop()
    print('{} sek'.format(timer.get_execution_time()))

    print('start min {}'.format(loss_vec[0]))
    print('best min {}'.format(min(loss_vec)))
    # plot_vec(x=np.arange(len(loss_vec)), y=loss_vec, title=r'$\mathcal{L}$', block=False)
    # plt.show()


