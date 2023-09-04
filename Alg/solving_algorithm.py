import os.path
import typing
from typing import Tuple
from typing import List
from typing import Any
from typing import Dict
from typing import Type
from copy import deepcopy

import torch
import numpy as np
from matplotlib import pyplot as plt

import config
from general_purpose_functions.using_the_function import f_value_in_center_of_segment
from RulesMaker.rules_class import Rules
from FileSystem.general_purpose_functions import delete_all_data_from_directory, remove_dir, mkdir_if_not_exists
from SQLHelpers.query_helpers import SqlString
import sqlite3
from RulesMaker.func_det import Distrib
from general import compute_gauss, compute_seq_vec
from CustomModels.my_models import ModelTrainer
from DataMaker.grid_maker import GridMaker


# def save_model(self, model):
#     torch.save(model,'')


# class SequenceOfMappings:
#     seq_ = None
#
#     def __init__(self, seq: List[Tuple[str, ModelWrapper]]):
#         '''
#
#         :param seq:
#         [
#         (model_name,ModelWrapper)_1,
#         (model_name,ModelWrapper)_2,
#         ...,
#         (model_name,ModelWrapper)_n
#         ]
#         '''
#         if not self.__check_sizes_of_inputs_and_outputs(seq):
#             print('отображения в последовательности несовместимы')
#             return
#
#         self.seq_ = seq
#
#     def __call__(self, x: Any):
#         current_x = x
#         for el in self.seq_:
#             mapping = el[1]
#             current_x = mapping(current_x)
#
#         return current_x
#
#     def __check_sizes_of_inputs_and_outputs(self, seq: List[Tuple[str, ModelWrapper]]):
#         for i in range(len(seq) - 1):
#             if seq[i][1].get_shape_of_output() != seq[i + 1][1].get_shape_of_input():
#                 print('отображения в последовательности несовместимы')
#                 return False
#         return True


class Alg:
    seq_ = None

    def __init__(self, models_config: Dict[str, str]):
        # models_config
        '''
        :param models_config:
        {
            'key1':...
            ...
            'cache_dir':...
            ...
        }
        '''

        # [(name_of_alg, alg),(name_of_alg, alg),...]

        # load resnet

        # load svd

        # load catboost

        pass

    def __call__(self,
                 paths_to_images: list[str],
                 list_to_save_answer: list[int]):
        # load  batch of images,images to tensor for resnet
        # eval resnet
        # eval svd
        # eval catboost
        # covert catboost answer to source label
        # write alg answer to list
        pass


class SharedTrainPartOfModel:

    # using data source- rules and t_omega
    @staticmethod
    def compute_a_h_f_coeff_list(rules: Rules, t_omega):
        f = [[] for i in range(rules.n_r)]
        k1 = 0
        for rule_key in rules.get_ordered_keys():
            IF_distrs = rules.rules_dict[rule_key]['IF']
            THEN_distrs = rules.rules_dict[rule_key]['THEN']
            k2 = 0
            for distr in IF_distrs:
                vec_of_f_k = torch.zeros(size=(distr.num_of_segments,), requires_grad=False)
                for i in range(distr.num_of_segments):
                    vec_of_f_k[i] = f_value_in_center_of_segment(distr, t_omega[k1][k2][i])
                f[k1].append(vec_of_f_k)
                k2 += 1

            for distr in THEN_distrs:
                vec_of_f_k = torch.zeros(size=(distr.num_of_segments,), requires_grad=False)
                for i in range(distr.num_of_segments):
                    vec_of_f_k[i] = f_value_in_center_of_segment(distr, t_omega[k1][k2][i])
                f[k1].append(vec_of_f_k)
                k2 += 1
            k1 += 1

        h = [[] for i in range(rules.n_r)]
        k1 = 0
        for rule_key in rules.get_ordered_keys():
            IF_distrs = rules.rules_dict[rule_key]['IF']
            THEN_distrs = rules.rules_dict[rule_key]['THEN']
            k2 = 0
            for distr in IF_distrs:
                h_vec = np.zeros(shape=(distr.num_of_segments,))
                for i in range(distr.num_of_segments):
                    h_vec[i] = distr.grid[i + 1] - distr.grid[i]
                h[k1].append(torch.tensor(h_vec, requires_grad=False))
                k2 += 1
            for distr in THEN_distrs:
                h_vec = np.zeros(shape=(distr.num_of_segments,))
                for i in range(distr.num_of_segments):
                    h_vec[i] = distr.grid[i + 1] - distr.grid[i]
                h[k1].append(torch.tensor(h_vec, requires_grad=False))
                k2 += 1
            k1 += 1

        a = []
        k1 = 0
        for rule_key in rules.get_ordered_keys():
            size_of_pth_tensor = []
            IF_distrs = rules.rules_dict[rule_key]['IF']
            THEN_distrs = rules.rules_dict[rule_key]['THEN']
            for distr in IF_distrs:
                size_of_pth_tensor.append(distr.num_of_segments)
            for distr in THEN_distrs:
                size_of_pth_tensor.append(distr.num_of_segments)

            a_p_np = np.zeros(shape=tuple(size_of_pth_tensor))

            for k_1 in range(size_of_pth_tensor[0]):
                for k_2 in range(size_of_pth_tensor[1]):
                    for k_3 in range(size_of_pth_tensor[2]):
                        a_p_np[k_1][k_2][k_3] = h[k1][0][k_1] * h[k1][1][k_2] * h[k1][2][k_3]
            a.append(torch.from_numpy(a_p_np))
            k1 += 1

        coeff_list = [[] for i in range(rules.n_r)]
        k1 = 0
        for rule_key in rules.get_ordered_keys():
            IF_distrs = rules.rules_dict[rule_key]['IF']
            THEN_distrs = rules.rules_dict[rule_key]['THEN']
            for distr in IF_distrs:
                coeff_list[k1].append((distr.support[1] - distr.support[0]) ** 2)
            for distr in THEN_distrs:
                coeff_list[k1].append((distr.support[1] - distr.support[0]) ** 2)
            k1 += 1

        return a, h, f, coeff_list

    @staticmethod
    def make_list_of_rect_from_rules(rules: Rules):
        # list_of_rect = [
        #     [[ax1_1, bx1_1], [ax1_2, bx1_2]],
        #     [[ax2_1, bx2_1], [ax2_2, bx2_2]],
        #     [[ax3_1, bx3_1], [ax3_2, bx3_2]],
        #     [[ax4_1, bx4_1], [ax4_2, bx4_2]],
        # ]
        list_of_rect = [[] for i in range(rules.n_r)]
        k = 0
        for rule_key in rules.get_ordered_keys():
            IF_distrs = rules.rules_dict[rule_key]['IF']
            THEN_distrs = rules.rules_dict[rule_key]['THEN']
            for distr in IF_distrs:
                list_of_rect[k].append([distr.support[0], distr.support[1]])
            for distr in THEN_distrs:
                list_of_rect[k].append([distr.support[0], distr.support[1]])
            k += 1

        return list_of_rect


class ModelGenerator:
    cache_dir: str
    rules: Rules
    shared_data: Dict[str, Any]


    def make_shared_data(self, rules):
        a_i_j, b_i_j, x_i_vec, y_i_vec, z_i_vec = self.__make_omega_for_init_z(rules=rules,
                                                                               num_of_rules=rules.n_r,
                                                                               dimension=rules.dimension)
        t_omega, np_omega = GridMaker.make_omega_and_np_omega(rules)
        shared_data_ = {
            't_omega': t_omega,
            'np_omega': np_omega,
            'list_of_rect': SharedTrainPartOfModel.make_list_of_rect_from_rules(rules),
            'ahfcoeff_list': SharedTrainPartOfModel.compute_a_h_f_coeff_list(rules, t_omega),
            'a_i_j': a_i_j,
            'b_i_j': b_i_j,
            'x_i_vec': x_i_vec,
            'y_i_vec': y_i_vec,
            'z_i_vec': z_i_vec,
            'distrib_of_rules': np.ones(shape=(rules.n_r,)) / rules.n_r
        }
        return shared_data_

    def __init__(self, rules: Rules, cache_dir: str, clear_cache: bool = False):
        self.cache_dir = cache_dir
        self.rules = rules

        if clear_cache == True:
            delete_all_data_from_directory(self.cache_dir)

        if not os.path.exists(cache_dir):
            os.makedirs(self.cache_dir)
            shared_data_ = self.make_shared_data(rules)
            torch.save(shared_data_, os.path.join(cache_dir, 'shared_data.txt'))
            self.shared_data = shared_data_
        elif os.path.exists(cache_dir) and (not os.path.exists(os.path.join(cache_dir, 'shared_data.txt'))):
            shared_data_ = self.make_shared_data(rules)
            self.shared_data = shared_data_
            torch.save(shared_data_, os.path.join(cache_dir, 'shared_data.txt'))
        elif os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, 'shared_data.txt')):
            self.shared_data = torch.load(os.path.join(cache_dir, 'shared_data.txt'))

    def make_new_seed(self,is_approx):
        random_seed = self.make_random_random_seed_for_start_z_point(num_of_rules=self.rules.n_r,
                                                                     a_i_j=self.shared_data['a_i_j'],
                                                                     b_i_j=self.shared_data['b_i_j'],
                                                                     is_approx=is_approx)
        return random_seed

    def CreateModelFromScratch(self, is_approx,gr_steps,lr=0.01):
        random_seed = self.make_new_seed(is_approx)
        a, h, f, coeff_list = self.shared_data['ahfcoeff_list']
        t_omega = self.shared_data['t_omega']
        init_z = self.make_init_z(num_of_rules=self.rules.n_r,
                                  distrib_of_rules=self.shared_data['distrib_of_rules'],
                                  a_i_j=self.shared_data['a_i_j'],
                                  b_i_j=self.shared_data['b_i_j'],
                                  random_seed=random_seed,
                                  x_i_vec=self.shared_data['x_i_vec'],
                                  y_i_vec=self.shared_data['y_i_vec'],
                                  z_i_vec=self.shared_data['z_i_vec'],
                                  razb_tensor_a=a)
        model_trainer = ModelTrainer()
        z_list = model_trainer.train(
            num_of_rules=self.rules.n_r,
            dimension=self.rules.dimension,
            rules=self.rules,
            omega=t_omega,
            train_dict={"max_num_of_epochs": gr_steps, "lr": lr},
            a=a, h=h, f=f, coeff_list=coeff_list,
            init_z=init_z, distrib_of_rules=self.shared_data['distrib_of_rules'],
            check_target_values=False,
            min_cons=0.001,
            min_norm=0.001,
            min_distr=0.001,
            print_time_of_this_func=True,
            plot_gradien_loss=False,
            plot_consystency=False,
            print_num_of_restart_gradient=False,
            print_tmp_cons_and_loss=True
        )

        return z_list



    @staticmethod
    def __make_omega_for_init_z(rules: Rules, num_of_rules: int, dimension: int):
        a_i_j = np.zeros(shape=(num_of_rules, dimension))
        b_i_j = np.zeros(shape=(num_of_rules, dimension))

        x_i_vec = []
        y_i_vec = []
        z_i_vec = []
        k1 = 0
        for rule_key in rules.get_ordered_keys():
            size_of_pth_tensor = []
            IF_distrs = rules.rules_dict[rule_key]['IF']
            THEN_distrs = rules.rules_dict[rule_key]['THEN']
            k2 = 0
            for distr in IF_distrs:
                a_i_j[k1][k2] = distr.support[0]
                b_i_j[k1][k2] = distr.support[1]
                size_of_pth_tensor.append(distr.num_of_segments)
                k2 += 1
            for distr in THEN_distrs:
                a_i_j[k1][k2] = distr.support[0]
                b_i_j[k1][k2] = distr.support[1]
                size_of_pth_tensor.append(distr.num_of_segments)
                k2 += 1

            x_i_vec.append(np.zeros(size_of_pth_tensor[0], ))
            y_i_vec.append(np.zeros(size_of_pth_tensor[1], ))
            z_i_vec.append(np.zeros(size_of_pth_tensor[2], ))
            for i in range(size_of_pth_tensor[0]):
                x_i_vec[k1][i] = 0.5 * (IF_distrs[0].grid[i + 1] + IF_distrs[0].grid[i])
            for i in range(size_of_pth_tensor[1]):
                y_i_vec[k1][i] = 0.5 * (IF_distrs[1].grid[i + 1] + IF_distrs[1].grid[i])
            for i in range(size_of_pth_tensor[2]):
                z_i_vec[k1][i] = 0.5 * (THEN_distrs[0].grid[i + 1] + THEN_distrs[0].grid[i])
            k1 += 1

        return a_i_j, b_i_j, x_i_vec, y_i_vec, z_i_vec

    @staticmethod
    def make_random_random_seed_for_start_z_point(num_of_rules, a_i_j, b_i_j,is_approx):
        random_seed = {}
        N = 1
        c_i_j_k = []
        for i in range(num_of_rules):
            c_i_j_k.append(np.random.rand(2 * N + 1, 2 * N + 1, 2 * N + 1))
        mu_x = np.zeros(shape=(num_of_rules,), )
        mu_y = np.zeros(shape=(num_of_rules,), )
        mu_z = np.zeros(shape=(num_of_rules,), )
        sigma_x = np.zeros(shape=(num_of_rules,), )
        sigma_y = np.zeros(shape=(num_of_rules,), )
        sigma_z = np.zeros(shape=(num_of_rules,), )
        for i in range(num_of_rules):
            # tmp_a_i_j = a_i_j[i]
            # tmp_b_i_j = b_i_j[i]
            # mu_x[i] = np.random.uniform(low=a_i_j[i][0] - (b_i_j[i][0] - a_i_j[i][0]) * 0.25,
            #                             high=a_i_j[i][0] + (b_i_j[i][0] - a_i_j[i][0]) * 0.25)
            # mu_y[i] = np.random.uniform(low=a_i_j[i][1] - (b_i_j[i][1] - a_i_j[i][1]) * 0.25,
            #                             high=a_i_j[i][1] + (b_i_j[i][1] - a_i_j[i][1]) * 0.25)
            # mu_z[i] = np.random.uniform(low=a_i_j[i][2] - (b_i_j[i][2] - a_i_j[i][2]) * 0.25,
            #                             high=a_i_j[i][2] + (b_i_j[i][2] - a_i_j[i][2]) * 0.25)

            mu_x[i] = np.random.uniform(low=a_i_j[i][0],
                                        high=b_i_j[i][0])
            mu_y[i] = np.random.uniform(low=a_i_j[i][1],
                                        high=b_i_j[i][1])
            mu_z[i] = np.random.uniform(low=a_i_j[i][2],
                                        high=b_i_j[i][2])
            # 3\simga = b-a
            # 5\simga = b-a :localized gauss
            if is_approx:
                sigma_x[i] = np.random.uniform(low=(b_i_j[i][0] - a_i_j[i][0]) /5,
                                               high=(b_i_j[i][0] - a_i_j[i][0]) /4)
                sigma_y[i] = np.random.uniform(low=(b_i_j[i][1] - a_i_j[i][1]) /5,
                                               high=(b_i_j[i][1] - a_i_j[i][1]) /4)
                sigma_z[i] = np.random.uniform(low=(b_i_j[i][2] - a_i_j[i][2]) /5,
                                               high=(b_i_j[i][2] - a_i_j[i][2]) /4)
            else:
                # coin_ = np.random.randint(low=-1,high=1)
                # mu_z_ = 0.0
                # if coin_ == -1:
                #     mu_z_ = -1.0
                # elif coin_ == 1:
                #     mu_z_ = 1.0
                # elif coin_ ==0:
                #     mu_z_ = 0.0
                # mu_z[i] = mu_z_


                coin_ = np.random.randint(low=0,high=2)
                mu_z_ = 0.0
                if coin_ == 0:
                    mu_z_ = -1.0
                elif coin_ == 1:
                    mu_z_ = 1.0
                    # print('coin is positive')
                mu_z[i] = mu_z_



                # mid_x  = 0.5*(a_i_j[i][0]+b_i_j[i][0])
                # mid_y = 0.5*(a_i_j[i][1]+b_i_j[i][1])
                # if mid_x < 0.0:
                #     mu_z[i] = -1.0
                # else:
                #     mu_z[i] =1.0

                # mu_x[i] = 0.5*(a_i_j[i][0]+b_i_j[i][0])
                # mu_y[i] = 0.5*(a_i_j[i][1]+b_i_j[i][1])

                # mu_x_ = mu_x[i] 
                # mu_y_ = mu_y[i]
                # sigma_x_ = sigma_x[i]
                # sigma_y_ = sigma_y[i]
                # sigma_z_ = sigma_z[i] 

                sigma_x[i] = np.random.uniform(low=(b_i_j[i][0] - a_i_j[i][0]) /10,
                                               high=(b_i_j[i][0] - a_i_j[i][0]) /8)
                sigma_y[i] = np.random.uniform(low=(b_i_j[i][1] - a_i_j[i][1]) /10,
                                               high=(b_i_j[i][1] - a_i_j[i][1]) /8)
                sigma_z[i] = np.random.uniform(low=(b_i_j[i][2] - a_i_j[i][2]) /10,
                                               high=(b_i_j[i][2] - a_i_j[i][2]) /8)
        random_seed.update({"N": N})
        random_seed.update({"c_i_j_k": c_i_j_k})
        random_seed.update({"mu_x": mu_x})
        random_seed.update({"mu_y": mu_y})
        random_seed.update({"mu_z": mu_z})
        random_seed.update({"sigma_x": sigma_x})
        random_seed.update({"sigma_y": sigma_y})
        random_seed.update({"sigma_z": sigma_z})

        return random_seed

    @staticmethod
    def make_init_z(num_of_rules, distrib_of_rules, a_i_j, b_i_j, random_seed, x_i_vec, y_i_vec, z_i_vec,
                    razb_tensor_a):
        init_z = []  # не прологарифмированные значения высот:
        N = random_seed["N"]
        c_i_j_k = random_seed["c_i_j_k"]
        mu_x = random_seed["mu_x"]
        mu_y = random_seed["mu_y"]
        mu_z = random_seed["mu_z"]
        sigma_x = random_seed["sigma_x"]
        sigma_y = random_seed["sigma_y"]
        sigma_z = random_seed["sigma_z"]
        for i in range(num_of_rules):
            x_from_torch = torch.from_numpy(x_i_vec[i]).view(len(x_i_vec[i]), 1)
            y_from_torch = torch.from_numpy(y_i_vec[i]).view(len(y_i_vec[i]), 1)
            z_from_torch = torch.from_numpy(z_i_vec[i]).view(len(z_i_vec[i]), 1)
            # not_normed_z = compute_seq_vec(x_from_torch, y_from_torch, z_from_torch, c_i_j_k[i], N, a_i_j[i][0],
            #                                a_i_j[i][1],
            #                                a_i_j[i][2],
            #                                b_i_j[i][0], b_i_j[i][1],
            #                                b_i_j[i][2]) * compute_gauss(x_from_torch, y_from_torch, z_from_torch,
            #                                                             mu_x[i],
            #                                                             sigma_x[i],
            #                                                             mu_y[i],
            #                                                             sigma_y[i],
            #                                                             mu_z[i],
            #                                                             sigma_z[i])
            # coin_ = np.random.randint(low=-1,high=1)
            # mu_z_ = 0.0
            # if coin_ < 0:
            #     mu_z_ = -1.0
            # else:
            #     mu_z = 1.0
            # mu_x_ = mu_x[i] 
            # mu_y_ = mu_y[i]
            # sigma_x_ = sigma_x[i]
            # sigma_y_ = sigma_y[i]
            # sigma_z_ = sigma_z[i] 
            not_normed_z = compute_gauss(x_from_torch, y_from_torch, z_from_torch,
                                                                        mu_x[i],
                                                                        sigma_x[i],
                                                                        mu_y[i],
                                                                        sigma_y[i],
                                                                        mu_z[i],
                                                                        sigma_z[i])
            # if torch.sum(torch.isnan(not_normed_z))!=0:
            #     print('nan in init z')
            norm = torch.sum(not_normed_z * razb_tensor_a[i])
            normed_z = not_normed_z / norm
            # if torch.sum(torch.isnan(normed_z))!=0:
            #     print('norm is zero')
            #     raise SystemExit
            init_z.append(normed_z * distrib_of_rules[i])
        return init_z
