import copy
import os.path
import time
import typing
from typing import Tuple
from typing import List
from typing import Any
from typing import Dict
from typing import Type
from copy import deepcopy
from numba import jit
import numba
import numpy as np
from numba.experimental import jitclass

from Model.model_wrapper import ModelWrapper
from RulesMaker.func_det import Distrib
import torch
from general import *


class ModelTrainer:

    def __comp_loss(self, z, f, num_of_rules, a, h, coeff_list, init_z, distrib_of_rules,
                    index_of_iteration, index_of_apply_raznost, max_squares):
        z_list = []
        for i in range(num_of_rules):
            z_list.append(torch.exp(z[i]))

        z_a_list = []
        for i in range(num_of_rules):
            z_a_list.append(z_list[i] * a[i])

        D_i_vec = torch.zeros(num_of_rules, requires_grad=False)
        for i in range(num_of_rules):
            D_i_vec[i] = torch.sum(z_a_list[i])

        # норма совметсного распредления
        norm_loss = torch.square(1 - torch.sum(D_i_vec))
        distrib_of_rules_loss = torch.sum(torch.square(D_i_vec - distrib_of_rules))

        S_p_j_k = []
        consistency_loss = torch.zeros(1, requires_grad=False)

        for i in range(num_of_rules):
            S_p_j_k.append([])
            S_p_j_k[i].append((torch.sum(z_a_list[i], (1, 2)) / h[i][0]) / D_i_vec[i])
            S_p_j_k[i].append((torch.sum(z_a_list[i], (0, 2)) / h[i][1]) / D_i_vec[i])
            S_p_j_k[i].append((torch.sum(z_a_list[i], (0, 1)) / h[i][2]) / D_i_vec[i])

            # consistency_loss = consistency_loss + \
            #                    coeff_list[i][0] * torch.sum(
            #     torch.square((f[i][0] - (torch.sum(z_a_list[i], (1, 2)) / h[i][0]) / D_i_vec[i]))) + \
            #                    coeff_list[i][1] * torch.sum(
            #     torch.square((f[i][1] - (torch.sum(z_a_list[i], (0, 2)) / h[i][1]) / D_i_vec[i]))) + \
            #                    coeff_list[i][2] * torch.sum(
            #     torch.square((f[i][2] - (torch.sum(z_a_list[i], (0, 1)) / h[i][2]) / D_i_vec[i])))

        ro_vec = torch.zeros(num_of_rules, requires_grad=False)

        for i in range(num_of_rules):
            ro_vec[i] = torch.sum(a[i] * torch.square(init_z[i] - z_list[i]))
        ro = torch.sum(ro_vec)
        if index_of_iteration > index_of_apply_raznost:
            for i in range(num_of_rules):
                consistency_loss = consistency_loss + \
                                   torch.sum(torch.square((f[i][0] - S_p_j_k[i][0]) / max_squares[i][0])) + \
                                   torch.sum(torch.square((f[i][1] - S_p_j_k[i][1]) / max_squares[i][1])) + \
                                   torch.sum(torch.square((f[i][2] - S_p_j_k[i][2]) / max_squares[i][2]))
            loss = consistency_loss + norm_loss + distrib_of_rules_loss + ro
            return loss, consistency_loss, norm_loss, ro, ro_vec, distrib_of_rules_loss
        else:
            for i in range(num_of_rules):
                consistency_loss = consistency_loss + \
                                   coeff_list[i][0] * torch.sum(torch.square(f[i][0] - S_p_j_k[i][0])) + \
                                   coeff_list[i][1] * torch.sum(torch.square(f[i][1] - S_p_j_k[i][1])) + \
                                   coeff_list[i][2] * torch.sum(torch.square(f[i][2] - S_p_j_k[i][2]))

        loss = consistency_loss + norm_loss + distrib_of_rules_loss + ro
        return loss, consistency_loss, norm_loss, ro, ro_vec, distrib_of_rules_loss

    def __train(self, f, num_of_rules, dimension, train_dict, init_z, distrib_of_rules,
                min_cons,
                min_norm,
                min_distr,
                check_target_values,
                plot_gradien_loss, a, h,
                coeff_list,
                print_tmp_cons_and_loss,
                ):
        train_info = {}

        z = []

        distrib_of_rules_torch = torch.tensor(distrib_of_rules)

        for i in range(num_of_rules):
            z.append(torch.log(init_z[i]).clone().detach().requires_grad_(True))

        optimizer = torch.optim.Adam(z, train_dict["lr"], [0.5, 0.7])

        loss_vec = np.zeros(train_dict["max_num_of_epochs"], )
        consyst_vec = np.zeros(train_dict["max_num_of_epochs"], )
        norm_vec = np.zeros(train_dict["max_num_of_epochs"], )

        ro_vec = np.zeros(train_dict["max_num_of_epochs"], )
        distr_of_rules_vec = np.zeros(train_dict["max_num_of_epochs"], )

        last_lr = train_dict["lr"]

        # max_squares
        max_squares = []
        for i in range(num_of_rules):
            max_squares.append([])
            for j in range(dimension):
                max = torch.square(torch.max(f[i][j]))
                min = torch.square(torch.min(f[i][j]))
                value_to_vzvesh = (max - min) / 20
                max_squares[i].append(torch.square(value_to_vzvesh))

        last_index_for_plot = 0
        max_index = 5
        index_of_mean_computation = 5

        N_half = int(train_dict["max_num_of_epochs"]/2)

        for i in range(train_dict["max_num_of_epochs"]):
            if i in train_dict:
                for g in optimizer.param_groups:
                    g['lr'] = train_dict[i]["lr"]
                    last_lr = train_dict[i]["lr"]

            if i > max_index:
                last_mean_loss = np.mean(loss_vec[i - index_of_mean_computation:i])
                last_loss = loss_vec[i - 1]

                if last_loss < last_mean_loss * 0.9:
                    last_lr = last_lr * 0.99
                    for g in optimizer.param_groups:
                        g['lr'] = last_lr

            optimizer.zero_grad()
            loss, consistency, norm, ro, ro_rules_values, distr_of_rules = self.__comp_loss(z, f, num_of_rules, a, h,
                                                                                            coeff_list,
                                                                                            init_z=init_z,
                                                                                            distrib_of_rules=distrib_of_rules_torch,
                                                                                            index_of_iteration=i,
                                                                                            index_of_apply_raznost=N_half,
                                                                                            max_squares=max_squares
                                                                                            )
            norm_loss = float(norm.cpu().detach().numpy())
            cons_loss = float(consistency.cpu().detach().numpy())
            ro_loss = float(ro.cpu().detach().numpy())
            distr_of_rules_loss = float(distr_of_rules.cpu().detach().numpy())

            if print_tmp_cons_and_loss == True:
                print(
                    "\r>>   {} ep lr {:10.11f} consistency: {:10.11f}   norm: {:10.11f}  ro {:10.11f} distr {:10.11f}".format(
                        i, last_lr,
                        cons_loss,
                        norm_loss,
                        ro_loss, distr_of_rules_loss
                    ),
                    end='')
            loss_for_plot = float(loss.cpu().detach().numpy())

            loss_vec[i] = loss_for_plot
            consyst_vec[i] = cons_loss
            norm_vec[i] = norm_loss

            ro_vec[i] = ro_loss
            distr_of_rules_vec[i] = distr_of_rules_loss
            # reg_vec[i]= reg_loss

            # if (cons_loss < 0.005 and norm_loss < 0.001 and distr_of_rules_loss < 0.001):
            if check_target_values == True:
                if (cons_loss < min_cons and norm_loss < min_norm and distr_of_rules_loss < min_distr):
                    last_index_for_plot = i
                    break
            loss.backward()
            optimizer.step()

        for i in range(len(z)):
            z[i] = torch.exp(z[i]).detach().numpy()

        if plot_gradien_loss == True:
            fig_loss, axs_loss = plt.subplots(1, 4)
            if last_index_for_plot == 0:
                last_index_for_plot = train_dict["max_num_of_epochs"] - 1
            # loss_line, = axs_loss[0].plot(loss_vec[:last_index_for_plot])
            # axs_loss[0].set_title("loss")
            # axs_loss[0].set_yscale("log")
            consistency_line, = axs_loss[0].plot(consyst_vec[:last_index_for_plot])
            axs_loss[0].set_title("consistency")
            axs_loss[0].set_yscale("linear")

            norm_line, = axs_loss[1].plot(norm_vec[:last_index_for_plot])
            axs_loss[1].set_title("norm")
            axs_loss[1].set_yscale("linear")

            ro_line, = axs_loss[2].plot(ro_vec[:last_index_for_plot])
            axs_loss[2].set_title("ro")
            axs_loss[2].set_yscale("linear")

            distr_line = axs_loss[3].plot(distr_of_rules_vec[:last_index_for_plot])
            axs_loss[3].set_title("distr")
            axs_loss[3].set_yscale("linear")

            plt.show(block=True)

        train_info.update({"last_consistency": consyst_vec[-1]})
        train_info.update({"last_norm": norm_vec[-1]})
        train_info.update({"last_distr": distr_of_rules_vec[-1]})
        return z, train_info

    def train(self, num_of_rules, dimension, rules, omega, train_dict, a, h, f, coeff_list, init_z,
              distrib_of_rules,
              check_target_values,
              min_cons,
              min_norm,
              min_distr,
              print_time_of_this_func, plot_gradien_loss, plot_consystency, print_num_of_restart_gradient,
              print_tmp_cons_and_loss):
        start_time_of_this_func = 0
        if print_time_of_this_func == True:
            start_time_of_this_func = time.time()
        l = 0
        while (True):
            z_list, train_info = self.__train(f=f, num_of_rules=num_of_rules, dimension=dimension,
                                              train_dict=train_dict, init_z=init_z,
                                              distrib_of_rules=distrib_of_rules,
                                              min_cons=min_cons,
                                              min_norm=min_norm,
                                              min_distr=min_distr,
                                              check_target_values=check_target_values,
                                              plot_gradien_loss=plot_gradien_loss, a=a, h=h,
                                              coeff_list=coeff_list,
                                              print_tmp_cons_and_loss=print_tmp_cons_and_loss)
            l += 1
            # if train_info["last_consistency"] < 0.7 and train_info["last_norm"] < 0.001:
            if check_target_values == True:
                if train_info["last_consistency"] < min_cons and train_info["last_norm"] < min_norm and train_info[
                    "last_distr"] < min_distr:
                    if (l > 1):
                        if print_num_of_restart_gradient == True:
                            print("\nлишних попыток на взятие градиентов {}".format(l - 1))
                    if plot_consystency == True:
                        plot_consistency(z_list, rules, omega, a, h)
                    break
            else:
                if plot_consystency == True:
                    plot_consistency(z_list, rules, omega, a, h)
                break
        if print_time_of_this_func == True:
            print("\n     time of gradients: {} sek".format(time.time() - start_time_of_this_func))

        return z_list


def intergate_list_of_rules_on_tmp_new_omega(p_xi_eta_gamma, tmp_new_omega, list_of_rules,
                                             print_time_of_integration_for_each_rect):
    if print_time_of_integration_for_each_rect == True:
        print("     integrate z di for {} rules".format(list_of_rules))
    size_of_new_z_tensor = []
    for i in range(len(tmp_new_omega)):
        size_of_new_z_tensor.append(len(tmp_new_omega[i]))
    new_z = np.zeros(tuple(size_of_new_z_tensor))

    list_of_args = np.zeros(len(size_of_new_z_tensor), )

    start_time = 0
    if print_time_of_integration_for_each_rect == True:
        start_time = time.time()

    for i_1 in range(size_of_new_z_tensor[0]):
        for i_2 in range(size_of_new_z_tensor[1]):
            for i_3 in range(size_of_new_z_tensor[2]):
                list_of_args[0] = tmp_new_omega[0][i_1][0] + (
                        tmp_new_omega[0][i_1][1] - tmp_new_omega[0][i_1][0]) / 2
                list_of_args[1] = tmp_new_omega[1][i_2][0] + (
                        tmp_new_omega[1][i_2][1] - tmp_new_omega[1][i_2][0]) / 2
                list_of_args[2] = tmp_new_omega[2][i_3][0] + (
                        tmp_new_omega[2][i_3][1] - tmp_new_omega[2][i_3][0]) / 2
                sum = 0.0
                for p_index in list_of_rules:
                    sum += p_xi_eta_gamma(list_of_args, p_index)
                new_z[i_1][i_2][i_3] = sum
    if print_time_of_integration_for_each_rect == True:
        print("     time of integration: {} sek".format(time.time() - start_time))
        # print("########################################################################")
    return new_z


class Integrator:
    path_to_file: str
    shared_integration_supports: Dict[str, Any]

    def __init__(self, dir_: str, shared_data: Dict[str, Any], clear_cache: bool = False):
        self.path_to_file = os.path.join(dir_, 'integration_supports.txt')
        if clear_cache == True:
            if os.path.exists(dir_):
                if os.path.exists(self.path_to_file):
                    os.remove(self.path_to_file)

        if not os.path.exists(self.path_to_file):
            if not os.path.exists(dir_):
                os.makedirs(dir_)

            list_of_rect = shared_data['list_of_rect']
            np_omega = shared_data['np_omega']

            tensor_with_info, omega_for_rules = Integrator.__make_tensor_from_list_of_rects(list_of_rect=list_of_rect)
            list_for_integrate = Integrator.__make_list_for_integrate(omega_for_rules=omega_for_rules,
                                                                      tensor_with_info=tensor_with_info)
            new_omega_list = Integrator.__make_new_omega_list(np_omega=np_omega, list_for_integrate=list_for_integrate,
                                                              print_num_of_rect=True)

            projection_to_x_y_info = Integrator.__find_the_projection_to_X_of_the_superposition_of_grids(new_omega_list,
                                                                                                         list_for_integrate)
            Rects, Grids = Integrator.__make_Rects_Grids(projection_to_x_y_info=projection_to_x_y_info)

            self.shared_integration_supports = {
                'new_omega_list': new_omega_list,
                'list_for_integrate': list_for_integrate,
                'projection_to_x_y_info': projection_to_x_y_info,
                'Rects': Rects,
                'Grids': Grids
            }
            torch.save(self.shared_integration_supports, self.path_to_file)
        else:
            self.shared_integration_supports = torch.load(self.path_to_file)

    @staticmethod
    def __make_tensor_from_list_of_rects(list_of_rect):
        # функция не тестировалась
        axis_left_sides = []
        axis_right_sides = []
        for axis in range(len(list_of_rect[0])):
            axis_left_sides.append([])
            axis_right_sides.append([])
            for rect_index in range(len(list_of_rect)):
                axis_left_sides[axis].append(list_of_rect[rect_index][axis][0])
                axis_right_sides[axis].append(list_of_rect[rect_index][axis][1])
        axis_left_sides = np.asarray(axis_left_sides)
        axis_right_sides = np.asarray(axis_right_sides)
        mins_of_x_i = np.zeros(len(axis_left_sides), )
        maxs_of_x_i = np.zeros(len(axis_right_sides), )
        for axis in range(len(axis_left_sides)):
            mins_of_x_i[axis] = np.min(axis_left_sides[axis])
            maxs_of_x_i[axis] = np.max(axis_right_sides[axis])

        omega_for_rules = []
        for i in range(len(list_of_rect[0])):
            omega_for_rules.append([mins_of_x_i[i]])
        for i in range(len(list_of_rect[0])):
            while True:
                tmp_left_border = omega_for_rules[i][-1]
                if tmp_left_border == maxs_of_x_i[i]:
                    break
                tmp_min_new_left_border = []
                args_of_tmp_min_new_left_border = []
                for p in range(len(list_of_rect)):
                    if tmp_left_border > list_of_rect[p][i][1]:
                        continue
                    for k in range(len(list_of_rect[p][i])):
                        if list_of_rect[p][i][k] > tmp_left_border:
                            tmp_min_new_left_border.append(list_of_rect[p][i][k])
                            break
                tmp = np.asarray(tmp_min_new_left_border)
                tmp_min = np.min(tmp)
                omega_for_rules[i].append(tmp_min)
        return_omega_for_rules = []
        for i in range(len(mins_of_x_i)):
            return_omega_for_rules.append([])
            for j in range(len(omega_for_rules[i]) - 1):
                return_omega_for_rules[i].append([omega_for_rules[i][j], omega_for_rules[i][j + 1]])

        # хардкод можно переписать с помощью рекурсии
        # инициализация
        tensor_with_information = []
        for i_1 in range(len(return_omega_for_rules[0])):
            tensor_with_information.append([])
            for i_2 in range(len(return_omega_for_rules[1])):
                tensor_with_information[i_1].append([])
                for i_3 in range(len(return_omega_for_rules[2])):
                    tensor_with_information[i_1][i_2].append([])
        # заполнение информацией о накрытии каким-то(ми-то) правилом текущего i_1 i_2 i_3 i_4 -го прямоугольника
        for i_1 in range(len(return_omega_for_rules[0])):
            for i_2 in range(len(return_omega_for_rules[1])):
                for i_3 in range(len(return_omega_for_rules[2])):
                    segment_0 = return_omega_for_rules[0][i_1]
                    segment_1 = return_omega_for_rules[1][i_2]
                    segment_2 = return_omega_for_rules[2][i_3]
                    for index_of_rect in range(len(list_of_rect)):
                        if segment_0[0] >= list_of_rect[index_of_rect][0][0] and segment_0[1] <= \
                                list_of_rect[index_of_rect][0][1]:
                            if segment_1[0] >= list_of_rect[index_of_rect][1][0] and segment_1[1] <= \
                                    list_of_rect[index_of_rect][1][1]:
                                if segment_2[0] >= list_of_rect[index_of_rect][2][0] and segment_2[1] <= \
                                        list_of_rect[index_of_rect][2][1]:
                                    tensor_with_information[i_1][i_2][i_3].append(index_of_rect)

        return tensor_with_information, return_omega_for_rules

    @staticmethod
    def __make_list_for_integrate(omega_for_rules, tensor_with_info):
        list_for_integrate = []
        #   пройдемся по всем элементам tensor_with_information_about_intersection и проинтегрируем там где не пустой элемент
        for i_1 in range(len(omega_for_rules[0])):
            for i_2 in range(len(omega_for_rules[1])):
                for i_3 in range(len(omega_for_rules[2])):
                    if len(tensor_with_info[i_1][i_2][i_3]) > 0:
                        # в области определения соответствующей прямоугольнику с индексами i_1 i_2 i_3 i_4
                        # пересеклись опредлененные правила. их список мы знаем. это может быть и одно правило, тогда
                        # длина списка = 1, но если длина списка = 1 то не нужно ни копировать ,ни интегрировать
                        # если длина списка > 1 то нужно посчитать интеграл
                        # зная индексы мы можем сказать левые и правые границы по каждой из осей этого прямоугольника
                        list_for_integrate.append(
                            [
                                [omega_for_rules[0][i_1], omega_for_rules[1][i_2], omega_for_rules[2][i_3]],

                                tensor_with_info[i_1][i_2][i_3]
                            ]
                        )
        return list_for_integrate

    @staticmethod
    def __make_new_omega_list(np_omega, list_for_integrate, print_num_of_rect):
        new_omega_list = []
        num_of_rect_in_intersection_list = []
        for i in range(len(list_for_integrate)):
            tmp_new_omega, num_of_rect_in_intersection = make_new_omega_for_rect(old_omega=np_omega,
                                                                                 rect=list_for_integrate[i][0],
                                                                                 list_of_rules=list_for_integrate[i][1])
            new_omega_list.append(tmp_new_omega)
            num_of_rect_in_intersection_list.append(num_of_rect_in_intersection)
        if print_num_of_rect == True:
            print("     num_of_rect_in_intersection {}".format(sum(num_of_rect_in_intersection_list)))
        return new_omega_list

    @staticmethod
    def __find_the_projection_to_X_of_the_superposition_of_grids(new_omega_list, list_for_integrate):
        unique_x_y_areas = {}
        for i in range(len(list_for_integrate)):
            rect = list_for_integrate[i][0]
            x_area_of_rect = rect[0]
            y_area_of_rect = rect[1]
            # проверим, не проверяли ли мы уже эту область в словаре unique_x_y_areass
            already_in_dict = 0
            for key in unique_x_y_areas.keys():
                rect_x_y = unique_x_y_areas[key]
                x_in_rect = rect_x_y[0]
                y_in_rect = rect_x_y[1]
                if x_in_rect[0] == x_area_of_rect[0] and x_in_rect[1] == x_area_of_rect[1] and y_in_rect[0] == \
                        y_area_of_rect[0] and y_in_rect[1] == y_area_of_rect[1]:
                    already_in_dict = 1
                    break
            if already_in_dict == 1:
                continue
            else:
                unique_x_y_areas.update({i: rect})
        areas_of_x_y = []
        for key in unique_x_y_areas.keys():
            rect = unique_x_y_areas[key]
            areas_of_x_y.append([[rect[0][0], rect[0][1]], [rect[1][0], rect[1][1]]])
        # pprint.pprint(areas_of_x_y)
        # plt.rcParams["figure.figsize"] = [14, 7]
        # plt.rcParams["figure.autolayout"] = True
        # fig = plt.figure()
        # axs = fig.add_subplot(111)
        # for i in range(len(areas_of_x_y)):
        #     rect = areas_of_x_y[i]
        #     plot_rect_for_rules(axs,rect[0][0],rect[0][1],rect[1][0],rect[1][1],'g')
        #
        # axs.set_xlim([-1, 362])
        # axs.set_ylim([-1, 23])
        # axs.set_xlabel(r"$расстояние \:до \:стоп-линии  \: x_{1}, \: м$")
        # axs.set_ylabel(r'$скорость \: x_{2}, \: м \cdot c^{-1}$')
        # plt.title(r'$ускорение \: y_{1}, \: м\cdot c^{-2} $')
        # plt.show(block=True)

        # мы получили все уникальные области по (x,y)
        # теперь нужно пройтись по всем областям вдоль оси z, которые попадают по (x,y) в эти найденные области
        # взять, все эти разбиения по (x,y) и найти их суперпозицию
        output = []
        for i in range(len(areas_of_x_y)):
            x_y_of_reference_area = areas_of_x_y[i]
            x_of_reference_area = x_y_of_reference_area[0]
            y_of_reference_area = x_y_of_reference_area[1]
            # найдем индексы всех областей которые попадат в данный прямоугольник
            areas_that_contain_x_y_area = []
            for j in range(len(list_for_integrate)):
                rect = list_for_integrate[j][0]
                x_of_tmp_area = rect[0]
                y_of_tmp_area = rect[1]
                if x_of_reference_area[0] == x_of_tmp_area[0] and x_of_reference_area[1] == x_of_tmp_area[1] and \
                        y_of_reference_area[0] == \
                        y_of_tmp_area[0] and y_of_reference_area[1] == y_of_tmp_area[1]:
                    areas_that_contain_x_y_area.append(j)

            grids_of_x = []
            grids_of_y = []
            for index_of_area in areas_that_contain_x_y_area:
                grids_of_x.append(new_omega_list[index_of_area][0])
                grids_of_y.append(new_omega_list[index_of_area][1])
            grid_of_x = []
            for j in range(len(grids_of_x)):
                for k in range(len(grids_of_x[j])):
                    grid_of_x.append(grids_of_x[j][k][0])
                    grid_of_x.append(grids_of_x[j][k][1])
            superposition_of_x = np.unique(grid_of_x)
            grid_of_y = []
            for j in range(len(grids_of_y)):
                for k in range(len(grids_of_y[j])):
                    grid_of_y.append(grids_of_y[j][k][0])
                    grid_of_y.append(grids_of_y[j][k][1])
            superposition_of_y = np.unique(grid_of_y)

            superposition_of_grids_in_reference_area = [[], []]
            for j in range(len(superposition_of_x) - 1):
                superposition_of_grids_in_reference_area[0].append([superposition_of_x[j], superposition_of_x[j + 1]])
            for j in range(len(superposition_of_y) - 1):
                superposition_of_grids_in_reference_area[1].append([superposition_of_y[j], superposition_of_y[j + 1]])
            output.append(
                [x_y_of_reference_area, areas_that_contain_x_y_area, superposition_of_grids_in_reference_area])
        return output

    @staticmethod
    def __make_Rects_Grids(projection_to_x_y_info):
        Rects = [[] for rect_index in range(len(projection_to_x_y_info))]
        Grids = [[] for rect_index in range(len(projection_to_x_y_info))]
        for i in range(len(projection_to_x_y_info)):
            info = projection_to_x_y_info[i]
            rect_x_y = info[0]
            Rects[i] = rect_x_y
            superposition_of_grids_in_reference_area = info[2]
            Grids[i] = superposition_of_grids_in_reference_area
        return Rects, Grids

    @staticmethod
    def make_policy_function(p_xi_eta_gamma,
                             new_omega_list,
                             list_for_integrate,
                             projection_to_x_y_info,
                             shared_Rects,
                             shared_Grids,
                             print_time_of_this_func=False,
                             print_time_of_integration_for_each_rect=False,
                             print_tmp_computed=False):

        start_time_of_integration = 0
        if print_time_of_this_func == True:
            start_time_of_integration = time.time()

        num_of_intersections_areas = 0
        if print_tmp_computed == True:
            for i in range(len(list_for_integrate)):
                if len(list_for_integrate[i][1]) > 1:
                    num_of_intersections_areas += 1

        list_of_integrals_z_di = []
        tmp_computed = 0
        for i in range(len(list_for_integrate)):

            if len(list_for_integrate[i][1]) > 1:

                integral_in_rect = intergate_list_of_rules_on_tmp_new_omega(p_xi_eta_gamma=p_xi_eta_gamma,
                                                                            tmp_new_omega=new_omega_list[i],
                                                                            list_of_rules=list_for_integrate[i][1],
                                                                            print_time_of_integration_for_each_rect= \
                                                                                print_time_of_integration_for_each_rect)
                list_of_integrals_z_di.append(integral_in_rect)
                if print_tmp_computed == True:
                    tmp_computed += 1
                    print("computed {} of {}".format(tmp_computed, num_of_intersections_areas))

        p_func = PolicyFucntionMaker.make_policy_function(p_xi_eta_gamma=p_xi_eta_gamma,
                                                          projection_to_x_y_info=projection_to_x_y_info,
                                                          list_for_integrate=list_for_integrate,
                                                          new_omega_list=new_omega_list,
                                                          list_of_integrals_z_di=list_of_integrals_z_di,
                                                          Rects=shared_Rects,
                                                          Grids=shared_Grids)

        if print_time_of_this_func == True:
            print("\n     time of integration: {} sek".format(time.time() - start_time_of_integration))

        return p_func


def get_indicies(input, new_omega):
    indicies = []
    for i in range(len(input)):
        for j in range(len(new_omega[i])):
            if ((input[i] >= new_omega[i][j][0]) and (input[i] <= new_omega[i][j][1])):
                indicies.append(j)
                break
    return tuple(indicies)


def copy_p_from_list_of_integrals_to_x_y_list(
        projection_to_x_y_info,
        old_z, old_omega, list_for_integrate, new_omega_list,
        list_of_integrals_z_di):
    # у нас есть проинегрированная плотность(там где это нужно),н-р, из 122 областей, областей где проитегрирована плотность всего
    # 47 штук. но эти 47 штук не пронумерованы, к какой области из 122 они тносятся
    # получим эти индексы
    list_of_indicies_for_indegrated_areas = []  # массив индексов областей, в которых была проинтегрирована плотность
    for i in range(len(list_for_integrate)):
        if len(list_for_integrate[i][1]) > 1:
            list_of_indicies_for_indegrated_areas.append(i)
    list_of_indicies_for_indegrated_areas = np.asarray(list_of_indicies_for_indegrated_areas)
    list_of_z_and_p_with_fixed_x_y = []
    for i in range(len(projection_to_x_y_info)):
        info = projection_to_x_y_info[i]
        x_y_of_reference_area = info[0]
        areas_that_contain_x_y_area = info[1]
        superposition_of_grids_in_reference_area = info[2]
        list_of_z = []
        list_of_p = []

        for j in range(len(superposition_of_grids_in_reference_area[0])):
            list_of_z.append([])
            list_of_p.append([])
            for k in range(len(superposition_of_grids_in_reference_area[1])):

                a_x = superposition_of_grids_in_reference_area[0][j][0]
                b_x = superposition_of_grids_in_reference_area[0][j][1]
                a_y = superposition_of_grids_in_reference_area[1][k][0]
                b_y = superposition_of_grids_in_reference_area[1][k][1]
                input = [(a_x + b_x) / 2, (a_y + b_y) / 2]
                # Зафиксировали (x,y) в этой области и теперь пройдемся вдоль направления z и скопируем данные(само разбиение по z и плотности им соответствующие)
                superposition_of_z_throught_all_areas = []
                all_p_of_z_throught_all_areas = []
                for index_of_area in areas_that_contain_x_y_area:
                    all_rules_in_this_area = list_for_integrate[index_of_area][1]
                    z_in_this_area = new_omega_list[index_of_area][2]
                    p_in_this_area = []
                    if len(all_rules_in_this_area) > 1:
                        # мы попали в проинтегрированную область
                        index_of_integral = np.where(list_of_indicies_for_indegrated_areas == index_of_area)
                        index_of_integral = int(index_of_integral[0])
                        array_of_p = list_of_integrals_z_di[index_of_integral]
                        indicies_of_x_y = get_indicies(input, new_omega_list[index_of_area])
                        p_in_this_area = array_of_p[indicies_of_x_y[0], indicies_of_x_y[1], :]
                    else:
                        # мы попали в непроинтегрированную область
                        p = all_rules_in_this_area[0]
                        array_of_p = old_z[p]
                        a_z = list_for_integrate[index_of_area][0][2][0]
                        b_z = list_for_integrate[index_of_area][0][2][1]
                        start_index_of_z = -1
                        stop_index_of_z = -1

                        for tmp_index in range(len(old_omega[p][2])):
                            otrezok = old_omega[p][2][tmp_index]
                            if a_z >= otrezok[0] and a_z <= otrezok[1]:
                                start_index_of_z = tmp_index
                            if b_z >= otrezok[0] and b_z <= otrezok[1]:
                                stop_index_of_z = tmp_index
                                break

                        index_of_x = -1
                        for tmp_index in range(len(old_omega[p][0])):
                            otrezok = old_omega[p][0][tmp_index]
                            if input[0] >= otrezok[0] and input[0] <= otrezok[1]:
                                index_of_x = tmp_index
                                break
                        index_of_y = -1
                        for tmp_index in range(len(old_omega[p][1])):
                            otrezok = old_omega[p][1][tmp_index]
                            if input[1] >= otrezok[0] and input[1] <= otrezok[1]:
                                index_of_y = tmp_index
                                break

                        p_in_this_area = array_of_p[index_of_x, index_of_y, start_index_of_z:stop_index_of_z + 1]

                    superposition_of_z_throught_all_areas.append(z_in_this_area)
                    all_p_of_z_throught_all_areas.append(p_in_this_area)
                output_superpos_of_z = []
                output_superpos_of_p = []
                for index_of_chunk in range(len(superposition_of_z_throught_all_areas)):
                    for index_of_otrezok in range(len(superposition_of_z_throught_all_areas[index_of_chunk])):
                        output_superpos_of_z.append(
                            superposition_of_z_throught_all_areas[index_of_chunk][index_of_otrezok])
                        output_superpos_of_p.append(all_p_of_z_throught_all_areas[index_of_chunk][index_of_otrezok])
                list_of_z[j].append(np.asarray(output_superpos_of_z))
                list_of_p[j].append(np.asarray(output_superpos_of_p))

        list_of_z_and_p_with_fixed_x_y.append([list_of_z, list_of_p])
    return list_of_z_and_p_with_fixed_x_y


@jit(nopython=True, cache=True)
def find_rect_of_grid(x_1:float, x_2:float, shared_Rects_: np.array)->int:
    index_of_rect = 99999
    for i in range(len(shared_Rects_)):
        rect = shared_Rects_[i]
        if not (rect[0][0] <= x_1 <= rect[0][1]):
            continue
        else:
            if not (rect[1][0] <= x_2 <= rect[1][1]):
                continue
            else:
                index_of_rect = i
                break
    return index_of_rect

@jit(nopython=True,cache=True)
def get_x_y_index(xgrid: np.array, ygrid:np.array, x_1: float, x_2: float):
    x_index = -1
    for x_segment_index in range(len(xgrid)):
        x_segment = xgrid[x_segment_index]
        if x_segment[0] <= x_1 <= x_segment[1]:
            x_index = x_segment_index
            break
    y_index = -1
    for y_segment_index in range(len(ygrid)):
        y_segment = ygrid[y_segment_index]
        if y_segment[0] <= x_2 <= y_segment[1]:
            y_index = y_segment_index
            break
    return x_index, y_index


def get_ans_by_input(x_1: float, x_2: float,
                     shared_Rects_,
                     shared_XGrids_,
                     shared_YGrids_,
                     __P
                     ) -> float:
    index_of_rect = find_rect_of_grid(x_1, x_2, shared_Rects_)
    # найти сегмент в прямоугольнике, в котором лежит точка (x_1,x_2)
    xgrid = shared_XGrids_[index_of_rect]
    ygrid = shared_YGrids_[index_of_rect]
    #
    x_index, y_index = get_x_y_index(xgrid,ygrid, x_1, x_2)
    return __P[index_of_rect][x_index][y_index]


class PolicyFunction:
    # list_of_z_and_p_with_fixed_x_y = copy_p_from_list_of_integrals_to_x_y_list(projection_to_x_y_info,
    #     z_np,np_omega, list_for_integrate, new_omega_list,
    #     list_of_integrals_z_di)
    __P: List[np.array]
    shared_Rects_: List[np.array]
    shared_Grids_: List[np.array]

    def __init__(self,
                 p_xi_eta_gamma,
                 projection_to_x_y_info,
                 list_for_integrate,
                 new_omega_list,
                 list_of_integrals_z_di,
                 Rects,
                 Grids
                 ):
        list_of_z_and_p_with_fixed_x_y = copy_p_from_list_of_integrals_to_x_y_list(
            projection_to_x_y_info=projection_to_x_y_info,
            old_z=p_xi_eta_gamma.z_list,
            old_omega=p_xi_eta_gamma.np_omega,
            list_for_integrate=list_for_integrate,
            new_omega_list=new_omega_list,
            list_of_integrals_z_di=list_of_integrals_z_di)
        self.__P = PolicyFunction.__compute_P(list_of_z_and_p_with_fixed_x_y, Grids)
        # self.shared_Grids_ = [np.asarray(Grids[i],dtype=object) for i in range(len(Grids))]
        # self.shared_Rects_ = [np.asarray(Rects[i],dtype=object) for i in range(len(Rects))]
        # n_list = numba.typed.List()
        # for i in range(len(Grids)):
        #     # tmp_list = numba.typed.List()
        #     grid_i = Grids[i]
        #     x_segments_array = np.asarray(grid_i[0])
        #     y_segments_array = np.asarray(grid_i[1])
        #     x_y_segments = numba.typed.List()
        #     x_y_segments.append(x_segments_array)
        #     x_y_segments.append(y_segments_array)
        #     n_list.append(x_y_segments)

        self.shared_XGrids_ = []
        self.shared_YGrids_ = []
        for i in range(len(Grids)):
            grid_i = Grids[i]
            x_segments_array = np.asarray(grid_i[0])
            y_segments_array = np.asarray(grid_i[1])
            self.shared_XGrids_.append(x_segments_array)
            self.shared_YGrids_.append(y_segments_array)

        self.shared_Rects_ = np.asarray(Rects)

    def __call__(self, x_1: float, x_2: float) -> float:
        # '''
        #
        # :param x_1: x_1 input e.g. theta V
        # :param x_2: x_2 input e.g. omega V
        # :return: y_1: y_1 output e.g. force V
        # '''
        # # найти прямоугольник, в котором лежит точка (x_1,x_2)
        # index_of_rect = -1
        # for i in range(len(self.shared_Rects_)):
        #     rect = self.shared_Rects_[i]
        #     if not (rect[0][0] <= x_1 <= rect[0][1]):
        #         continue
        #     else:
        #         if not (rect[1][0] <= x_2 <= rect[1][1]):
        #             continue
        #         else:
        #             index_of_rect = i
        #             break
        # # найти сегмент в прямоугольнике, в котором лежит точка (x_1,x_2)
        # grid = self.shared_Grids_[index_of_rect]
        # #
        # x_index = -1
        # for x_segment_index in range(len(grid[0])):
        #     x_segment = grid[0][x_segment_index]
        #     if x_segment[0] <= x_1 <= x_segment[1]:
        #         x_index = x_segment_index
        #         break
        # y_index = -1
        # for y_segment_index in range(len(grid[1])):
        #     y_segment = grid[1][y_segment_index]
        #     if y_segment[0] <= x_2 <= y_segment[1]:
        #         y_index = y_segment_index
        #         break
        # return self.__P[index_of_rect][x_index][y_index]
        return get_ans_by_input(x_1, x_2,
                                self.shared_Rects_,
                                self.shared_XGrids_,
                                self.shared_YGrids_,
                                self.__P)

    @staticmethod
    def __compute_P(list_of_z_and_p_with_fixed_x_y, Grids):
        P = [[] for rect_index in range(len(Grids))]
        # P = []
        for i in range(len(Grids)):
            # Grids[i]:
            # [
            #   [1segment_of_x,2_segment_of_x,...],
            #   [1segment_of_y,2_segment_of_y,...]
            # ]
            # P.append(np.zeros(shape=(len(Grids[i][0]), len(Grids[i][1]))))
            P[i] = [[] for x_grid_segment_index in range(len(Grids[i][0]))]
            for j in range(len(Grids[i][0])):
                P[i][j] = [0.0 for y_grid_segment_index in range(len(Grids[i][1]))]
                for k in range(len(Grids[i][1])):
                    # # mean                                      r_i  x_j y_k
                    all_z = list_of_z_and_p_with_fixed_x_y[i][0][j][k]
                    all_p = list_of_z_and_p_with_fixed_x_y[i][1][j][k]
                    # compute integral p(x_1,x_2,y_1)dy_1 and y_1p(x_1,x_2,y_1)dy_1
                    int_p_x1_x2_y1_dy1 = 0.0
                    int_y_mul_p_x1_x2_y1_dy1 = 0.0
                    for int_index in range(all_z.shape[0]):
                        a = all_z[int_index][0]
                        b = all_z[int_index][1]
                        delta_z = b - a
                        p_ = all_p[int_index]  # constant value on integratation segment
                        int_p_x1_x2_y1_dy1 += delta_z * p_
                        int_y_mul_p_x1_x2_y1_dy1 += p_ * (np.square(b) - np.square(a)) / 2
                    answer_to_controll_query = int_y_mul_p_x1_x2_y1_dy1 / int_p_x1_x2_y1_dy1
                    P[i][j][k] = answer_to_controll_query
                    # argmax
                    # all_z = list_of_z_and_p_with_fixed_x_y[i][0][j][k]
                    # all_p = list_of_z_and_p_with_fixed_x_y[i][1][j][k]
                    # argmax = np.argmax(all_p)
                    # segment_of_answer  = all_z[argmax]
                    # P[i][j][k] = (segment_of_answer[0]+segment_of_answer[1])/2

        return P


class Distrib4D:
    z_list: Any
    np_omega: Any
    def modify_z_in_rule(self, indexes_of_rules: np.array,num_of_rules:int) -> None:
        rule_probability = 1.0/num_of_rules
        for ind in indexes_of_rules:
            shape_of_z = np.shape(self.z_list[ind])
            tmp = np.random.rand(*shape_of_z)
            tmp = tmp/np.sum(tmp)*rule_probability
            self.z_list[ind] = tmp
    def cast_to_float32(self):
        for i in range(len(self.z_list)):
            self.z_list[i] = self.z_list[i].astype(dtype=np.float32)

    def __compute_z(self, z, omega, list_of_args, p):
        """
        :param z:list of torch tensors
        :param omega: list of list of torch tensors
        :param list_of_args: x_1, x_2,...,x_8; x_1 in |R, x_2 in |R,...,x_8 in |R
        :param p: номер правила p=0,1,2,...80 например
        :return: z(x_1,x_2,...,x_8,i)
        """

        # нужно проверить не находится ли точка (x_1,x_2,...,x_8) вне прямоугольника
        for i in range(len(list_of_args)):
            if (list_of_args[i] < omega[p][i][0][0]) or (list_of_args[i] > omega[p][i][-1][1]):
                return 0.0

        list_of_indicies = []
        for i in range(len(list_of_args)):
            for k in range(len(omega[p][i])):
                if (list_of_args[i] >= omega[p][i][k][0]) and (list_of_args[i] <= omega[p][i][k][1]):
                    list_of_indicies.append(k)
                    break
        list_of_indicies = tuple(list_of_indicies)
        return z[p][list_of_indicies]

    def __init__(self, z_list_, np_omega_):
        self.z_list = z_list_
        self.np_omega = np_omega_

    def __call__(self, xi_eta_list, gamma):
        return self.__compute_z(self.z_list, self.np_omega, xi_eta_list, gamma)

@jit(nopython = True)
def get_index_of_segment_by_value(x:float, grid:np.array):
    N = len(grid)
    if x < grid[0] or x > grid[N-1]:
        return -1 
    for i in range(N-2):
        if x >= grid[i] and x < grid[i+1]:
            return i
    return N-2

class Distrib3D:
    grid: List[np.array]
    midx: np.float32
    midy: np.float32
    midz: np.float32
    values: np.array
    volumes: np.array
    def __init__(self, grid:List[np.array], values:np.array, volumes: np.array):
        # grid: index of dimension, index of point on grid 
        self.grid = grid 
        self.values = values
        midx= []
        gx = grid[0]
        for i in range(len(gx)-1):
            midx.append(0.5*(gx[i]+gx[i+1])) 

        midy= []
        gy = grid[1]
        for i in range(len(gy)-1):
            midy.append(0.5*(gy[i]+gy[i+1])) 

        midz= []
        gz = grid[2]
        for i in range(len(gz)-1):
            midz.append(0.5*(gz[i]+gz[i+1])) 
        midx = np.array(midx, dtype=np.float32)
        midy = np.array(midy, dtype=np.float32) 
        midz = np.array(midz, dtype=np.float32) 
        self.midx = midx
        self.midy = midy
        self.midz = midz
        self.volumes = volumes

    def __call__(self, x_1:float,x_2:float,x_3:float)->float:
        i_x = get_index_of_segment_by_value(x_1, self.grid[0])        
        i_y = get_index_of_segment_by_value(x_2, self.grid[1])        
        i_z = get_index_of_segment_by_value(x_3, self.grid[2])
        if i_x == -1 or i_y == -1 or i_z == -1:
            return 0.0
        else:
            return self.values[i_x][i_y][i_z]
    def math_expectation(self, x_1:float,x_2:float)->float:
        # out: math expectation of x_3 at (x_1,x_2) point
        i_x = get_index_of_segment_by_value(x_1, self.grid[0])        
        i_y = get_index_of_segment_by_value(x_2, self.grid[1])   
        if i_x == -1 or i_y == -1:
            return np.nan
        
        zgrid = self.grid[2]
        vs = self.values[i_x][i_y]
        delta = np.diff(zgrid)
        p_xyz =  vs
        p_xy  =  np.sum(vs*delta)
        cond_probs = p_xyz/p_xy
        return np.sum(0.5*cond_probs* np.diff(np.square(zgrid)))
    
    def random_choice(self, x_1:float,x_2:float)->float:
        i_x = get_index_of_segment_by_value(x_1, self.grid[0])        
        i_y = get_index_of_segment_by_value(x_2, self.grid[1])   
        if i_x == -1 or i_y == -1:
            return np.nan
        zgrid = self.grid[2]
        vs = self.values[i_x][i_y]
        delta = np.diff(zgrid)
        p_xyz =  vs
        p_xy  =  np.sum(vs*delta)
        cond_probs = p_xyz/p_xy
        probs_of_discrete_values = cond_probs*delta
        return np.random.choice(self.midz,p=probs_of_discrete_values)
    

def cup_grids(grids:List[List[np.array]]) -> List[np.array]:
    x_grids = [el[0] for el in grids]
    y_grids = [el[1] for el in grids]
    z_grids = [el[2] for el in grids]
    x_flatten = []
    for xgr in x_grids:
        for j in range(len(xgr)):
            x_flatten.append(xgr[j])
    x_flatten = np.unique(x_flatten)
    x_superposition = np.sort(x_flatten)

    y_flatten = []
    for ygr in y_grids:
        for j in range(len(ygr)):
            y_flatten.append(ygr[j])
    y_flatten = np.unique(y_flatten)
    y_superposition = np.sort(y_flatten)

    z_flatten = []
    for zgr in z_grids:
        for j in range(len(zgr)):
            z_flatten.append(zgr[j])
    z_flatten = np.unique(z_flatten)
    z_superposition = np.sort(z_flatten)
    return [x_superposition, y_superposition, z_superposition]

def weighted_amount_in_point(x_1, x_2, x_3, distributions:List[Distrib3D], alpha_vec:np.array):
    sum_ = 0.0
    # print(alpha_vec)
    # vs = []
    for i in range(len(distributions)):
        p_ = distributions[i]
        sum_ += alpha_vec[i]*p_(x_1, x_2, x_3)
        # vs.append(p_(x_1, x_2, x_3))
    # print(vs)
    # print(np.sum(vs))
    # raise SystemExit
    # print(sum_)
    return sum_


def make_distrib_from_weighted_sum_of_distributions(distributions:List[Distrib3D],alpha_vec:np.array):
    grids = [distributions[i].grid for i in range(len(distributions))]
    Grid = cup_grids(grids)
    nx  = len(Grid[0])-1
    ny = len(Grid[1])-1
    nz = len(Grid[2])-1
    Values = np.zeros(shape=(nx, ny, nz),dtype=np.float32)
    xgrid = Grid[0]
    ygrid = Grid[1]
    zgrid = Grid[2]
    Volumes = compute_volumes_from_grid(Grid)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x_mid = (xgrid[i]+xgrid[i+1])*0.5 
                y_mid = (ygrid[j]+ygrid[j+1])*0.5 
                z_mid = (zgrid[k]+zgrid[k+1])*0.5 
                Values[i][j][k] = weighted_amount_in_point(x_mid,
                                                           y_mid,
                                                           z_mid,
                                                           distributions,
                                                           alpha_vec) 
    Values = Values/np.sum(Values*Volumes)
    return Distrib3D(Grid,Values,Volumes)

def compute_volumes_from_grid(grid):
    xgrid,ygrid,zgrid = grid
    nx = len(xgrid)-1
    ny = len(ygrid)-1
    nz = len(zgrid)-1
    volumes = np.zeros(shape=(nx,ny,nz),dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v_ = (xgrid[i+1]-xgrid[i])*(ygrid[j+1]-ygrid[j])*(zgrid[k+1]-zgrid[k])
                volumes[i][j][k] = v_
    return volumes

def build_distrib3d_from_policy_matrix(p_from_policy_matrix,
                                       xgrid,ygrid,zgrid,number_of_states,number_of_actions):
    grid = [xgrid,ygrid,zgrid]
    nx = len(xgrid)-1
    ny = len(ygrid)-1
    nz = len(zgrid)-1
    values = np.zeros(shape=(nx,ny,nz),dtype=np.float32)
    volumes = compute_volumes_from_grid(grid)
    for i in range(nx):
        x_mid = (xgrid[i]+xgrid[i+1])*0.5
        deltax =  xgrid[i+1]-xgrid[i]
        for j in range(ny):
            y_mid = (ygrid[j]+ygrid[j+1])*0.5 
            deltay=  ygrid[j+1]-ygrid[j]
            for k in range(nz):
                z_mid = (zgrid[k]+zgrid[k+1])*0.5
                deltaz = zgrid[k+1]-zgrid[k]
                v_ = deltax*deltay*deltaz
                values[i][j][k] = p_from_policy_matrix(x_mid,y_mid,z_mid)*(1.0/number_of_states)/volumes[i][j][k]
    values = values/np.sum(values*volumes)
    # print(volumes)
    p_ = Distrib3D(grid,values,volumes)
    return p_




def build_p_from_policy_matrix(PI,N1,N2,M,xgrid,ygrid,zgrid):
    '''
    returns cond probs p(a|theta,omega)
    '''
    def p_(x1:float,x2:float,y1:float)->float:
        k1 = get_index_of_segment_by_value(x1, xgrid)
        k2 = get_index_of_segment_by_value(x2, ygrid)
        i_ = N2*k1+k2 # index of row
        k3 = get_index_of_segment_by_value(y1,zgrid) # index of column
        return PI[i_][k3]
    return p_

def convert_segments_grid_to_grid(segments_grid):
    # index of dim, index of segment, index of point in segment
    xsegments= segments_grid[0]
    ysegments= segments_grid[1]
    zsegments= segments_grid[2]
    xgrid = []
    for i in range(len(xsegments)):
        xgrid.append(xsegments[i][0])
        xgrid.append(xsegments[i][1])
    xgrid = np.array(
        np.unique(xgrid),dtype=np.float32
    )
    ygrid = []
    for i in range(len(ysegments)):
        ygrid.append(ysegments[i][0])
        ygrid.append(ysegments[i][1])
    ygrid = np.array(
        np.unique(ygrid),dtype=np.float32
    )
    zgrid = []
    for i in range(len(zsegments)):
        zgrid.append(zsegments[i][0])
        zgrid.append(zsegments[i][1])
    zgrid = np.array(
        np.unique(zgrid),dtype=np.float32
    )
    return [xgrid,ygrid,zgrid]

def plot_func_on_grid(func, x1grid,x2grid,xlabel=r"$\theta, \: V$",ylabel=r'$\omega, \: V$',title=r'$F, \: V $',size=(16,9)):
    fig,ax = plt.subplots()
    fig.set_size_inches(*size)

    x = []
    y = []
    rects_info = []
    highs = []
    num_of_points_in_plot = 0
    # в этом случае передали surf
    for i in range(len(x1grid)-1):
        for j in range(len(x2grid)-1):
            x_1 = x1grid[i]
            x_2 = x1grid[i+1]
            y_1 = x2grid[j]
            y_2 = x2grid[j+1]
            x.append((x_2 + x_1) / 2)
            y.append((y_2 + y_1) / 2)
            rects_info.append([x_1, x_2, y_1, y_2])
            expectation = func((x_2 + x_1) / 2, (y_2 + y_1) / 2)
            # print(expectation)
            highs.append(expectation)
            num_of_points_in_plot += 1
    norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
    m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    tmp_len = len(rects_info)
    for i in range(len(rects_info)):
        print("\r interation {} of {}".format(i, tmp_len), end='')
        plot_rect(ax, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],
                    m.to_rgba(highs[i]))

    plt.colorbar(m, ax=ax)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig,ax  






def weighted_amount(list_of_distributions: List[Distrib4D], alpha_list: np.array):
    # TODO : процедура интегрирвоания ни как не проверялась
    if len(list_of_distributions) <= 0:
        print('error')
        return
    if len(list_of_distributions) != len(alpha_list):
        print('error')
        return
    z_out = []
    n_r = len(list_of_distributions[0].z_list)
    for i in range(n_r):
        z_i = np.zeros(shape=list_of_distributions[0].z_list[i].shape)
        for j in range(len(list_of_distributions)):
            z_i += list_of_distributions[j].z_list[i] * alpha_list[j]
        z_out.append(z_i)

    output_distrib = Distrib4D(z_list_=z_out, np_omega_=list_of_distributions[0].np_omega)
    return output_distrib


def z_ro(z_list1, z_list2, a_shared):
    ro_12 = 0.0
    for i in range(len(z_list1)):
        f_1_kth_minus_f_2_kth = z_list1[i] - z_list2[i]
        square_ = np.square(f_1_kth_minus_f_2_kth)
        kth_integral = np.sum(square_ * a_shared[i])
        ro_12 += kth_integral
    return np.sqrt(ro_12)


def ro_Distrib4D(p_1: Distrib4D, p_2: Distrib4D, a_shared: List[np.array]):
    if len(p_1.z_list) != len(p_2.z_list):
        print('error')
        return
    # euclidean distance
    # ro_12 = 0.0
    # for i in range(len(p_1.z_list)):
    #     ro_12 += np.linalg.norm(p_1.z_list[i]-p_2.z_list[i])
    # integral distance
    # ro_12 = 0.0
    # for i in range(len(p_1.z_list)):
    #     f_1_kth_minus_f_2_kth = p_1.z_list[i]-p_2.z_list[i]
    #     square_ = np.square(f_1_kth_minus_f_2_kth)
    #     kth_integral = np.sum(square_*a_shared[i])
    #     ro_12 += kth_integral
    # return np.sqrt(ro_12)
    return z_ro(p_1.z_list, p_2.z_list, a_shared)


class DistrMaker:

    @staticmethod
    def make_distrib4D(x_: Dict[str, Any]):
        z_list = x_['z_list']
        np_omega = x_['np_omega']
        return Distrib4D(z_list_=z_list, np_omega_=np_omega)


class PolicyFucntionMaker:

    @staticmethod
    def make_policy_function(p_xi_eta_gamma,
                             projection_to_x_y_info,
                             list_for_integrate,
                             new_omega_list,
                             list_of_integrals_z_di,
                             Rects,
                             Grids):
        return PolicyFunction(p_xi_eta_gamma=p_xi_eta_gamma,
                              projection_to_x_y_info=projection_to_x_y_info,
                              list_for_integrate=list_for_integrate,
                              new_omega_list=new_omega_list,
                              list_of_integrals_z_di=list_of_integrals_z_di,
                              Rects=Rects,
                              Grids=Grids)


class PointwiseApprox(ModelWrapper):
    def __init__(self,
                 input_shape: List[int],
                 output_shape: List[int],
                 cache_dir: str,
                 view_name: str,
                 list_of_possible_states: List[str] = ['train', 'eval']
                 ):
        ModelWrapper.__init__(self,
                              input_shape,
                              output_shape,
                              cache_dir,
                              view_name,
                              list_of_possible_states
                              )

        def call(self):
            pass

        def save_to(self):
            pass

        def load_from(self):
            pass

        def to_train(self):
            pass

        def to_eval(self):
            pass
