from typing import Tuple
from typing import Dict
from typing import List

import torch
import numpy as np
from RulesMaker.rules_class import Rules


class GridMaker:

    @staticmethod
    def make_omega_and_np_omega(rules: Rules):
        # прочитаем разбиение(рисочки) из словаря с правилами и сделаем отрезки из рисочек разбиения
        '''
        :return:
        torch_omega
        np_omega
        shape = (i, j, 2) - переменный размер по второму индексу i in 1,n_r j in 1,J(i)
        (rule_index, distrib_index, segment_index, 2)
        '''

        t_omega = [[] for i in range(rules.n_r)]
        np_omega = [[] for i in range(rules.n_r)]

        k = 0
        for rule_key in rules.get_ordered_keys():
            IF_distrs = rules.rules_dict[rule_key]['IF']
            THEN_distrs = rules.rules_dict[rule_key]['THEN']

            for distr in IF_distrs:
                t_segment_seq = []
                np_segment_seq = []
                for i in range(len(distr.grid) - 1):
                    a = distr.grid[i]
                    b = distr.grid[i + 1]
                    t_segment = torch.zeros(size=(2,), requires_grad=False)
                    t_segment[0] = a
                    t_segment[1] = b
                    np_segment = np.zeros(shape=(2,))
                    np_segment[0] = a
                    np_segment[1] = b
                    t_segment_seq.append(t_segment)
                    np_segment_seq.append(np_segment)
                t_omega[k].append(t_segment_seq)
                np_omega[k].append(np_segment_seq)

            for distr in THEN_distrs:
                t_segment_seq = []
                np_segment_seq = []
                for i in range(len(distr.grid) - 1):
                    a = distr.grid[i]
                    b = distr.grid[i + 1]
                    t_segment = torch.zeros(size=(2,), requires_grad=False)
                    t_segment[0] = a
                    t_segment[1] = b
                    np_segment = np.zeros(shape=(2,))
                    np_segment[0] = a
                    np_segment[1] = b
                    t_segment_seq.append(t_segment)
                    np_segment_seq.append(np_segment)
                t_omega[k].append(t_segment_seq)
                np_omega[k].append(np_segment_seq)
            k += 1

        for i in range(len(np_omega)):
            for j in range(len(np_omega[i])):
                np_omega[i][j] = np.asarray(np_omega[i][j])
        for i in range(len(np_omega)):
            np_omega[i] = np.asarray(np_omega[i], dtype="object")

        return t_omega, np_omega

