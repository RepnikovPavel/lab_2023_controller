import copy

import numba
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import r_regression
from general import plot_vec, get_averaged_arr, calc_a_vec, WMA
from numba import jit
import numba as nb
from typing import Tuple
import config


@jit(nopython=True, cache=True)
def F_vec(F_vec_buffer: np.ndarray, Y_vec_t_n: np.ndarray,
          L: float, g: float, f: float, b: float, m: float, M: float
          ) -> np.ndarray:
    theta = Y_vec_t_n[0]
    omega = Y_vec_t_n[1]
    y = Y_vec_t_n[2]
    v = Y_vec_t_n[3]

    F_vec_buffer[0] = omega
    F_vec_buffer[1] = 1 / (1 - b * np.square(np.cos(theta))) * (
            3 * g / (7 * L) * np.sin(theta) - b * f / (m * L) * np.cos(theta) - b * np.sin(theta) * np.cos(
        theta) * np.square(omega))
    F_vec_buffer[2] = v
    F_vec_buffer[3] = 1 / (1 - b * np.square(np.cos(theta))) * (
            f / (M + m) - b * g * np.sin(theta) * np.cos(theta) + 7 / 3 * np.square(omega) * b * L * np.sin(theta))
    return F_vec_buffer


@jit(nopython=True, cache=True)
def next_point(tau: float, Y_vec_t_n: np.ndarray,
               F_vec_buffer: np.ndarray,
               L: float, g: float, f: float, b: float, m: float, M: float
               ) -> np.ndarray:
    """
    :return Y_vec(t_{n+1})
    """

    return Y_vec_t_n + tau * F_vec(F_vec_buffer, Y_vec_t_n, L, g, f, b, m, M)


@jit(nopython=True, cache=True)
def is_the_condition_met(solution_t, condition_of_break):
    if not ((condition_of_break[0][0] <= solution_t[0] <= condition_of_break[0][1]) and
            (condition_of_break[1][0] <= solution_t[1] <= condition_of_break[1][1]) and
            (condition_of_break[2][0] <= solution_t[2] <= condition_of_break[2][1]) and
            (condition_of_break[3][0] <= solution_t[3] <= condition_of_break[3][1])):
        return False
    else:
        return True

@jit(nopython=True, cache=False)
def do_alg_need_to_do_action(areas_with_actions_si: np.array, theta_si: float, omega_si: float) -> bool:
    for i in range(len(areas_with_actions_si)):
        x_area = areas_with_actions_si[i][0]
        y_area = areas_with_actions_si[i][1]
        if (x_area[0] <= theta_si <= x_area[1]) and (y_area[0] <= omega_si <= y_area[1]):
            return True
    return False


def solve_ode(tau: float, t_0: float, t_end: float,
              Y_0: np.ndarray, condition_of_break: np.ndarray,
              L: float, g: float, b: float, m: float, M: float,
              psi, use_an_early_stop=True, action_check = True
              ) -> Tuple[int, np.array, float, np.array]:
    # TODO: добавить уловие остановки - ускорение перестает меняться
    '''
    code_of_sim == 1: выход из условия во время симуляции
    code_of_sim == 0: симуляция прошла без выхода из условия
    code_of_sim == 2: ранняя остановка ,т.к. ускорение тележки не меняется
    '''
    code_of_sim = 0

    num_of_points_in_solved_vec = int((t_end - t_0) / tau)

    solution = np.zeros(shape=(num_of_points_in_solved_vec, 4), dtype=np.float32)
    control_actions = np.zeros(shape=(num_of_points_in_solved_vec,), dtype=np.float32)
    solution[0][0] = Y_0[0]
    solution[0][1] = Y_0[1]
    solution[0][2] = Y_0[2]
    solution[0][3] = Y_0[3]

    areas_with_action = config.areas_with_action
    F_vec_buffer = np.zeros(shape=(4,), dtype=np.float32)

    time_of_simulation = 0
    # проход по всем точкам разбиения tau и вычисление вектора X_vec(t), решающего систему
    # так же проверяется условие на выход из разрешенной области.
    if not is_the_condition_met(solution_t=solution[0], condition_of_break=condition_of_break):
        code_of_sim = 1
        return code_of_sim, solution[:1], time_of_simulation, control_actions[:1]
    Forse_t_n = 0.0
    control_actions[0] = Forse_t_n
    solution[1] = next_point(tau, solution[0],
                             F_vec_buffer,
                             L, g, Forse_t_n, b, m, M
                             )

    time_window_size = 1.0
    window_size = int(np.floor(time_window_size / tau))
    # a_check_time_size = 0.1
    # a_w_size = int(np.floor(a_check_time_size/tau))

    for i in range(1, num_of_points_in_solved_vec - 1):
        if not is_the_condition_met(solution_t=solution[i], condition_of_break=condition_of_break):
            code_of_sim = 1
            time_of_simulation = i * tau
            return code_of_sim, solution[:i + 1], time_of_simulation, control_actions[:i + 1]
        # if use_an_early_stop:
        #     if i % window_size == 0:
        #         # y_true = (control_actions[i-window_size:i]).reshape(window_size,1)
        #         y_true = solution[i - window_size:i, 3].reshape(window_size, 1)
        #         y_1 = y_true[0]
        #         y_2 = y_true[1]
        #         x_1 = (i - window_size) * tau
        #         x_2 = i * tau
        #         dx = x_2 - x_1
        #         dy = y_2 - y_1
        #         a_ = dy / dx
        #         b_ = (-x_1 * dy + y_1 * dx) / dx
        #         delta_t = np.arange(start=0.0, stop=x_2 - x_1, step=tau, dtype=np.float32)
        #         y_pred = a_ * delta_t + b_
        #         r2 = r_regression(y_true, y_pred)
        #         if r2 > 0.9:
        #             code_of_sim = 2
        #             time_of_simulation = i * tau
        #             return code_of_sim, solution[:i + 1], time_of_simulation, control_actions[:i + 1]
        Forse_t_n = 0.0
        if not action_check:
            Forse_t_n = psi(solution[i - 1][0], solution[i - 1][1])
        elif do_alg_need_to_do_action(areas_with_action, solution[i - 1][0], solution[i - 1][1]):
            Forse_t_n = psi(solution[i - 1][0], solution[i - 1][1])
        else :
            Forse_t_n = 0.0
            # print('not action')
        if np.isnan(Forse_t_n):
            Forse_t_n = 0.0  # нейтральным управляющим действием в данном случае можно считать нулевую силу

        control_actions[i] = Forse_t_n
        solution[i + 1] = next_point(tau, solution[i],
                                     F_vec_buffer,
                                     L, g, Forse_t_n, b, m, M
                                     )

    time_of_simulation = np.abs(t_end - t_0)
    return code_of_sim, solution, time_of_simulation, control_actions
