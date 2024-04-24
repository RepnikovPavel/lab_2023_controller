import numpy as np

from RulesMaker.func_det import make_func
from RulesMaker.func_det import Distrib
from RulesMaker.rules_class import Rules
import os
from Simulation.units import make_f_1_f_2_f_3
from config_supports.config_fucntions import *

tensorboard_path = r'/home/user/work/penv/bin/tensorboard'


task_dir = r'/home/user/lab_2023_controller_data'


if not os.path.exists(task_dir):
    os.makedirs(task_dir)
plots_path = os.path.join(task_dir, 'plots')
plots_simplex_descent_path = os.path.join(task_dir, 'simplex_desc')

if not os.path.exists(plots_path):
    os.makedirs(plots_path)

if not os.path.exists(plots_simplex_descent_path):
    os.makedirs(plots_simplex_descent_path)
    

p_0_for_mixture_path = os.path.join(task_dir, 'p0_for_mixture.txt')
rules_cache_dir = os.path.join(task_dir,'rules_cache_dir')
Phi_cache_dir = os.path.join(task_dir,'Phi_cache_dir')
Phi_adj_matrix = os.path.join(task_dir,'Phi_adj_m.txt')
Phi_vector_representation = os.path.join(task_dir, 'Phi_vectors.txt')
Phi_loss1 = os.path.join(task_dir, 'Phi_qualityv1.txt')
Phi_descent_best_p_path = os.path.join(task_dir, 'Phi_descent_best_p.txt')


random_decent_best_filepath = os.path.join(task_dir, 'random_decs_best.txt')
inc_random_dec_file_path = os.path.join(task_dir, 'inc_random_decs_best.txt')
sql_files_path = './'
z_storage_dir = os.path.join(task_dir, 'z_storage')
integrator_dir = os.path.join(task_dir, 'integrator_storage')

p_storage_base_dir = os.path.join(task_dir, 'p_storage')
p_sim_results_base_dir = os.path.join(task_dir, 'sim_results')

descent_html = os.path.join(task_dir, 'desc_html.html')

debug_min_search_base_path = os.path.join(task_dir, 'plotting_find_min_of_L')

graph_with_projections_to_Phi_base_dir = os.path.join(task_dir, 'projections_graph')

SQLite3_init_script = os.path.join(sql_files_path, 'init_db.sql')
SQLite3_alg_db_path = os.path.join(task_dir, 'alg.db')
models_nav_table_name = 'models'

models = {
    'pointwise_approx':
        {
            'model_name': 'p_a_v1',
            'dir': os.path.join(task_dir, 'p_a_v1')
        }
}
models_gen = {
    'pointwise_approx':
        {
            'dir': os.path.join(task_dir, 'mg_p_a_v1')
        }
}

incomplete_rules_cache_dir = os.path.join(task_dir,'inc_rules')
incomplete_rules_integrator_dir = os.path.join(task_dir, 'inc_integrator_storage')
inc_mg_cache = os.path.join(task_dir, 'inc_rules_mg_cache')
inc_p0 = os.path.join(task_dir,'inc_p0.txt')



model_generator_cache_dir = os.path.join(task_dir, 'model_gen_cache')

phys_params = {
    'm': 0.050,  # in si  # масса стержня 50 грамм
    'L': 0.25,  # in si  # длина стержня 2*L= 50 см
    'g': 9.81,  # in si
    'M': 1.0,  # in si
    'b_const': 0.02040816326531  # (3 * m) / (7 * (M + m))
}

plot_trajectories_params = {
    'x_1_range': [-0.9, 0.9],
    'x_2_range': [-0.9, 0.9],
    'n_x_for_sim': 20,  # число стартовых точек для симуляции
    'n_y_for_sim': 20,  # число стартовых точек для симуляции
}
descent_start_points_params = {
    'x_1_range': [-0.9, 0.9],
    'x_2_range': [-0.9, 0.9],
    'n_x_for_sim': 5,  # число стартовых точек для симуляции
    'n_y_for_sim': 5,  # число стартовых точек для симуляции
}

# plot_trajectories_params = {
#     'x_1_range': [-0.9, 0.9],
#     'x_2_range': [-0.9, 0.9],
#     'n_x_for_sim': 8,  # число стартовых точек для симуляции
#     'n_y_for_sim': 8,  # число стартовых точек для симуляции
# }

phys_sim_params = {
    't_0': 0,  # in si начало одной симуляции
    't_end': 3,  # in si конец одной симуляции
    'tau': 0.01,  # in si шаг по времени в симуляции
    'v_0': 0.0,  # in si
    'y_0': 0.0  # in si,
}

# fuzzy_inf_params = {
#     'num_of_points_in_fuzzy_inf': 100,
#     'th_max': 0.2617993877991494,  # in si диапазон параметра
#     # 'F_max': 10,  # in si диапазон параметра
#     'F_max': 10,  # in si диапазон параметра
#     'omega_max': 3  # in si диапазон параметра
# }
fuzzy_inf_params = {
    'num_of_points_in_fuzzy_inf': 100,
    'th_max': 0.2617993877991494,  # in si диапазон параметра
    'F_max': 20,  # in si диапазон параметра
    # 'F_max': 10,  # in si диапазон параметра
    'omega_max': 3  # in si диапазон параметра
}

theta_NB = Distrib(make_func("triangle_m_f", [-10.0, -1.0, -2 / 3]), supp_of_func=(-1.0, -2 / 3))
theta_NM = Distrib(make_func("triangle_m_f", [-1.0, -2 / 3, -1 / 3]), supp_of_func=(-1.0, -1 / 3))
theta_NS = Distrib(make_func("triangle_m_f", [-2 / 3, -1 / 3, 0.0]), supp_of_func=(-2 / 3, 0.0))
theta_ZE = Distrib(make_func("triangle_m_f", [-1 / 3, 0, 1 / 3]), supp_of_func=(-1 / 3, 1 / 3))
theta_PS = Distrib(make_func("triangle_m_f", [0, 1 / 3, 2 / 3]), supp_of_func=(0, 2 / 3))
theta_PM = Distrib(make_func("triangle_m_f", [1 / 3, 2 / 3, 1.0]), supp_of_func=(1 / 3, 1.0))
theta_PB = Distrib(make_func("triangle_m_f", [2 / 3, 1.0, 10.0]), supp_of_func=(2 / 3, 1.0))



omega_NB = Distrib(make_func("triangle_m_f", [-10.0, -1.0, -2 / 3]), supp_of_func=(-1.0, -2 / 3))
omega_NM = Distrib(make_func("triangle_m_f", [-1.0, -2 / 3, -1 / 3]), supp_of_func=(-1.0, -1 / 3))
omega_NS = Distrib(make_func("triangle_m_f", [-2 / 3, -1 / 3, 0.0]), supp_of_func=(-2 / 3, 0.0))
omega_ZE = Distrib(make_func("triangle_m_f", [-1 / 3, 0, 1 / 3]), supp_of_func=(-1 / 3, 1 / 3))
omega_PS = Distrib(make_func("triangle_m_f", [0, 1 / 3, 2 / 3]), supp_of_func=(0, 2 / 3))
omega_PM = Distrib(make_func("triangle_m_f", [1 / 3, 2 / 3, 1.0]), supp_of_func=(1 / 3, 1.0))
omega_PB = Distrib(make_func("triangle_m_f", [2 / 3, 1.0, 10.0]), supp_of_func=(2 / 3, 1.0))


f_NB = Distrib(make_func("triangle_m_f", [-10.0, -1.0, -2 / 3]), supp_of_func=(-1.0, -2 / 3))
f_NM = Distrib(make_func("triangle_m_f", [-1.0, -2 / 3, -1 / 3]), supp_of_func=(-1.0, -1 / 3))
f_NS = Distrib(make_func("triangle_m_f", [-2 / 3, -1 / 3, 0.0]), supp_of_func=(-2 / 3, 0.0))
f_ZE = Distrib(make_func("triangle_m_f", [-1 / 3, 0, 1 / 3]), supp_of_func=(-1 / 3, 1 / 3))
f_PS = Distrib(make_func("triangle_m_f", [0, 1 / 3, 2 / 3]), supp_of_func=(0, 2 / 3))
f_PM = Distrib(make_func("triangle_m_f", [1 / 3, 2 / 3, 1.0]), supp_of_func=(1 / 3, 1.0))
f_PB = Distrib(make_func("triangle_m_f", [2 / 3, 1.0, 10.0]), supp_of_func=(2 / 3, 1.0))


rules_dict = {
    1: {
        'IF': [theta_PB, omega_NB],
        'THEN': [f_ZE]
    },
    2: {
        'IF': [theta_PB, omega_NM],
        'THEN': [f_PS]
    },
    3: {
        'IF': [theta_PB, omega_NS],
        'THEN': [f_PM]
    },
    4: {
        'IF': [theta_PB, omega_ZE],
        'THEN': [f_PB]
    },
    5: {
        'IF': [theta_PB, omega_PS],
        'THEN': [f_PB]
    },
    6: {
        'IF': [theta_PB, omega_PM],
        'THEN': [f_PB]
    },
    7: {
        'IF': [theta_PB, omega_PB],
        'THEN': [f_PB]
    },
    8: {
        'IF': [theta_PM, omega_NB],
        'THEN': [f_NS]
    },
    9: {
        'IF': [theta_PM, omega_NM],
        'THEN': [f_ZE]
    },
    10: {
        'IF': [theta_PM, omega_NS],
        'THEN': [f_PS]
    },
    11: {
        'IF': [theta_PM, omega_ZE],
        'THEN': [f_PM]
    },
    12: {
        'IF': [theta_PM, omega_PS],
        'THEN': [f_PB]
    },
    13: {
        'IF': [theta_PM, omega_PM],
        'THEN': [f_PB]
    },
    14: {
        'IF': [theta_PM, omega_PB],
        'THEN': [f_PB]
    },
    15: {
        'IF': [theta_PS, omega_NB],
        'THEN': [f_NM]
    },
    16: {
        'IF': [theta_PS, omega_NM],
        'THEN': [f_NS]
    },
    17: {
        'IF': [theta_PS, omega_NS],
        'THEN': [f_ZE]
    },
    18: {
        'IF': [theta_PS, omega_ZE],
        'THEN': [f_PS]
    },
    19: {
        'IF': [theta_PS, omega_PS],
        'THEN': [f_PM]
    },
    20: {
        'IF': [theta_PS, omega_PM],
        'THEN': [f_PB]
    },
    21: {
        'IF': [theta_PS, omega_PB],
        'THEN': [f_PB]
    },
    22: {
        'IF': [theta_ZE, omega_NB],
        'THEN': [f_NB]
    },
    23: {
        'IF': [theta_ZE, omega_NM],
        'THEN': [f_NM]
    },
    24: {
        'IF': [theta_ZE, omega_NS],
        'THEN': [f_NS]
    },
    25: {
        'IF': [theta_ZE, omega_ZE],
        'THEN': [f_ZE]
    },
    26: {
        'IF': [theta_ZE, omega_PS],
        'THEN': [f_PS]
    },
    27: {
        'IF': [theta_ZE, omega_PM],
        'THEN': [f_PM]
    },
    28: {
        'IF': [theta_ZE, omega_PB],
        'THEN': [f_PB]
    },
    29: {
        'IF': [theta_NS, omega_NB],
        'THEN': [f_NB]
    },
    30: {
        'IF': [theta_NS, omega_NM],
        'THEN': [f_NB]
    },
    31: {
        'IF': [theta_NS, omega_NS],
        'THEN': [f_NM]
    },
    32: {
        'IF': [theta_NS, omega_ZE],
        'THEN': [f_NS]
    },
    33: {
        'IF': [theta_NS, omega_PS],
        'THEN': [f_ZE]
    },
    34: {
        'IF': [theta_NS, omega_PM],
        'THEN': [f_PS]
    },
    35: {
        'IF': [theta_NS, omega_PB],
        'THEN': [f_PM]
    },
    36: {
        'IF': [theta_NM, omega_NB],
        'THEN': [f_NB]
    },
    37: {
        'IF': [theta_NM, omega_NM],
        'THEN': [f_NB]
    },
    38: {
        'IF': [theta_NM, omega_NS],
        'THEN': [f_NB]
    },
    39: {
        'IF': [theta_NM, omega_ZE],
        'THEN': [f_NM]
    },
    40: {
        'IF': [theta_NM, omega_PS],
        'THEN': [f_NS]
    },
    41: {
        'IF': [theta_NM, omega_PM],
        'THEN': [f_ZE]
    },
    42: {
        'IF': [theta_NM, omega_PB],
        'THEN': [f_PS]
    },
    43: {
        'IF': [theta_NB, omega_NB],
        'THEN': [f_NB]
    },
    44: {
        'IF': [theta_NB, omega_NM],
        'THEN': [f_NB]
    },
    45: {
        'IF': [theta_NB, omega_NS],
        'THEN': [f_NB]
    },
    46: {
        'IF': [theta_NB, omega_ZE],
        'THEN': [f_NB]
    },
    47: {
        'IF': [theta_NB, omega_PS],
        'THEN': [f_NM]
    },
    48: {
        'IF': [theta_NB, omega_PM],
        'THEN': [f_NS]
    },
    49: {
        'IF': [theta_NB, omega_PB],
        'THEN': [f_ZE]
    }
}
theta_inc = Distrib(make_func("triangle_m_f", [-1/3, 0.0, 1/3]), supp_of_func=(-1/3, 1/3), num_of_segments=20)
# omega_inc_NB = Distrib(make_func("triangle_m_f", [0.0, 1/3, 1.0]), supp_of_func=(0.0, 1.0), num_of_segments=20)
f_inc = Distrib(make_func("triangle_m_f", [-1.0, 0.0, 1.0]), supp_of_func=(-1.0, 1.0), num_of_segments=20)


theta_00 = Distrib(make_func("triangle_m_f", [0/1-1/3,0/1,0/1+1/3]), supp_of_func=(0/1 -1/3,0/1+1/3))
theta_p16 = Distrib(make_func("triangle_m_f", [1/6-1/3,1/6,1/6+1/3]), supp_of_func=(1/6-1/3,1/6+1/3))
theta_p13 = Distrib(make_func("triangle_m_f", [1/3-1/3,1/3,1/3+1/3]), supp_of_func=(1/3-1/3,1/3+1/3))
theta_p12 = Distrib(make_func("triangle_m_f", [1/2-1/3,1/2,1/2+1/3]), supp_of_func=(1/2-1/3,1/2+1/3))

theta_m16 = Distrib(make_func("triangle_m_f", [-1/6-1/3,-1/6,-1/6+1/3]), supp_of_func=(-1/6-1/3,-1/6+1/3))
theta_m13 = Distrib(make_func("triangle_m_f", [-1/3-1/3,-1/3,-1/3+1/3]), supp_of_func=(-1/3-1/3,-1/3+1/3))
theta_m12 = Distrib(make_func("triangle_m_f", [-1/2-1/3,-1/2,-1/2+1/3]), supp_of_func=(-1/2-1/3,-1/2+1/3))

omega_00 = Distrib(make_func("triangle_m_f", [0/1-1/3,0/1,0/1+1/3]), supp_of_func=(0/1-1/3,0/1+1/3))
omega_p16 = Distrib(make_func("triangle_m_f", [1/6-1/3,1/6,1/6+1/3]), supp_of_func=(1/6-1/3,1/6+1/3))
omega_p13 = Distrib(make_func("triangle_m_f", [1/3-1/3,1/3,1/3+1/3]), supp_of_func=(1/3-1/3,1/3+1/3))
omega_p12 = Distrib(make_func("triangle_m_f", [1/2-1/3,1/2,1/2+1/3]), supp_of_func=(1/2-1/3,1/2+1/3))

omega_m16 = Distrib(make_func("triangle_m_f", [-1/6-1/3,-1/6,-1/6+1/3]), supp_of_func=(-1/6-1/3,-1/6+1/3))
omega_m13 = Distrib(make_func("triangle_m_f", [-1/3-1/3,-1/3,-1/3+1/3]), supp_of_func=(-1/3-1/3,-1/3+1/3))
omega_m12 = Distrib(make_func("triangle_m_f", [-1/2-1/3,-1/2,-1/2+1/3]), supp_of_func=(-1/2-1/3,-1/2+1/3))

incomplete_rules_dict = {
    1: {
        'IF': [theta_PB, omega_NB],
        'THEN': [f_ZE]
    },
    2: {
        'IF': [theta_PB, omega_NM],
        'THEN': [f_PS]
    },
    3: {
        'IF': [theta_PB, omega_NS],
        'THEN': [f_PM]
    },
    4: {
        'IF': [theta_PB, omega_ZE],
        'THEN': [f_PB]
    },
    5: {
        'IF': [theta_PB, omega_PS],
        'THEN': [f_PB]
    },
    6: {
        'IF': [theta_PB, omega_PM],
        'THEN': [f_PB]
    },
    7: {
        'IF': [theta_PB, omega_PB],
        'THEN': [f_PB]
    },
    8: {
        'IF': [theta_PM, omega_NB],
        'THEN': [f_NS]
    },
    9: {
        'IF': [theta_PM, omega_NM],
        'THEN': [f_ZE]
    },
    10: {
        'IF': [theta_PM, omega_NS],
        'THEN': [f_PS]
    },
    11: {
        'IF': [theta_PM, omega_ZE],
        'THEN': [f_PM]
    },
    12: {
        'IF': [theta_PM, omega_PS],
        'THEN': [f_PB]
    },
    13: {
        'IF': [theta_PM, omega_PM],
        'THEN': [f_PB]
    },
    14: {
        'IF': [theta_PM, omega_PB],
        'THEN': [f_PB]
    },
    36: {
        'IF': [theta_NM, omega_NB],
        'THEN': [f_NB]
    },
    37: {
        'IF': [theta_NM, omega_NM],
        'THEN': [f_NB]
    },
    38: {
        'IF': [theta_NM, omega_NS],
        'THEN': [f_NB]
    },
    39: {
        'IF': [theta_NM, omega_ZE],
        'THEN': [f_NM]
    },
    40: {
        'IF': [theta_NM, omega_PS],
        'THEN': [f_NS]
    },
    41: {
        'IF': [theta_NM, omega_PM],
        'THEN': [f_ZE]
    },
    42: {
        'IF': [theta_NM, omega_PB],
        'THEN': [f_PS]
    },
    43: {
        'IF': [theta_NB, omega_NB],
        'THEN': [f_NB]
    },
    44: {
        'IF': [theta_NB, omega_NM],
        'THEN': [f_NB]
    },
    45: {
        'IF': [theta_NB, omega_NS],
        'THEN': [f_NB]
    },
    46: {
        'IF': [theta_NB, omega_ZE],
        'THEN': [f_NB]
    },
    47: {
        'IF': [theta_NB, omega_PS],
        'THEN': [f_NM]
    },
    48: {
        'IF': [theta_NB, omega_PM],
        'THEN': [f_NS]
    },
    49: {
        'IF': [theta_NB, omega_PB],
        'THEN': [f_ZE]
    },
    # # fictitious rule

    50: {
        'IF': [theta_00, omega_m13],
        'THEN': [f_inc]
    },
    51: {
        'IF': [theta_m13, omega_00],
        'THEN': [f_inc]
    },
    52: {
        'IF': [theta_00, omega_p13],
        'THEN': [f_inc]
    },
    53: {
        'IF': [theta_p13, omega_00],
        'THEN': [f_inc]
    },

    54: {
        'IF': [theta_00, omega_m12],
        'THEN': [f_inc]
    },
    55: {
        'IF': [theta_m12, omega_00],
        'THEN': [f_inc]
    },
    56: {
        'IF': [theta_00, omega_p12],
        'THEN': [f_inc]
    },
    57: {
        'IF': [theta_p12, omega_00],
        'THEN': [f_inc]
    },


    58: {
        'IF': [theta_m13, omega_m13],
        'THEN': [f_inc]
    },
    59: {
        'IF': [theta_m13, omega_p13],
        'THEN': [f_inc]
    },
    60: {
        'IF': [theta_p13, omega_p13],
        'THEN': [f_inc]
    },
    61: {
        'IF': [theta_p13, omega_m13],
        'THEN': [f_inc]
    }
}
# print('len of rules='+str(len(incomplete_rules_dict)))
# for simulation
# make empty if gradiend descent
# for nosifying
# mask_for_rules_for_noise = np.array([len(incomplete_rules_dict.keys())-1], dtype=np.uint32)
# mask_for_rules_for_noise = np.array([
#     last_i,
#     last_i-1,
#     last_i-2,
#     last_i-3,
#     last_i-4,
#     last_i-5,
#     last_i-6

#                                      ], dtype=np.uint32)

# inc_rules_area_of_xi_eta = [
#     [-1.0, 1.0],
#     [-1.0, 1.0],
#     [-1.0, 1.0]
# ]


rules = Rules(rules_dict=rules_dict, dimension=3, xi_dim_=2, eta_dim_=1)
incomplete_rules = Rules(rules_dict = incomplete_rules_dict, dimension=3,xi_dim_=2,eta_dim_=1)
th_max = fuzzy_inf_params['th_max']  # in si
F_max = fuzzy_inf_params['F_max']  # in si
omega_max = fuzzy_inf_params['omega_max']  # in si
theta_range = np.asarray([-th_max, th_max])
omega_range = np.asarray([-omega_max, omega_max])
F_range = np.asarray([-F_max, F_max])
theta_v_range = np.asarray([-1.0, 1.0])
omega_v_range = np.asarray([-1.0, 1.0])
F_v_range = np.asarray([-1.0, 1.0])

translators_units_of_measurement = make_f_1_f_2_f_3(
    theta_range, theta_v_range, omega_range, omega_v_range, F_range, F_v_range)
# if test without rules
# areas_with_action = get_areas_with_actions(translators_units_of_measurement,incomplete_rules_dict,mask_of_disable_rule=[50,51,52,53,54,55,56,57,58,59,60,61])
# if train rules
areas_with_action = get_areas_with_actions(translators_units_of_measurement,incomplete_rules_dict,mask_of_disable_rule=[])
