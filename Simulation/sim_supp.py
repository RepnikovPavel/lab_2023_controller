import Simulation.solver as solver
import numpy as np
from copy import deepcopy as deepcopy
from numba import jit

def make_psi(policy_func,
             translators_units_of_measurement):
    from_v_in_volt_to_v_in_si = deepcopy(translators_units_of_measurement['from_v_in_volt_to_v_in_si'])
    from_th_in_si_to_th_in_volt = deepcopy(translators_units_of_measurement['from_th_in_si_to_th_in_volt'])
    from_omega_in_si_to_omega_in_volt = deepcopy(translators_units_of_measurement['from_omega_in_si_to_omega_in_volt'])
    def psi(theta: float, omega: float) -> float:
        return from_v_in_volt_to_v_in_si(policy_func(
            x_1=from_th_in_si_to_th_in_volt(theta),
            x_2=from_omega_in_si_to_omega_in_volt(omega)
        ))

    return psi


def make_simulation_for_one_policy_function(psi, condition_of_break, phys_sim_params, object_params,use_an_early_stop=True,action_check=True):
    t_0 = phys_sim_params['t_0']
    t_end = phys_sim_params['t_end']
    tau = phys_sim_params['tau']

    L = object_params['L']
    g = object_params['g']
    b_const = object_params['b_const']
    m = object_params['m']
    M = object_params['M']

    def simulate_car(th_0, omega_0, y_0, v_0):
        return solver.solve_ode(
            tau=tau, t_0=t_0, t_end=t_end,
            Y_0=np.asarray([th_0, omega_0, y_0, v_0]),
            condition_of_break=condition_of_break,
            L=L, g=g, b=b_const, m=m, M=M,
            psi=psi,
            use_an_early_stop=use_an_early_stop,action_check=action_check)

    return simulate_car
