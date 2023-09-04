import numpy as np
from numba import jit
from typing import Tuple,Dict,List,Callable


def make_f_1_f_2_f_3(th_range:np.array, th_v_range:np.array, omega_range:np.array,
                     omega_v_range:np.array, v_range:np.array, v_v_range:np.array)->\
        Dict[str,Callable]:

    @jit(nopython=True, cache=True)
    def from_th_in_si_to_th_in_volt(th: float) -> float:
        # return th in volt
        return th_v_range[0] + (th_v_range[1] - th_v_range[0]) / (th_range[1] - th_range[0]) * (
                th - th_range[0])

    @jit(nopython=True, cache=True)
    def from_omega_in_si_to_omega_in_volt(omega: float) -> float:
        # return omega in volt
        return omega_v_range[0] + (omega_v_range[1] - omega_v_range[0]) / (omega_range[1] - omega_range[0]) * (
                omega - omega_range[0])

    @jit(nopython=True, cache=True)
    def from_v_in_si_to_v_in_volt(v: float) -> float:
        # return v in volt
        return v_v_range[0] + (v_v_range[1] - v_v_range[0]) / (v_range[1] - v_range[0]) * (
                v - v_range[0])

    @jit(nopython=True, cache=True)
    def from_v_in_volt_to_v_in_si(v_in_volt: float) -> float:
        # input v in volt
        # return v in m/s
        return v_range[0] + (v_range[1] - v_range[0]) / (v_v_range[1] - v_v_range[0]) * (v_in_volt - v_v_range[0])

    @jit(nopython=True, cache=True)
    def from_omega_in_volt_to_omega_in_si(omega_in_volt: float) -> float:
        # input v in volt
        # return v in m/s
        return omega_range[0] + (omega_range[1] - omega_range[0]) / (omega_v_range[1] - omega_v_range[0]) * (omega_in_volt - omega_v_range[0])

    @jit(nopython=True, cache=True)
    def from_th_in_volt_to_th_in_si(th_in_volt: float) -> float:
        # input v in volt
        # return v in m/s
        return th_range[0] + (th_range[1] - th_range[0]) / (th_v_range[1] - th_v_range[0]) * (th_in_volt - th_v_range[0])

    translators_units_of_measurement = {
        'from_th_in_si_to_th_in_volt': from_th_in_si_to_th_in_volt,
        'from_omega_in_si_to_omega_in_volt': from_omega_in_si_to_omega_in_volt,
        'from_v_in_si_to_v_in_volt': from_v_in_si_to_v_in_volt,
        'from_v_in_volt_to_v_in_si': from_v_in_volt_to_v_in_si,
        'from_omega_in_volt_to_omega_in_si': from_omega_in_volt_to_omega_in_si,
        'from_th_in_volt_to_th_in_si': from_th_in_volt_to_th_in_si
    }

    return translators_units_of_measurement