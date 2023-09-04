from typing import List
from typing import Tuple

import numpy as np
from scipy.integrate import quad
from copy import deepcopy

def triangle_m_f(x: float, l_point: float, m_point: float, r_point: float) -> float:
    if x <= l_point:
        return 0.0
    elif x >= r_point:
        return 0.0
    elif l_point < x <= m_point:
        return 1 / (m_point - l_point) * x + l_point / (l_point - m_point)
    elif m_point < x < r_point:
        return 1 / (m_point - r_point) * x + r_point / (r_point - m_point)


def make_func(type_of_func: str, func_params: list):
    """
    type_of_func: "triangle_m_f" only supported
    """
    if type_of_func == "triangle_m_f":
        def m_f(x: float) -> float:
            return triangle_m_f(x, func_params[0], func_params[1], func_params[2])

        return m_f
    else:
        print("smth went wrong in make_func function")
        raise SystemExit


class Distrib:
    support: Tuple[float, float]
    distrib = None
    grid: np.array
    num_of_segments: int

    def __init__(self, func, supp_of_func: Tuple[float, float], num_of_segments=5):
        integral = quad(func, supp_of_func[0],
                        supp_of_func[1])[0]

        def distr(x: float):
            return 1 / integral * func(x)

        self.distrib = distr
        self.support = supp_of_func
        self.num_of_segments = num_of_segments
        self.grid = np.linspace(start=supp_of_func[0], stop=supp_of_func[1], num=self.num_of_segments+1)

    def __call__(self, x: float):
        return self.distrib(x)


