from typing import Dict
from typing import List
from copy import deepcopy
import numpy as np
from RulesMaker.func_det import Distrib

class Rules:
    rules_dict: Dict[int, Dict[str, List[Distrib]]]
    n_r: int
    dimension: int
    xi_dim: int
    eta_dim: int
    def __init__(self, rules_dict: Dict[int, Dict[str, List[Distrib]]],dimension:int,xi_dim_:int, eta_dim_:int):
        self.rules_dict = deepcopy(rules_dict)
        self.n_r = len(self.rules_dict.keys())
        self.dimension = dimension
        self.xi_dim =xi_dim_
        self.eta_dim = eta_dim_
    def get_ordered_keys(self):
        copy_of_keys = list(self.rules_dict.keys())
        copy_of_keys.sort()
        return copy_of_keys
