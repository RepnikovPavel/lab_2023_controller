import os.path
import typing
from typing import Tuple
from typing import List
from typing import Any
from typing import Dict
from typing import Type
from copy import deepcopy

from FileSystem.general_purpose_functions import *
from SQLHelpers.query_helpers import *
import sqlite3

import config

from abc import abstractmethod
from abc import ABC


class ModelMathInfo:
    __input_shape: List[int]
    __output_shape: List[int]

    def __init__(self, input_shape: List[int], output_shape: List[int]):
        self.__input_shape = input_shape
        self.__output_shape = output_shape

    def get_input_shape(self):
        return deepcopy(self.__input_shape)

    def get_output_shape(self):
        return deepcopy(self.__output_shape)


def are_models_compatible_in_shapes(math_info1: Type[ModelMathInfo], math_info2: Type[ModelMathInfo]):
    return math_info1.get_output_shape() == math_info2.get_input_shape()


class ModelState:
    __list_of_states: List[str] = ['train', 'eval']
    __current_state: str = None

    def __init__(self, list_of_possible_states: List[str]):
        if len(list_of_possible_states) != 0:
            self.__list_of_states = deepcopy(list_of_possible_states)

    def __is_the_behavior_of_the_model_defined(self):
        if self.__current_state is None:
            return False
        else:
            return True

    def get_possible_model_states(self) -> List[str]:
        return deepcopy(self.__list_of_states)

    def get_current_state(self):
        return deepcopy(self.__current_state)

    def set_model_state(self, state: str):
        if state in self.__list_of_states:
            self.__current_state = state
        else:
            print('state "{}" does not exist'.format(state))


class ModelStorage:
    model_name: str
    base_dir: str
    torch_cache_dir: str
    torch_shared_data_dir: str
    torch_seeds_dir: str

    def __init__(self, dir_: str, view_name: str):
        '''
        reads config file
        {
            'model_name': 'p_a_v1',
            'dir': os.path.join(task_dir, 'p_a_v1')
        }

        '''
        self.base_dir = dir_
        self.model_name = view_name
        self.torch_cache_dir = 'cache'
        self.torch_seeds_dir = 'seeds'
        self.torch_shared_data_dir = 'shared'

        mkdir_if_not_exists(self.base_dir)
        mkdir_if_not_exists(os.path.join(self.base_dir, self.torch_cache_dir))
        mkdir_if_not_exists(os.path.join(self.base_dir, self.torch_seeds_dir))
        mkdir_if_not_exists(os.path.join(self.base_dir, self.torch_shared_data_dir))

    def get_information_for_registration(self):
        out = {
            'dir': self.base_dir,
            'view_name': self.model_name
        }
        return out


class ModelWrapper(ABC):
    math_info: ModelMathInfo
    state: ModelState
    storage: ModelStorage

    def __init__(self,
                 input_shape: List[int],
                 output_shape: List[int],
                 cache_dir: str,
                 view_name: str,
                 list_of_possible_states: List[str] = ['train', 'eval']
                 ):
        self.math_info = ModelMathInfo(input_shape=input_shape, output_shape=output_shape)
        self.state = ModelState(list_of_possible_states=list_of_possible_states)
        self.storage = ModelStorage(dir_=cache_dir, view_name=view_name)

    @abstractmethod
    def call(self):
        pass

    @abstractmethod
    def save_to(self):
        pass

    @abstractmethod
    def load_from(self):
        pass

    @abstractmethod
    def to_train(self):
        pass

    @abstractmethod
    def to_eval(self):
        pass
