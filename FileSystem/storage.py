import os.path
import typing
from typing import List, Set,Dict, Any
import torch
from FileSystem.general_purpose_functions import delete_all_data_from_directory

class ZStorage:
    dir_: str
    list_of_paths: List[str]
    uid: int

    def __init__(self, dir_ , clear_cache=False):
        if clear_cache:
            delete_all_data_from_directory(dir_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        self.list_of_paths = [os.path.join(dir_, el) for el in os.listdir(dir_)]
        self.uid = len(self.list_of_paths)
        self.dir_ = dir_

    def append(self, zlist):
        filename = os.path.join(self.dir_, 'zlist_{}.txt'.format(self.uid))
        torch.save(zlist, filename)
        self.list_of_paths.append(filename)
        self.uid += 1

    def laod_by_index(self, index):
        return torch.load(self.list_of_paths[index])

class PStorage:
    dir_: str
    exe_dir_:str
    list_of_paths: List[str]
    uid: int

    def __init__(self, dir_ ,index_of_exe, clear_cache=False):
        self.exe_dir_= os.path.join(dir_, index_of_exe)
        if clear_cache:
            delete_all_data_from_directory(self.exe_dir_)
        if not os.path.exists(self.exe_dir_):
            os.makedirs(self.exe_dir_)
        self.list_of_paths = [os.path.join(self.exe_dir_, el) for el in os.listdir(self.exe_dir_)]
        self.uid = len(self.list_of_paths)
        self.dir_ = dir_

    def append(self, distrib):
        filename = os.path.join(self.exe_dir_, 'distrib_{}.txt'.format(self.uid))
        torch.save(distrib, filename)
        self.list_of_paths.append(filename)
        self.uid += 1

    def get_number_of_elements(self):
        return len(self.list_of_paths)

    def load_by_index(self, index):
        return torch.load(self.list_of_paths[index])

    def get_path_by_index(self, index):
        index_str = str(index)
        for i in range(len(self.list_of_paths)):
            path_ = self.list_of_paths[i]
            k_ = 5
            reversed_index = ''
            while path_[-k_] != '_':
                reversed_index+=path_[-k_]
                k_+=1
            index_str_ = reversed_index[::-1]
            if index_str_ == index_str:
                return path_


class SimResultsManager:
    base_dir: str
    current_index = 0

    def __init__(self, base_dir_, index_of_exe,clear_cache = False):
        self.base_dir = os.path.join(base_dir_, index_of_exe)
        if clear_cache:
            delete_all_data_from_directory(self.base_dir)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def write_results(self, source_distrib_path, results):
        # if overwrite:
        #     self.already_tested_objects.update({source_distrib_path: results})
        # else:
        #     if source_distrib_path not in self.already_tested_objects:
        #         self.already_tested_objects.update({source_distrib_path: results})
        torch.save({
                    'results':results,
                    'source_path':source_distrib_path
                    },
            os.path.join(self.base_dir, 'results_{}.txt'.format(self.current_index)))
        self.current_index += 1

    def get_sim_results(self):
        filepaths = os.listdir(self.base_dir)
        results = []
        for i in range(len(filepaths)):
            results.append(torch.load(os.path.join(
                self.base_dir, filepaths[i]
            )))
        return results






