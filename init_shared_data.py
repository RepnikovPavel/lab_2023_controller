from FileSystem.general_purpose_functions import *
from Alg.solving_algorithm import ModelGenerator
from FileSystem.storage import PStorage
from CustomModels.my_models import DistrMaker
from CustomModels.my_models import Integrator
import config
from general import plot_policy_function, plot_trajectories
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function
import sys
from general_purpose_functions import time_mesuament

if __name__ =='__main__':
    shared_data = ModelGenerator(rules=config.rules,
                                 cache_dir=config.models_gen['pointwise_approx']['dir'],
                                 clear_cache=True).shared_data
    shared_integration_supports = Integrator(dir_=config.integrator_dir,
                                             shared_data=shared_data,
                                             clear_cache=True).shared_integration_supports