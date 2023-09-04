from Alg.solving_algorithm import ModelGenerator
from FileSystem.storage import PStorage
from CustomModels.my_models import DistrMaker
from CustomModels.my_models import Integrator
import config
import sys
from general_purpose_functions import time_mesuament

if __name__ == '__main__':
    index_of_exe = sys.argv[1]
    # index_of_exe ='test'
    N = 200
    print('exe {} approximate execution time {} m'.format(index_of_exe, 16 * N * 40 / 60 / 7))
    timer = time_mesuament.Timer()
    timer.start()

    shared_data = ModelGenerator(rules=config.rules,
                                 cache_dir=config.models_gen['pointwise_approx']['dir'],
                                 clear_cache=False).shared_data
    shared_integration_supports = Integrator(dir_=config.integrator_dir,
                                             shared_data=shared_data,
                                             clear_cache=False).shared_integration_supports
    gen_config = config.models_gen['pointwise_approx']

    pstorage = PStorage(dir_=config.p_storage_base_dir,
                        index_of_exe=index_of_exe,
                        clear_cache=False)

    mg = ModelGenerator(rules=config.rules,
                        cache_dir=gen_config['dir'],
                        clear_cache=False)

    for i in range(N):
        zlist = mg.CreateModelFromScratch()
        p_xi_eta_gamma = DistrMaker.make_distrib4D(x_={
            'z_list': zlist,
            'np_omega': shared_data['np_omega']
        })
        pstorage.append(p_xi_eta_gamma)

    timer.stop()
    print('the process {} has completed its work with time {} m'.format(index_of_exe, timer.get_execution_time() / 60))
