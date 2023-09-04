import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network
import pprint
from FileSystem.general_purpose_functions import *
from FileSystem.storage import SimResultsManager
import config


from SearchAlg.genetic_alg_general_functions import Loss, plot_hist_of_list, GetPhysicalLoss

import scipy

if __name__ == '__main__':

    # join sim results
    results_ = {}
    dirs = os.listdir(config.p_sim_results_base_dir)
    for i in range(len(dirs)):
        if not dirs[i].isnumeric():
            print(dirs[i])
            continue
        sim_storage = SimResultsManager(base_dir_=config.p_sim_results_base_dir, index_of_exe=dirs[i],
                                        clear_cache=False)
        results_i = sim_storage.get_sim_results()
        for j in range(len(results_i)):
            result = results_i[j]
            key = result['source_path']
            results_.update({key: result['results']})

    print('start comp loss')
    all_names = []
    all_results = []
    all_values = []
    T = config.phys_sim_params['t_end']
    for key in results_:
        all_names.append(key)
        all_results.append(results_[key])
        res_= results_[key]
        loss = GetPhysicalLoss(sim_results_=res_, T=T)
        all_values.append(loss)
    argmin = np.argmin(all_values)
    print(all_names[argmin])
    print(all_results[argmin])
    print(all_values[argmin])


    plot_hist_of_list(all_values)







