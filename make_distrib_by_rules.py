import torch
import config
from Alg.solving_algorithm import ModelGenerator
from CustomModels.my_models import Integrator, DistrMaker

if __name__ == '__main__':
    mg = ModelGenerator(rules=config.rules,
                                 cache_dir=config.rules_cache_dir,
                                 clear_cache=True)
    shared_data = mg.shared_data
    shared_integration_supports = Integrator(dir_=config.incomplete_rules_integrator_dir,
                                             shared_data=shared_data,
                                             clear_cache=True).shared_integration_supports
    # learn existing rules
    z_list_0 = mg.CreateModelFromScratch(is_approx=True,gr_steps = 2000,lr=0.01)
    p_0 = DistrMaker.make_distrib4D(x_={
        'z_list': z_list_0,
        'np_omega': shared_data['np_omega']
    })

    torch.save(p_0, config.p_0_for_mixture_path)
