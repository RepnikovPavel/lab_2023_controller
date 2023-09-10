import torch
import config
import numpy as np
import matplotlib.pyplot as plt
import os
from Alg.solving_algorithm import ModelGenerator

N = 1000

mg = ModelGenerator(rules=config.rules,
                            cache_dir=config.Phi_cache_dir,
                            clear_cache=False)
vectors = torch.load(config.Phi_vector_representation)
all_p = [torch.load(os.path.join(mg.cache_dir, 'distrib4D_{}.txt'.format(i))) for i in range(N)]
all_v = []
