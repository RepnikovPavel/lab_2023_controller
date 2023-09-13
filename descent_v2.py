import torch
import config
import numpy as np
import matplotlib.pyplot as plt
import os
from Alg.solving_algorithm import ModelGenerator
from CustomModels.my_models import Integrator
from CustomModels.my_models import weighted_amount
from aml.plotting import *
from Losses.Losses import *
from sklearn import decomposition
from tqdm import tqdm
from scipy.spatial import ConvexHull

def adjust_alpha(alpha_n):
    for j in range(len(alpha_n)):
        if alpha_n[j]<0.0:
            alpha_n[j] = 0.0
        elif alpha_n[j] > 1.0:
            alpha_n[j] = 1.0
    alpha_n = alpha_n/np.sum(alpha_n)
    return alpha_n 

def grad_descent_from_alpha_in_simplex(alpha_vec,p_list, shared_integration_supports):
    d = len(p_list)
    alpha_n = np.copy(alpha_vec)
    alpha_n = alpha_n/np.sum(alpha_n)
    n = 0
    p_mid = None
    L_mid = None
    all_losses = []
    while True:
        p_mid = weighted_amount(list_of_distributions=p_list, alpha_list=alpha_n)
        L_mid = get_L2_Distrib4D(p_mid,shared_integration_supports)
        # print('n {} L {}'.format(n, L_mid))
        # board_reference.Push(experiment_metadata=experiment_metadata,
        #     x=n,y= L_mid, label='L')
        gradient_ = np.zeros(shape=(d,))
        for j in range(d):
            epsilon_ = 10**(-6)
            if alpha_n[j] < epsilon_:
                alpha_1 = np.copy(alpha_n)
                alpha_1[j] = alpha_1[j] + epsilon_
                p_1 = weighted_amount(list_of_distributions=p_list, alpha_list=alpha_1)
                L_1 = get_L2_Distrib4D(p_1,shared_integration_supports)
                gradient_[j] = (L_1 - L_mid)/epsilon_
                continue
            if alpha_n[j] > 1.0-epsilon_:
                alpha_2 = np.copy(alpha_n)
                alpha_2[j] = alpha_2[j] - epsilon_
                p_2 = weighted_amount(list_of_distributions=p_list, alpha_list=alpha_2)
                L_2 = get_L2_Distrib4D(p_2,shared_integration_supports)
                gradient_[j] = (L_mid-L_2)/epsilon_
                continue

            alpha_1 = np.copy(alpha_n)
            alpha_1[j] = alpha_1[j] + epsilon_
            alpha_2 = np.copy(alpha_n)
            alpha_2[j] = alpha_2[j] - epsilon_

            p_1 = weighted_amount(list_of_distributions=p_list, alpha_list=alpha_1)
            p_2 = weighted_amount(list_of_distributions=p_list, alpha_list=alpha_2)

            L_1 = get_L2_Distrib4D(p_1,shared_integration_supports)
            L_2 = get_L2_Distrib4D(p_2,shared_integration_supports)
            gradient_[j] = (L_1 - L_2)/(2*epsilon_)
        

        lambda_vec = np.logspace(start=-1,stop=-6,num=10)
        lambda_best = None
        loss_current = L_mid
        ls = []
        lambda_ls = []
        for lambda_ in lambda_vec:
            alpha_copy = np.copy(alpha_n)
            alpha_after = alpha_copy - lambda_*gradient_
            if np.sum(alpha_after < 0.0)==d:
                continue
            alpha_after = adjust_alpha(alpha_after)        
            p_after = weighted_amount(list_of_distributions=p_list, alpha_list=alpha_after)
            L_after = get_L2_Distrib4D(p_after,shared_integration_supports)
            ls.append(L_after)
            lambda_ls.append(lambda_)
            if L_after<loss_current:
                loss_current = L_after
                lambda_best = lambda_
        # arg_best = np.argsort(ls)[0]
        # left_pos = np.maximum(0, arg_best-1)
        # right_pos = np.minimum(len(ls)-1, arg_best+1)

        # left = lambda_ls[left_pos]
        # right = lambda_ls[right_pos]

        # lambda_vec = np.linspace(left,right,10)
        # addls = []
        # addlambda_ls = []
        # for lambda_ in lambda_vec:
        #     alpha_copy = np.copy(alpha_n)
        #     alpha_after = alpha_copy - lambda_*gradient_
        #     if np.sum(alpha_after < 0.0)==d:
        #         continue
        #     alpha_after = adjust_alpha(alpha_after)        
        #     p_after = weighted_amount(list_of_distributions=p_list, alpha_list=alpha_after)
        #     L_after = get_L2_Distrib4D(p_after,shared_integration_supports)
        #     addls.append(L_after)
        #     addlambda_ls.append(lambda_)
        #     if L_after<loss_current:
        #         loss_current = L_after
        #         lambda_best = lambda_
        # fig,ax = plt.subplots()
        # ax.plot(lambda_ls,ls,color= 'k')
        # ax.plot(addlambda_ls,addls,color= 'r')
        # plt.show()
        

        all_losses.append(loss_current)
        if loss_current == L_mid:
            break        

        alpha_n = alpha_n - lambda_best*gradient_
        alpha_n = adjust_alpha(alpha_n)        

        # board_reference.Push(experiment_metadata=experiment_metadata,
        #     x=n,y= lambda_best, label='best_lambda')
        
        # board_reference.Push(experiment_metadata=experiment_metadata,
        #     x=n,y= np.linalg.norm(gradient_), label='grad_norm')

        n+=1
    
    return p_mid, L_mid,all_losses, alpha_n

def grad_descent(p_list, shared_integration_supports):
    d = len(p_list)
    alpha_mid = np.ones(shape=(d,))
    alpha_mid = alpha_mid/np.sum(alpha_mid)
    alpha_vecs = [alpha_mid]    
    for j in range(d):
        alpha_ = np.zeros(shape=(d,))
        alpha_[j] = 1.0
        alpha_vecs.append(alpha_)
    
    losses_vecs = []
    L_per_start_iter=[]
    P_per_start_iter=[]
    for j in range(len(alpha_vecs)):
        alpha_j = alpha_vecs[j]
        best_p, best_L, losses_j, best_alpha = grad_descent_from_alpha_in_simplex(alpha_j,p_list, shared_integration_supports)
        losses_vecs.append(losses_j)
        L_per_start_iter.append(best_L)
        P_per_start_iter.append(best_p)
    argmin_ = np.argmin(L_per_start_iter)
    L_best_in_simplex = L_per_start_iter[argmin_] 
    p_best_in_simplex =  P_per_start_iter[argmin_]
    # fig,ax = plt.subplots()
    # for j in range(len(alpha_vecs)):
    #     x_ = np.arange(0,len(losses_vecs[j]))
    #     y_ = losses_vecs[j]
    #     ax.plot(x_,y_)
    # plt.show()

    
    return p_best_in_simplex, L_best_in_simplex

     



N = 1000

mg = ModelGenerator(rules=config.rules,
                            cache_dir=config.Phi_cache_dir,
                            clear_cache=False)
vectors = torch.load(config.Phi_vector_representation)
all_p = [torch.load(os.path.join(mg.cache_dir, 'distrib4D_{}.txt'.format(i))) for i in range(N)]
all_v = torch.load(os.path.join(config.task_dir, 'L2_for_Phi.txt'))
# print(np.sort(all_v))
simplices = torch.load(os.path.join(config.task_dir, 'triangulation_simplexes.txt'))
sorted_vertesices = [el for el in np.argsort(all_v)]  
# get simliexes with best loss 
sorted_simplixes = []
simplexes_best = []
for best_vertex in sorted_vertesices:
    for simplex in simplices:
        if best_vertex in  simplex:
            sorted_simplixes.append(simplex)
            simplexes_best.append(all_v[best_vertex])

d = len(vectors[0])
support_vertexes = torch.load(os.path.join(config.task_dir, 'support_points.txt'))

mg = ModelGenerator(rules=config.rules,
                            cache_dir=config.Phi_cache_dir,
                            clear_cache=False)
shared_integration_supports = Integrator(dir_=config.integrator_dir,
                                        shared_data=mg.shared_data,
                                        clear_cache=True).shared_integration_supports



# board = TensorBoard(tensorboard_exe_path=config.tensorboard_path,
#                     logdir=os.path.join(config.task_dir, 'descent_log'),
#                     port= '64001')
# exp_metadata = 'sorted_simplexes'+get_time()
# board.InitExperiment(experiment_metadata= exp_metadata)


global_L_min = 10**99
p_best = None
losses_per_simplex_optimization = {i:[] for i in range(len(sorted_simplixes))}
# max_iter = np.maximum(int(0.005*len(sorted_simplixes)),1)
max_iter = 200
for i in tqdm(range(max_iter)):
    # j = np.random.randint(0, len(sorted_simplixes))
    # simplex = sorted_simplixes[j]
    # BestOfSimplex = simplexes_best[j]
    simplex = sorted_simplixes[i]
    BestOfSimplex = simplexes_best[i]

    p_in_simplex = [all_p[el] for el in simplex]
    # print(BestOfSimplex)
    p_fitted, L_last = grad_descent(p_in_simplex,shared_integration_supports)
    # losses_per_simplex_optimization[i] = all_losses
    if L_last < global_L_min:
        global_L_min = L_last
        p_best = p_fitted
        torch.save(p_best, config.Phi_descent_best_p_path)
    print('simplex L_0 {} simplex fitted L {}'.format(BestOfSimplex,L_last))

print(global_L_min)

# fig,ax = plt.subplots()
# max_n = np.max([len(losses_per_simplex_optimization[el]) for el in range(len(sorted_simplixes))])-1

# for i in range(len(sorted_simplixes)):
#     if len(losses_per_simplex_optimization[i]) == 0:
#         continue
#     y_ = losses_per_simplex_optimization[i]
#     x_ = np.arange(0, len(y_))  
#     ax.plot(x_,y_,color = (0,0,0))
# best_before_optimization = min(all_v)
# ax.plot([0, max_n],[best_before_optimization, best_before_optimization],color = (1,0,0),linestyle='dashed')
# plt.show()

# simplex L_0 1.6826208999482832 simplex fitted L 1.9186021968737395
# simplex L_0 1.7423562157937476 simplex fitted L 1.6822348353922052
# simplex L_0 1.7469186634864808 simplex fitted L 1.9097130059780238
# simplex L_0 1.7470295642878546 simplex fitted L 1.9636281906019348
# simplex L_0 1.7603488105364296 simplex fitted L 1.9887041683847702
# c:\paper\descent_v2.py:21: RuntimeWarning: invalid value encountered in divide
#   alpha_n = alpha_n/np.sum(alpha_n)
# simplex L_0 1.7856643762943485 simplex fitted L 0.0006927929992389181
# simplex L_0 1.7964567185284022 simplex fitted L 1.9847641969225287

# support_p = [all_p[el] for el in support_vertexes]
# support_L = [get_L2_Distrib4D(el,shared_integration_supports) for el in support_p]
# print('support losses')
# print(support_L)
# fig,ax = plot_float_distribution(support_L)
# plt.show()


# alpha_n = np.ones(shape=(len(support_vertexes),))
# alpha_n = alpha_n/ np.sum(alpha_n)
# T_0 = 3000
# T_n = T_0
# MAX_GENERATIONS = 1000
# MAX_ITER = 10**4
# L_best = 10**99
# L_current = 10**99
# alpha_best_ = None
# # board.Push(experiment_metadata=exp_metadata,
# #         x=0,y=T_n, label='T')


# for ITER in tqdm(range(MAX_ITER)):
#     sigma_n = T_n/T_0
#     for j in range(MAX_GENERATIONS):
#         alpha_ = GenAlphaVec(mu_current=alpha_n, value_of_noise=0.3*sigma_n)
#         p_ = weighted_amount(list_of_distributions=support_p, alpha_list=alpha_)
#         L_new = get_L2_Distrib4D(p_,shared_integration_supports)
#         H_ = ProbOfChangeToNewParams(L_new,L_current, 1.0,T_n)
#         alpha_updated, is_update = UpdateMu(alpha_n,alpha_,epsilon=0.8,ProbOfUpdate=H_)
        
#         # if is_update:
#         #     board.Push(experiment_metadata=exp_metadata,
#         #     x=ITER,y= L_new, label='L')

#         if is_update and (L_new<L_current):
#             L_current = L_new
#             alpha_n = alpha_updated
#             break
#         elif is_update and not (L_new<L_current):
#             L_current = L_new
#             alpha_n = alpha_updated

#         if L_new < L_best:
#             L_best = L_new
#             alpha_best_ = np.copy(alpha_updated)
        
#     T_n = Temp(ITER,T_0,MAX_ITER)
#     # board.Push(experiment_metadata=exp_metadata,
#     #         x=ITER,y= T_n, label='T')

# print('L best {}'.format(L_best))
# print(alpha_best_)


# def Temp(k,T_0,max_iter):
#     # return T_0/np.log(k+2)
#     # return T_0/k
#     a = -T_0/max_iter
#     b = T_0
#     return a*k+b

# def ProbOfChangeToNewParams(L_new, L_old, beta, current_temp):
#     H_ = 0.0
#     if L_new < L_old:
#         H_ = 1.0
#     else:
#         # H_=  np.exp((L_old-L_new)/(beta*current_temp))
#         H_ = 0.0
#     return H_

# def UpdateMu(mu_current,mu_new,epsilon,ProbOfUpdate):
#     if ProbOfUpdate == 1.0:
#         # new_ = epsilon*mu_current + (1-epsilon)*mu_new
#         new_ = mu_new
#         new_ = new_/np.sum(new_)
#         return  new_, True
#     if ProbOfUpdate == 0.0:
#         return  mu_current, False
    
#     v_ = ProbOfUpdate*1000
#     coin_=  np.random.randint(0,1001)
#     if coin_ <= v_:
#         new_ = mu_new
#         # new_ = epsilon*mu_current + (1-epsilon)*mu_new
#         new_ = new_/np.sum(new_)
#         return new_, True
#     else:
#         return mu_current,False
        
# def GenAlphaVec(mu_current, value_of_noise):
#     noise = np.random.rand(len(mu_current))
#     noise = noise/np.sum(noise)
#     new_ = (1.0-value_of_noise)*mu_current + value_of_noise* noise
#     new_ = new_/np.sum(new_)
#     return new_



# alpha_n = np.ones(shape=(len(support_vertexes),))
# alpha_n = alpha_n/ np.sum(alpha_n)
# T_0 = 3000
# T_n = T_0
# MAX_GENERATIONS = 1000
# MAX_ITER = 10**4
# L_best = 10**99
# L_current = 10**99
# alpha_best_ = None


# for ITER in tqdm(range(MAX_ITER)):
#     sigma_n = T_n/T_0
#     for j in range(MAX_GENERATIONS):
#         alpha_ = GenAlphaVec(mu_current=alpha_n, value_of_noise=0.3*sigma_n)
#         p_ = weighted_amount(list_of_distributions=support_p, alpha_list=alpha_)
#         L_new = get_L2_Distrib4D(p_,shared_integration_supports)
#         H_ = ProbOfChangeToNewParams(L_new,L_current, 1.0,T_n)
#         alpha_updated, is_update = UpdateMu(alpha_n,alpha_,epsilon=0.8,ProbOfUpdate=H_)
        
#         # if is_update:
#         #     board.Push(experiment_metadata=exp_metadata,
#         #     x=ITER,y= L_new, label='L')

#         if is_update and (L_new<L_current):
#             L_current = L_new
#             alpha_n = alpha_updated
#             break
#         elif is_update and not (L_new<L_current):
#             L_current = L_new
#             alpha_n = alpha_updated

#         if L_new < L_best:
#             L_best = L_new
#             alpha_best_ = np.copy(alpha_updated)
        
#     T_n = Temp(ITER,T_0,MAX_ITER)
#     # board.Push(experiment_metadata=exp_metadata,
#     #         x=ITER,y= T_n, label='T')