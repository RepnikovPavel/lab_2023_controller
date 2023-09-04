import numpy as np
import torch
import matplotlib.pyplot as plt
from general import plot_vec
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer


def target_f(x):
    return 1/x

def sol1(x, p):
    return p[0]*torch.exp(p[1]*x+p[2])+p[3]

def LOSS(target_f, solution,p, grid):
    error = target_f(grid)-solution(grid, p)
    sq_erorr = torch.max(torch.abs(error))
    agg_error = torch.sum(sq_erorr)
    return agg_error


if __name__ == '__main__':
    grid = np.linspace(start=1.0,stop=2.0,num=1000).reshape(1000,1)
    y_truth = target_f(grid)

    est = SymbolicRegressor()
    function_set = ['add', 'mul', 'cos', 'sub', 'sin']
    gp= SymbolicRegressor(population_size=5000,
                      generations=20, stopping_criteria=0.001,
                      p_crossover=0.7, p_subtree_mutation=0.1,
                      function_set=function_set,
                      p_hoist_mutation=0.05, p_point_mutation=0.1,
                      max_samples=0.9, verbose=1,
                      parsimony_coefficient=0.01, random_state=0)
    gp.fit(grid, y_truth)
    print(gp._program)

    predict = gp.predict(grid)


    fig, ax = plt.subplots(1, 1)
    ax.plot(grid, predict, label=r'predict')
    ax.plot(grid, y_truth, label=r'true')
    ax.legend()
    print('max abs error {}'.format(np.max(np.abs(y_truth-predict))))

    plt.show()
