import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from FindMinimumOfFunc.view_layer import plot_rect

def plotf(x:np.array, y:np.array, func_):

    plt.rcParams["figure.figsize"] = [14, 7]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    nx = len(x)
    ny= len(y)
    x_ = np.zeros(shape=(nx*ny,))
    y_ = np.zeros(shape=(nx*ny,))
    f_ = np.zeros(shape=(nx*ny,))
    k = 0
    for i in range(nx):
        for j in range(ny):
            x_[k] = x[i]
            y_[k] = y[j]
            f_[k] = func_(x[i], y[j])
            k+=1

    norm = matplotlib.colors.Normalize(vmin=min(f_), vmax=max(f_))
    m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    axs.scatter(x_, y_, f_, c=[m.to_rgba(f_[i]) for i in range(len(f_))])

    fig.colorbar(m, ax=axs)

    axs.set_xlabel(r"$t$")
    axs.set_ylabel(r'$x$')
    axs.set_title(r'$\mathcal{L}$')

    argmin_ = np.argmin(f_)
    print('min value = {} x= {}, y={}'.format(f_[argmin_],x_[argmin_], y_[argmin_]))

    print("plot response surface done")
    return axs

def func_(t, x):
    t0 = 0.0
    x_goal = 0.0
    T = 10.0 # seсond
    start_x_error_accumulation = T/2.0
    ex = np.abs(x-x_goal)
    xe_accum = max(0.0, ex*(t-t0)-ex*(start_x_error_accumulation-t0))
    early_error = ex/(t-t0)
    lx = early_error + xe_accum
    # +np.abs(x_end)*t_end
    L_ = lx
    return L_

if __name__ == '__main__':
    N = 80
    x_vec = np.linspace(start=-4.0, stop=4.0, num= N)
    x_vec = x_vec/4.0
    t_vec = np.linspace(start=0.1, stop=10.0, num=N)
    t_vec = t_vec
    axs = plotf(x=t_vec, y=x_vec, func_=func_)
    plt.show()