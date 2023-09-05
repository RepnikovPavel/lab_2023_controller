import numpy as np
import torch 
import config 
import os
from config import theta_v_range, omega_v_range, F_v_range
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
from general import plot_rect
from numba import jit
from Simulation.sim_supp import make_psi, make_simulation_for_one_policy_function

@jit(nopython=True)
def search_position_on_grid(grid:np.array,value:float)->int:
    # return segment index to which balue belongs
    N = len(grid)
    for i in range(N-2):
        if value >= grid[i] and value < grid[i+1]:
            return i
    i=N-2
    if value >= grid[i] and value <= grid[i+1]:
        return i
        
def make_psi_from_policy_matrix(PI_,s1_grid,s2_grid,a1_values,N1,N2,M,translators_units_of_measurement,stat_decision='random'):

    if stat_decision == 'mean':
        def policy_func(x_1:float,x_2:float) -> float:
            # input [-1,1] \times [-1,1]
            # output [-1,1]
            k1 = search_position_on_grid(s1_grid,x_1)
            k2 = search_position_on_grid(s2_grid,x_2)
            i_ =  N2*k1 + k2
            # return np.random.choice(a1_values,p=PI_[i_])
            # return a1_values[np.argmax(PI_[i_])]
            return np.sum(a1_values*PI_[i_])
        return make_psi(policy_func,translators_units_of_measurement)    
    elif stat_decision == 'argmax':
        def policy_func(x_1:float,x_2:float) -> float:
            # input [-1,1] \times [-1,1]
            # output [-1,1]
            k1 = search_position_on_grid(s1_grid,x_1)
            k2 = search_position_on_grid(s2_grid,x_2)
            i_ =  N2*k1 + k2
            # return np.random.choice(a1_values,p=PI_[i_])
            return a1_values[np.argmax(PI_[i_])]
            # return np.sum(a1_values*PI_[i_])
        return make_psi(policy_func,translators_units_of_measurement)    
    elif stat_decision == 'random':
        def policy_func(x_1:float,x_2:float) -> float:
            # input [-1,1] \times [-1,1]
            # output [-1,1]
            k1 = search_position_on_grid(s1_grid,x_1)
            k2 = search_position_on_grid(s2_grid,x_2)
            i_ =  N2*k1 + k2
            return np.random.choice(a1_values,p=PI_[i_])
            # return a1_values[np.argmax(PI_[i_])]
            # return np.sum(a1_values*PI_[i_])
        return make_psi(policy_func,translators_units_of_measurement)    


def make_voltage_psi_from_policy_matrix(PI_,s1_grid,s2_grid,a1_values,N1,N2,M,stat_decision='random'):
    if stat_decision == 'mean':
        def policy_func(x_1:float,x_2:float) -> float:
            # input [-1,1] \times [-1,1]
            # output [-1,1]
            k1 = search_position_on_grid(s1_grid,x_1)
            k2 = search_position_on_grid(s2_grid,x_2)
            i_ =  N2*k1 + k2
            # return np.random.choice(a1_values,p=PI_[i_])
            # return a1_values[np.argmax(PI_[i_])]
            return np.sum(a1_values*PI_[i_])
        return policy_func
    elif stat_decision == 'argmax':
        def policy_func(x_1:float,x_2:float) -> float:
            # input [-1,1] \times [-1,1]
            # output [-1,1]
            k1 = search_position_on_grid(s1_grid,x_1)
            k2 = search_position_on_grid(s2_grid,x_2)
            i_ =  N2*k1 + k2
            # return np.random.choice(a1_values,p=PI_[i_])
            return a1_values[np.argmax(PI_[i_])]
            # return np.sum(a1_values*PI_[i_])
        return policy_func
    elif stat_decision == 'random':
        def policy_func(x_1:float,x_2:float) -> float:
            # input [-1,1] \times [-1,1]
            # output [-1,1]
            k1 = search_position_on_grid(s1_grid,x_1)
            k2 = search_position_on_grid(s2_grid,x_2)
            i_ =  N2*k1 + k2
            return np.random.choice(a1_values,p=PI_[i_])
            # return a1_values[np.argmax(PI_[i_])]
            # return np.sum(a1_values*PI_[i_])
        return policy_func

N1 = 10
N2 = 10
M = 10

s1_step = (theta_v_range[1]-theta_v_range[0])/N1 
s2_step = (omega_v_range[1]-omega_v_range[0])/N2
a1_step = (F_v_range[1]-F_v_range[0])/M

s1_grid = np.linspace(start=theta_v_range[0],stop=theta_v_range[1],num = N1+1)
s2_grid = np.linspace(start=omega_v_range[0],stop=omega_v_range[1],num = N2+1)
a1_grid = np.linspace(start=F_v_range[0],stop=F_v_range[1],num = M+1)
s1_grid_si = np.array([config.translators_units_of_measurement['from_th_in_volt_to_th_in_si'](el) for el in s1_grid],dtype=np.float32)
s2_grid_si = np.array([config.translators_units_of_measurement['from_omega_in_volt_to_omega_in_si'](el) for el in s2_grid],dtype=np.float32)
a1_grid_si = np.array([config.translators_units_of_measurement['from_v_in_volt_to_v_in_si'](el) for el in a1_grid],dtype=np.float32)
s1_values = np.arange(start = theta_v_range[0]+s1_step/2,stop=theta_v_range[1],step= s1_step)
s2_values = np.arange(start = omega_v_range[0]+s2_step/2,stop=omega_v_range[1],step= s2_step)
a1_values = np.arange(start = F_v_range[0]+a1_step/2,stop=F_v_range[1],step= a1_step)

PI = torch.load(os.path.join(config.task_dir,'policy_ml_backup.txt'))

x1grid = s1_grid
x2grid = s2_grid
test_des = 'mean'
func = make_voltage_psi_from_policy_matrix(PI, s1_grid,s2_grid,a1_values,N1,N2,M,stat_decision=test_des)
psi = make_psi_from_policy_matrix(PI, s1_grid,s2_grid,a1_values,N1,N2,M,config.translators_units_of_measurement,stat_decision=test_des)
nx = 30
ny = 30
x_grid = np.linspace(-1.0,1.0, nx)
y_grid = np.linspace(-1.0,1.0, ny)

# x_grid = s1_grid
# y_grid = s2_grid
# nx = len(x_grid)
# ny= len(y_grid)
o_=  ''
for i in range(nx-1):
    for j in range(ny-1):
        p1 = (x_grid[i],y_grid[j])
        p2 = (x_grid[i+1],y_grid[j])
        p3 = (x_grid[i+1],y_grid[j+1])
        p4 = (x_grid[i+1],y_grid[j+1])
        p5 = (x_grid[i],y_grid[j+1])
        p6 = (x_grid[i],y_grid[j])
        ps = [p1,p2,p3,p4,p5,p6]
        for k in range(len(ps)):
            o_ += str(ps[k][0]) + '\t' + str(ps[k][1]) + '\t' + str(func(ps[k][0],ps[k][1])) + '\n'
with open(os.path.join(config.task_dir,'ML_controller_surf.dat'),'w') as file:
    file.write(o_)




# os.path.join(config.task_dir,'ML_controller_surf.dat')

fig,ax = plt.subplots()
fig.set_size_inches(16,9)

x = []
y = []
rects_info = []
highs = []
num_of_points_in_plot = 0
# в этом случае передали surf

for i in range(len(x1grid)-1):
    for j in range(len(x2grid)-1):
        x_1 = x1grid[i]
        x_2 = x1grid[i+1]
        y_1 = x2grid[j]
        y_2 = x2grid[j+1]
        x.append((x_2 + x_1) / 2)
        y.append((y_2 + y_1) / 2)
        rects_info.append([x_1, x_2, y_1, y_2])
        expectation = np.mean([func((x_2 + x_1) / 2, (y_2 + y_1) / 2) for k in range(100)])
        highs.append(expectation)
        num_of_points_in_plot += 1
condition_of_break = np.asarray([
    config.theta_range,
    config.omega_range,
    [-9999.0, 9999.0],
    [-9999.0, 9999.0]
])


simulation = make_simulation_for_one_policy_function(
    psi=psi,
    phys_sim_params=config.phys_sim_params,
    condition_of_break=condition_of_break,
    object_params=config.phys_params,
    use_an_early_stop=False,
    action_check=False
    
)
T = config.phys_sim_params['t_end']
y_0 = config.phys_sim_params['y_0']
v_0 = config.phys_sim_params['v_0']

from_th_in_volt_to_th_in_si = config.translators_units_of_measurement['from_th_in_volt_to_th_in_si']
from_omega_in_volt_to_omega_in_si = config.translators_units_of_measurement['from_omega_in_volt_to_omega_in_si']
from_th_in_si_to_th_in_volt = config.translators_units_of_measurement['from_th_in_si_to_th_in_volt']
from_omega_in_si_to_omega_in_volt = config.translators_units_of_measurement['from_omega_in_si_to_omega_in_volt']
trajectories = []
colors = []
alphas = []
n_bad = 0
n_good = 0
S_0 = []
for i in range(N1):
    for j in range(N2):
        S_0.append((s1_values[i],s2_values[j]))
N_max = 40
for i in range(N_max):
    # j = np.random.randint(0, len(S_0))
    # s_0 = S_0[j]
    s_0 = S_0[i]
    code_of_sim, solution, time_of_simulation, control_actions = simulation(
                                                                    from_th_in_volt_to_th_in_si(s_0[0]),
                                                                    from_omega_in_volt_to_omega_in_si(s_0[1]),
                                                                    y_0, v_0)
    trajectories.append(solution[:,:2])
    if code_of_sim == 1:
        colors.append('r')
        alphas.append(0.5)
        n_bad  +=1 
    else:
        colors.append('k')
        alphas.append(0.3)
        n_good +=1 

if not os.path.exists(os.path.join(config.task_dir, 'ml_trajectories')):
    os.makedirs(os.path.join(config.task_dir, 'ml_trajectories'))
# prune trajectories 
tr_ = []
N_in_tr = 100
for tr in trajectories: 
    step_ = np.maximum(int(len(tr)/N_in_tr),1)
    tr_.append(tr[::step_])

for i in range(len(tr_)):
    tr_i = tr_[i]
    o_ = ''
    for j in range(len(tr_i)):
        o_ += str(from_th_in_si_to_th_in_volt(tr_i[j][0])) + '\t' + str(from_omega_in_si_to_omega_in_volt(tr_i[j][1])) + '\n'
    with open(os.path.join(config.task_dir, 'ml_trajectories','trajectory_{}.dat'.format(i)),'w') as file:
        file.write(o_)



norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
tmp_len = len(rects_info)
for i in range(len(rects_info)):
    print("\r interation {} of {}".format(i, tmp_len), end='')
    plot_rect(ax, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],
                m.to_rgba(highs[i]))

plt.colorbar(m, ax=ax)
ax.set_xlim([-1.0, 1.0])
ax.set_ylim([-1.0, 1.0])
ax.set_xlabel(r"$\theta, \: V$")
ax.set_ylabel(r'$\omega, \: V$')
ax.set_title(r'$F, \: V $')
plt.show()