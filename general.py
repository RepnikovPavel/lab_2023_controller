import copy

import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, animation

import time
from RulesMaker.func_det import Distrib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_cube(axes, x_1, x_2, y_1, y_2, z_1, z_2):
    cube_definition = [
        (x_1, y_1, z_1), (x_1, y_2, z_1), (x_2, y_1, z_1), (x_1, y_1, z_2)
    ]
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    # faces = Poly3DCollection(edges, linewidths=1, edgecolors='k',alpha=1)
    faces = Poly3DCollection(edges)
    faces.set_facecolor((0, 0, 1, 0.1))

    axes.add_collection3d(faces)


def compute_seq_vec(x_vec, y_vec, z_vec, c_i_j_k, N, a_1, a_2, a_3, b_1, b_2, b_3):
    l_1 = (b_1 - a_1) / 2
    l_2 = (b_2 - a_2) / 2
    l_3 = (b_3 - a_3) / 2
    x_shape = len(x_vec)
    y_shape = len(y_vec)
    z_shape = len(z_vec)
    alpha_1 = torch.ones(size=(2 * N + 1, x_shape, 1))
    alpha_2 = torch.ones(size=(2 * N + 1, y_shape, 1))
    alpha_3 = torch.ones(size=(2 * N + 1, z_shape, 1))
    for i in range(1, 2 * N + 1, 2):
        alpha_1[i] = torch.cos(i * np.pi * x_vec / l_1)
        alpha_1[i + 1] = torch.sin(i * np.pi * x_vec / l_1)
        alpha_2[i] = torch.cos(i * np.pi * y_vec / l_2)
        alpha_2[i + 1] = torch.sin(i * np.pi * y_vec / l_2)
        alpha_3[i] = torch.cos(i * np.pi * z_vec / l_2)
        alpha_3[i + 1] = torch.sin(i * np.pi * z_vec / l_2)

    # sum = torch.zeros(size=(x_shape, y_shape, z_shape))
    sum = 0
    for i in range(2 * N + 1):
        for j in range(2 * N + 1):
            for k in range(2 * N + 1):
                sum += c_i_j_k[i][j][k] * (alpha_1[i] * torch.t(alpha_2[j])).view(x_shape, y_shape, 1) * torch.t(
                    alpha_3[k])

    return torch.abs(sum)


def compute_gauss(x_vec, y_vec, z_vec, mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z):
    x_shape = len(x_vec)
    y_shape = len(y_vec)
    return (1 / 2.50662827463 / sigma_x * torch.exp(-1 / 2 * torch.square((x_vec - mu_x) / sigma_x)) * torch.t(
        1 / 2.50662827463 / sigma_y * torch.exp(-1 / 2 * torch.square((y_vec - mu_y) / sigma_y)))).view(x_shape,
                                                                                                        y_shape,
                                                                                                        1) * torch.t(
        1 / 2.50662827463 / sigma_z * torch.exp(-1 / 2 * torch.square((z_vec - mu_z) / sigma_z)))


def plot_f(axs, z, number_of_rule, number_of_distribution, omega, a, h, distr: Distrib):
    axs.grid(True, 'both', 'both')

    x = np.linspace(distr.support[0], distr.support[1], 100)
    x_plot = np.zeros(99, )
    y = np.zeros(99, )
    for i in range(100 - 1):
        y[i] = distr(0.5 * (x[i] + x[i + 1]))
        x_plot[i] = 0.5 * (x[i] + x[i + 1])
    line_1, = axs.plot(x_plot, y)
    line_1.set_label('ground truth')

    sum_tuple = []
    for tmp_i in range(len(omega[number_of_rule - 1])):
        if (tmp_i != (number_of_distribution - 1)):
            sum_tuple.append(tmp_i)
    sum_tuple = tuple(sum_tuple)
    y2 = (((torch.sum(z[number_of_rule - 1] * a[number_of_rule - 1], sum_tuple)) / (
        torch.sum(z[number_of_rule - 1] * a[number_of_rule - 1]))) / h[number_of_rule - 1][
              number_of_distribution - 1]).cpu().detach().numpy()
    tmp_x2 = omega[number_of_rule - 1][number_of_distribution - 1]
    x2 = torch.zeros(len(tmp_x2), )
    for i in range(len(x2)):
        x2[i] = (tmp_x2[i][0] + (tmp_x2[i][1] - tmp_x2[i][0]) / 2)
    x2 = x2.cpu().detach().numpy()
    line_2, = axs.plot(x2, y2, '*')
    line_2.set_label('approx')
    axs.legend()
    axs.set_title("rule: {} distr:{} {}".format(number_of_rule, number_of_distribution,
                                                str(distr.distrib.__name__)))
    axs.set_yscale("linear")


def plot_consistency(z, rules, omega, a, h):
    z_ = [torch.tensor(z[i]) for i in range(len(z))]
    # print("########################################################################")
    print("drawing consistency:")
    print("     minimum and maximum values:")
    for p in range(len(z_)):
        print("     rule number: {}, min: {}   max: {}".format(p + 1, torch.min(z_[p]).cpu().detach().numpy(),
                                                               torch.max(z_[p]).cpu().detach().numpy()))
    # print("########################################################################")
    figures = []
    axes = []
    k = 0
    for rule_key in rules.get_ordered_keys():
        fig, axs = plt.subplots(1, len(z_[k].size()))
        figures.append(fig)
        axes.append(axs)
        IF_distrs = rules.rules_dict[rule_key]['IF']
        THEN_distrs = rules.rules_dict[rule_key]['THEN']
        j = 0
        for distr in IF_distrs:
            plot_f(axs[j], z_, k + 1, j + 1, omega, a, h, distr)
            j += 1
        for distr in THEN_distrs:
            plot_f(axs[j], z_, k + 1, j + 1, omega, a, h, distr)
            j += 1
        k += 1
    plt.show(block=True)


def make_new_omega_for_rect(old_omega, rect, list_of_rules):
    # создадим неравномерную сетку по осям основываясь на предыдущем рабиении и границах области определения переменных
    # находим левую и правую границу по каждой оси

    new_omega_for_rect = []
    for i in range(len(rect)):
        new_omega_for_rect.append([rect[i][0]])
    #    зафикисировали переменную для которой будем находить риски водль координатной прямой
    for i in range(len(rect)):
        while True:
            tmp_left_border = new_omega_for_rect[i][-1]
            if tmp_left_border == rect[i][1]:
                break
            # проходимся только по тем праавилам, которые указаны в спец. списке, т.к. этот прямоугольник только
            # они и накрывают
            tmp_min_new_left_border = []
            for p in list_of_rules:
                # узнаем диапазон индексов по которым нужно передвигаться в p-м правиле
                # вдоль i-го направления
                j_min = 0
                j_max = 0
                if len(old_omega[p][i]) == 1:
                    j_min = 0
                    j_max = 1
                else:
                    for i_of_segment in range(len(old_omega[p][i])):
                        # если риска попала в один из отрезков разбиения - запомним это
                        if old_omega[p][i][i_of_segment][0] <= rect[i][0] <= old_omega[p][i][i_of_segment][1]:
                            j_min = i_of_segment
                            for i_of_following_segment in range(i_of_segment, len(old_omega[p][i])):
                                if old_omega[p][i][i_of_following_segment][0] <= rect[i][1] <= \
                                        old_omega[p][i][i_of_following_segment][1]:
                                    j_max = i_of_following_segment
                                    break
                            break
                # зная диапазон индексов вычислим риску следущую за текущей риской, ближайщую к текущей
                if len(old_omega[p][i]) == 1:
                    if old_omega[p][i][0][1] > tmp_left_border:
                        tmp_min_new_left_border.append(old_omega[p][i][0][1])
                else:
                    # так и не понял, почему добовление j_min, j_max + 1 в место len(old_omega[p][i] меняет решение а не
                    # только ускоряет вычисления
                    for i_of_segment in range(j_min, j_max + 1):
                        if old_omega[p][i][i_of_segment][0] > tmp_left_border:
                            tmp_min_new_left_border.append(old_omega[p][i][i_of_segment][0])
                            break
            if len(tmp_min_new_left_border) == 0:
                tmp_min_new_left_border.append(rect[i][1])
            tmp = np.asarray(tmp_min_new_left_border)
            new_omega_for_rect[i].append(np.min(tmp))

    return_new_omega = []
    for i in range(len(rect)):
        return_new_omega.append([])
        for j in range(len(new_omega_for_rect[i]) - 1):
            return_new_omega[i].append([new_omega_for_rect[i][j], new_omega_for_rect[i][j + 1]])
    tmp_new_size = 1
    for i in range(len(new_omega_for_rect)):
        tmp_new_size *= len(new_omega_for_rect[i])

    return return_new_omega, tmp_new_size


def plot_rect(ax, x_1, x_2, y_1, y_2, color):
    h_1 = x_2 - x_1
    h_2 = y_2 - y_1
    rect = matplotlib.patches.Rectangle((x_1, y_1), h_1, h_2, color=color)
    ax.add_patch(rect)


def plot_empty_rect(ax, x_1, x_2, y_1, y_2,color):
    h_1 = x_2 - x_1
    h_2 = y_2 - y_1
    ax.plot([x_1,x_1],[y_1,y_2],c=color,linewidth=5)
    ax.plot([x_2,x_2],[y_1,y_2],c=color,linewidth=5)
    ax.plot([x_1,x_2],[y_1,y_1],c=color,linewidth=5)
    ax.plot([x_1,x_2],[y_2,y_2],c=color,linewidth=5)


def plot_policy_function(
        mode_of_plot,
        filepath_to_save_response_surface, p_func, Grids, block_canvas):
    '''
    plot_policy_function(mode_of_plot='map',
                         filepath_to_save_response_surface='',
                         p_func=p_func,
                         Grids=shared_integration_supports['Grids'],
                         block_canvas=True
                         )
    '''

    print("\nstart plot response surface")

    x = []
    y = []
    rects_info = []
    highs = []
    num_of_points_in_plot = 0

    # в этом случае передали surf
    for i in range(len(Grids)):
        superposition_of_grids_in_reference_area = Grids[i]
        for j in range(len(superposition_of_grids_in_reference_area[0])):
            for k in range(len(superposition_of_grids_in_reference_area[1])):
                x_1 = superposition_of_grids_in_reference_area[0][j][0]
                x_2 = superposition_of_grids_in_reference_area[0][j][1]
                y_1 = superposition_of_grids_in_reference_area[1][k][0]
                y_2 = superposition_of_grids_in_reference_area[1][k][1]
                x.append((x_2 + x_1) / 2)
                y.append((y_2 + y_1) / 2)
                rects_info.append([x_1, x_2, y_1, y_2])
                highs.append(p_func((x_2 + x_1) / 2, (y_2 + y_1) / 2))
                num_of_points_in_plot += 1
    print("\nnum_of_points_in_plot {}".format(num_of_points_in_plot))

    min_h = np.min(highs)
    max_h = np.max(highs)
    print('min F {}, max F {}'.format(min_h,max_h))
    if mode_of_plot == "map":
        plt.rcParams["figure.figsize"] = [14, 7]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        axs = fig.add_subplot(111)
        norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
        m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

        tmp_len = len(rects_info)
        for i in range(len(rects_info)):
            print("\r interation {} of {}".format(i, tmp_len), end='')
            plot_rect(axs, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],
                      m.to_rgba(highs[i]))

        plt.colorbar(m, ax=axs)
        axs.set_xlim([-1.0, 1.0])
        axs.set_ylim([-1.0, 1.0])
        axs.set_xlabel(r"$\theta, \: V$")
        axs.set_ylabel(r'$\omega, \: V$')
        axs.set_title(r'$F, \: V $')

        print("plot response surface done")
        return fig, axs

def draw_arrow(axs, arr_start, arr_end):
       dx = arr_end[0] - arr_start[0]
       dy = arr_end[1] - arr_start[1]
       axs.arrow(arr_start[0], arr_start[1], dx, dy, head_width=0.01, head_length=0.01, length_includes_head=True, color='black')

def plot_policy_function_with_trajectories(
        mode_of_plot,
        filepath_to_save_response_surface, p_func, Grids,
        all_theta,
        all_omega,
        loss_vec
        ):
    '''
    plot_policy_function(mode_of_plot='map',
                         filepath_to_save_response_surface='',
                         p_func=p_func,
                         Grids=shared_integration_supports['Grids'],
                         block_canvas=True
                         )
    '''

    # print("\nstart plot response surface")

    x = []
    y = []
    rects_info = []
    highs = []
    num_of_points_in_plot = 0

    # в этом случае передали surf
    for i in range(len(Grids)):
        superposition_of_grids_in_reference_area = Grids[i]
        for j in range(len(superposition_of_grids_in_reference_area[0])):
            for k in range(len(superposition_of_grids_in_reference_area[1])):
                x_1 = superposition_of_grids_in_reference_area[0][j][0]
                x_2 = superposition_of_grids_in_reference_area[0][j][1]
                y_1 = superposition_of_grids_in_reference_area[1][k][0]
                y_2 = superposition_of_grids_in_reference_area[1][k][1]
                x.append((x_2 + x_1) / 2)
                y.append((y_2 + y_1) / 2)
                rects_info.append([x_1, x_2, y_1, y_2])
                highs.append(p_func((x_2 + x_1) / 2, (y_2 + y_1) / 2))
                num_of_points_in_plot += 1
    # print("\nnum_of_points_in_plot {}".format(num_of_points_in_plot))

    min_h = np.min(highs)
    max_h = np.max(highs)
    # print('min F {}, max F {}'.format(min_h,max_h))
    if mode_of_plot == "map":
        plt.rcParams["figure.figsize"] = [22, 10]
        plt.rcParams["figure.autolayout"] = True
        fig, (axs, axs2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 3]})
        norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
        m = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

        tmp_len = len(rects_info)
        for i in range(len(rects_info)):
            # print("\r interation {} of {}".format(i, tmp_len), end='')
            plot_rect(axs, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],
                      m.to_rgba(highs[i]))

        plt.colorbar(m, ax=axs)
        axs.set_xlim([-1.0, 1.0])
        axs.set_ylim([-1.0, 1.0])
        axs.set_xlabel(r"$\theta, \: V$")
        axs.set_ylabel(r'$\omega, \: V$')
        axs.set_title(r'$F, \: V $')
        for i in range(len(all_theta)):
            axs.plot(all_theta[i],all_omega[i],c=(1,0,0,1))
            draw_arrow(axs,arr_start=[all_theta[i][-2],all_omega[i][-2]],arr_end=[all_theta[i][-1],all_omega[i][-1]])


        for i in range(len(loss_vec)):
            axs2.plot(loss_vec[i])
        axs2.set_title('loss along the trajectory')
        # print("plot response surface done")
        return fig


def plot_animation(theta_vec, y_vec, time_vec, L, tau, fps, control_actions, sim_results,
                   units_translators):

    print("\nmaking animation")
    fig = plt.figure()
    fig.set_figheight(9)
    fig.set_figwidth(16)
    fig.subplots_adjust(
        top=0.981,
        bottom=0.049,
        left=0.042,
        right=0.981,
        hspace=0.2,
        wspace=0.2
    )
    ax1 = plt.subplot(2, 4, 1)
    ax2 = plt.subplot(2, 4, 2)
    ax3 = plt.subplot(2, 4, 5)
    ax4 = plt.subplot(2, 4, 6)
    ax5 = plt.subplot(1, 2, 2)
    axs = [ax1, ax2, ax3, ax4,ax5]

    axs[0].set_xlabel(r'$x,sm$')
    axs[0].set_ylabel(r'$y,sm$')
    axs[0].set_aspect('equal')
    axs[0].grid()
    axs[0].set_xlim([-L * 100 * 2, L * 100 * 2])
    axs[0].set_ylim([-L * 100 * 2, L * 100 * 2])
    line_pivot, = axs[0].plot([], [], marker="*")
    time_text = axs[0].text(0.05, 0.9, '', transform=axs[0].transAxes)

    real_time_vec = np.linspace(start=0, stop=len(control_actions) * tau, num=len(control_actions))

    from_v_in_si_to_v_in_volt = units_translators['from_v_in_si_to_v_in_volt']
    from_omega_in_si_to_omega_in_volt = units_translators['from_omega_in_si_to_omega_in_volt']
    from_th_in_si_to_th_in_volt = units_translators['from_th_in_si_to_th_in_volt']
    F_copy = np.copy(control_actions)
    for i in range(len(control_actions)):
        F_copy[i]=from_v_in_si_to_v_in_volt(F_copy[i])

    axs[1].set_xlabel(r'$t,sek$')
    axs[1].set_ylabel(r'$F,V$')
    axs[1].grid()
    axs[1].plot(real_time_vec, F_copy, color='b')



    axs[4].set_xlabel(r'$\theta$,V')
    axs[4].set_ylabel(r'$\omega,V$')
    omega_copy = np.copy(sim_results[:, 1])
    theta_copy = np.copy(sim_results[:, 0])
    for i in range(len(omega_copy)):
        omega_copy[i]=from_omega_in_si_to_omega_in_volt(omega_copy[i])
        theta_copy[i]=from_th_in_si_to_th_in_volt(theta_copy[i])

    axs[4].plot(theta_copy, omega_copy, color='b', alpha=0.5)
    axs[4].grid()
    end_th = theta_copy[-1]
    end_omega = omega_copy[-1]
    axs[4].scatter(end_th, end_omega, color='k')
    start_th = theta_copy[0]
    start_omega = omega_copy[0]
    axs[4].scatter(start_th, start_omega, color='r')

    y_copy = np.copy(-sim_results[:, 2] * 100)
    axs[2].set_xlabel(r'$t,sek$')
    axs[2].set_ylabel(r'$y,sm$')
    axs[2].plot(real_time_vec[:len(y_copy)], y_copy, color='b', alpha=0.5)
    axs[2].grid()


    v_copy = np.copy(-sim_results[:, 3] * 100)
    axs[3].set_xlabel(r'$t,sek$')
    axs[3].set_ylabel(r'$v,sm/sek$')
    axs[3].plot(real_time_vec[:len(v_copy)], v_copy, color='b', alpha=0.5)
    axs[3].grid()

    v_source_vec = -sim_results[:, 3] * 100  # in sgs
    # v_copy = np.copy(sim_results[:,3])*100
    # acc_vec = []
    # acc_vec.append(0.0)
    # for i in range(1, len(sim_results) - 1):
    #     acc_vec.append((v_copy[i + 1] - v_copy[i - 1]) / (2 * tau))
    # axs[1, 2].set_xlabel(r'$t,sek$')
    # axs[1, 2].set_ylabel(r'$a,sm/sek^2$')
    # axs[1, 2].plot(real_time_vec[:-1], acc_vec, color='b', alpha=0.5)
    # axs[1, 2].grid()

    x_M = -y_vec[0]  # ось y в статье - это ось x в matplotlib. ось в статье направлена в другую сторону.
    y_M = 0.0
    x_m = -100 * (L * 2 * np.sin(theta_vec[0]) + y_vec[0])
    y_m = 100 * (L * 2 * np.cos(theta_vec[0]))
    t_end = str(time_vec[-1])[:3]

    def init():
        line_pivot.set_data([x_M, x_m], [y_M, y_m])
        time_text.set_text('')
        return line_pivot, time_text

    def animate(i):
        line_pivot.set_xdata([-100 * y_vec[i], -100 * (L * 2 * np.sin(theta_vec[i]) + y_vec[i])])
        line_pivot.set_ydata([0, 100 * (L * 2 * np.cos(theta_vec[i]))])
        time_text.set_text("{}/{}".format(str(time_vec[i])[:3], t_end))
        axs[0].set_xlim([-L * 100 * 2 - 100 * y_vec[i], L * 100 * 2 - 100 * y_vec[i]])
        fig.canvas.draw()
        # print('\r time {} sek'.format(t_array[i]), end='')
        return line_pivot, time_text

    ani = animation.FuncAnimation(
        fig, animate,
        frames=len(y_vec),
        blit=True,
        init_func=init)

    ani.save('debug.gif', writer='imagemagick', fps=fps)
    plt.close(fig)




def make_and_play_animation(solution, fps, time_of_simulation, tau, L, control_actions, sim_results,
                            units_translators):
    n_for_anim = int((1 / fps) / (tau))
    plot_animation(solution[:, 0][0:-1:n_for_anim], solution[:, 2][0:-1:n_for_anim],
                   np.linspace(start=0, stop=time_of_simulation, num=len(solution))[0:-1:n_for_anim], L, tau, fps,
                   control_actions, sim_results, units_translators)

def plot_vec(x, y,title='', block=True):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_title(title)
    if block == True:
        fig.waitforbuttonpress(timeout=-1)
    else:
        fig.show()


def calc_a_vec(v_vec, tau):
    N = len(v_vec)
    a_vec = np.zeros(shape=(len(v_vec,)))
    a_vec[0] = (v_vec[1]-v_vec[0])/tau
    a_vec[N-1] = (v_vec[N-1]-v_vec[N-2])/tau
    # j in 1,N-2
    for j in range(1, N-1):
        a_vec[j] = (v_vec[j+1]-v_vec[j-1])/(2*tau)
    return a_vec

def WMA(arr_):
    N = len(arr_)
    weights = np.arange(start=1, stop=N+1, step=1)
    weights = 2/(N*(N+1))*weights
    return np.ma.average(a=arr_, weights=weights)

def SMA(arr_):
    N = len(arr_)
    weights = np.ones(shape=(N,))
    weights = 1/N*weights
    return np.ma.average(a=arr_, weights=weights)


def get_averaged_arr(arr_, window_size):
    N = len(arr_)
    av_arr_ = np.zeros(shape=(N,))
    for i in range(N):
        if i+1 < window_size:
            av_arr_[i] = WMA(arr_[:i + 1])
        else:
            av_arr_[i] = WMA(arr_[(i + 1) - window_size:i + 1])

    return av_arr_

# T = 10.0 # seсond
# lx = np.abs(x_end)/t_end+np.abs(x_end)*t_end
# lv = np.abs(v_end)/t_end+np.abs(v_end)*t_end
# la = np.abs(WMA(subarray_of_a))*t_end
#
# #
# # lx большая lv маленькая la маенькая -  AGG большая.
# # сравнение двух L_= lx+lv+la будет не понимать, что на самом деле
# # lv + la сильно меньше у одного из них(если такое событие произошло)
# # минимальный набор
# # L_ = lx+lv
#
# L_ = lx+lv

# Приоритеты:
# В первую очередь смотрим на координаты
# Во-сторых смотрим на скорости
# В-третьих смотрим на ускорения
# L_ = 100*lx + 10*lv + 1*la

# среднее гармоническое ^{-1}

# L_ = (1 / lx + 1 / lv + 1 / la)/3

def sim_loss(x_end, v_end, a_end, t_end, T):

    # TODO: решить проблему с разными порядками величин
    x_p = x_end + v_end*(T-t_end) + a_end*(T-t_end)**2/2
    v_p = v_end + a_end*(T-t_end)

    lx = np.abs(x_p-0.0)
    lv = np.abs(v_p-0.0)
    # lv=0.0
    # la = np.maximum(0.0, np.abs(a_end-0.0)*(t_end-T/2))
    # la = 0.0
    la = np.abs(a_end-0.0)
    L_ = lx+lv+la

    return L_, lx, lv, la

def AGG_loss(errors_on_each_trajectory):
    return np.mean(errors_on_each_trajectory)

def scale_to_minusone_one(arr_, max_value):
    return arr_/max_value


def plot_trajectories(simulation, phys_sim_params, plot_tr_params, units_translators,make_animation):
    '''
    condition_of_break = np.asarray([
        config.theta_range,
        config.omega_range,
        [-9999.0, 9999.0],
        [-9999.0, 9999.0]
    ])
    psi = make_psi(policy_func=p_func,
                   translators_units_of_measurement=config.translators_units_of_measurement)

    simulation = make_simulation_for_one_policy_function(
        psi=p_func,
        phys_sim_params=config.phys_sim_params,
        condition_of_break=condition_of_break,
        object_params=config.phys_params
    )
    plot_trajectories(simulation=simulation,
                      phys_sim_params=config.phys_sim_params,
                      plot_tr_params=config.plot_trajectories_params,
                      units_translators=config.translators_units_of_measurement)
    '''
    fig, ax = plt.subplots(2, 2)
    ax[0][0].set_xlabel(r'$\theta$,V')
    ax[0][0].set_ylabel(r'$\omega,V$')

    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    n_x_for_sim = plot_tr_params['n_x_for_sim']
    n_y_for_sim = plot_tr_params['n_y_for_sim']
    x_1_range_v = plot_tr_params['x_1_range']
    x_2_range_v = plot_tr_params['x_2_range']

    y_0 = phys_sim_params['y_0']
    v_0 = phys_sim_params['v_0']
    t_0 = phys_sim_params['t_0']
    t_end = phys_sim_params['t_end']

    from_th_in_volt_to_th_in_si = copy.deepcopy(units_translators['from_th_in_volt_to_th_in_si'])
    from_omega_in_volt_to_omega_in_si = copy.deepcopy(units_translators['from_omega_in_volt_to_omega_in_si'])
    from_th_in_si_to_th_in_volt = copy.deepcopy(units_translators['from_th_in_si_to_th_in_volt'])
    from_omega_in_si_to_omega_in_volt = copy.deepcopy(units_translators['from_omega_in_si_to_omega_in_volt'])

    th_for_mul_sim = np.linspace(start=from_th_in_volt_to_th_in_si(x_1_range_v[0]),
                                 stop=from_th_in_volt_to_th_in_si(x_1_range_v[1]),
                                 num=n_x_for_sim)
    omega_for_mul_sim = np.linspace(start=from_omega_in_volt_to_omega_in_si(x_2_range_v[0]),
                                    stop=from_omega_in_volt_to_omega_in_si(x_2_range_v[1]),
                                    num=n_y_for_sim)

    sim_results = []
    times_of_sim = []

    n_bad = 0
    n_good = 0
    markers_for_sim_code = []
    # start_time = time.time()

    y_vecs = []
    v_vecs = []
    control_actions_vecs = []

    start_time = time.time()
    for i in range(n_x_for_sim):
        print('точка по theta {}/{}'.format(i+1, n_x_for_sim))
        for j in range(n_y_for_sim):
            th = from_th_in_si_to_th_in_volt(th_for_mul_sim[i])
            om = from_omega_in_si_to_omega_in_volt(omega_for_mul_sim[j])
            # if not (-0.5<=th<=0.5):
            #     continue
            # if th < -0.25 and om < -0.25:
            #     continue
            # if th > 0.25 and om > 0.25:
            #     continue
            if make_animation:
                code_of_sim, solution, time_of_simulation, control_actions = simulation(-0.13,
                                                                                        -0.5,
                                                                                        y_0, v_0)
                sim_results.append(solution)
                times_of_sim.append(time_of_simulation)
                markers_for_sim_code.append(code_of_sim)
                # code_of_sim == 0 - good try, 1-bad try
                if code_of_sim == 0:
                    n_good += 1
                elif code_of_sim == 1:
                    n_bad += 1
                make_and_play_animation(solution, 60, time_of_simulation, phys_sim_params['tau'], 0.25, control_actions,
                                        sim_results[-1],
                                        units_translators
                                        )
                raise SystemExit
            code_of_sim, solution, time_of_simulation, control_actions = simulation(th_for_mul_sim[i],
                                                                                    omega_for_mul_sim[j],
                                                                                    y_0, v_0)
            # plot_vec(np.linspace(start=t_0,stop=t_end,num=len(control_actions)),control_actions,title='',block=True)
            y_vecs.append(solution[:, 2])
            v_vecs.append(solution[:, 3])
            control_actions_vecs.append(control_actions)
            sim_results.append(solution)
            times_of_sim.append(time_of_simulation)
            markers_for_sim_code.append(code_of_sim)
            # code_of_sim == 0 - good try, 1-bad try
            if code_of_sim == 0:
                n_good += 1
            elif code_of_sim == 1:
                n_bad += 1
    print(time.time()-start_time)

    mean_time_of_sim = np.mean(times_of_sim)
    min_time_of_sim = np.min(times_of_sim)
    max_tim_of_sim = np.max(times_of_sim)
    target_sim_time = t_end - t_0

    for i in range(len(sim_results)):
        a = np.copy(sim_results[i][:, 0])
        b = np.copy(sim_results[i][:, 1])
        for j in range(len(a)):
            a[j] = from_th_in_si_to_th_in_volt(a[j])
            b[j] = from_omega_in_si_to_omega_in_volt(b[j])
        if markers_for_sim_code[i] == 0:
            ax[0][0].plot(a, b, color='b', alpha=0.5)
        if markers_for_sim_code[i] == 1:
            ax[0][0].plot(a, b, color='r', alpha=0.5)
        if markers_for_sim_code[i] == 2:
            ax[0][0].plot(a, b, color='g', alpha=0.5)

    for i in range(len(sim_results)):
        val_1 = from_th_in_si_to_th_in_volt(sim_results[i][0][0])
        val_2 = from_omega_in_si_to_omega_in_volt(sim_results[i][0][1])
        if markers_for_sim_code[i] == 0:
            ax[0][0].scatter(val_1, val_2, color='b')
        if markers_for_sim_code[i] == 1:
            ax[0][0].scatter(val_1, val_2, color='r')
        if markers_for_sim_code[i] == 2:
            ax[0][0].scatter(val_1, val_2, color='g')



    # ax[0][0].set_title(
    #     "n_bad={} n_good={}\nM(t)={} min={} max={} target={} sek\n".format(n_bad, n_good,
    #                                                                               str(mean_time_of_sim)[:4],
    #                                                                               str(min_time_of_sim)[:4],
    #                                                                               str(max_tim_of_sim)[:4],
    #                                                                               str(target_sim_time)[:4])
    #                                                                             )

    y_end = []
    v_end = []
    a_window_end = []
    max_a_vec = []
    all_a_end_smoothed = []

    tau = phys_sim_params['tau']
    time_window_size = 1.0
    window_size = int(np.floor(time_window_size/tau))

    for i in range(len(y_vecs)):
        y_vec_i = y_vecs[i]
        # if np.abs(y_vec_i[-1]*100) < 100:
        v_vec_i = v_vecs[i]
        control_actions_ith = control_actions_vecs[i]
        time_vec = np.linspace(start=0.0, stop=times_of_sim[i], num=len(y_vec_i))
        # ax[0][1].plot(time_vec, control_actions_ith) # do not plot

        ax[1][0].plot(time_vec, -y_vec_i*100)
        ax[1][1].plot(time_vec, -v_vec_i*100)

        # calc a and plot

        a_vec = calc_a_vec(v_vec_i, tau)
        max_a_vec.append(np.max(np.abs(a_vec)))
        # a_window_end.append(a_vec[-window_size:])
        # a_vec = get_averaged_arr(a_vec, window_size)
        # all_a_end_smoothed.append(WMA(a_vec[-window_size:]))
        ax[0][1].plot(time_vec, -a_vec*100)



        # if markers_for_sim_code[i] == 0:
        #     y_end.append(np.abs(y_vec_i[-1]) * 100)
        #     v_end.append(np.abs(v_vec_i[-1]) * 100)


    ax[0][1].set_title(r'$a, \frac{cm}{s^{2}}$')
    ax[1][1].set_title(r'$v, \frac{cm}{s}$')
    ax[1][0].set_title(r'$y, cm$')
    # ax[1][0].set_title('y, cm. worse = {} best = {}'.format(str(np.max(y_end)), str(np.min(y_end))))
    # ax[1][1].set_title('v, cm/s worse = {} best = {}'.format(str(np.max(v_end)), str(np.min(v_end))))

    # plt.show(block=True)

    #
    #
    # fig2 = plt.figure()
    # axs2 = fig2.add_subplot(projection='3d')
    #
    #
    # T = 10.0
    # losses = []
    # all_lx = []
    # all_lv = []
    # all_la = []
    # n_good = 0
    # for i in range(len(y_vecs)):
    #     if markers_for_sim_code[i] != 1:
    #         l_i, lx, lv, la = sim_loss(  y_vecs[i][-1],
    #                                      v_vecs[i][-1],
    #                                      all_a_end_smoothed[i],
    #                                      times_of_sim[i],
    #                                      T)
    #         losses.append(l_i)
    #         all_lx.append(lx)
    #         all_lv.append(lv)
    #         all_la.append(la)
    #         n_good += 1
    #
    # print('L = {}'.format(np.sum(losses)/n_good))
    # norm = matplotlib.colors.Normalize(vmin=min(losses), vmax=max(losses))
    # m = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    # k_= 0
    # for i in range(len(y_vecs)):
    #     if markers_for_sim_code[i] == 0 or markers_for_sim_code[i]==2:
    #         y_vec_i = y_vecs[i]
    #         time_vec = np.linspace(start=0.0, stop=times_of_sim[i], num=len(y_vec_i))
    #         axs2.plot(time_vec, -y_vec_i*100, np.zeros(shape=(len(y_vec_i),)), c='#000000')
    #         loss_ = losses[k_]
    #         axs2.scatter(times_of_sim[i], -y_vec_i[-1]*100, loss_, c=[m.to_rgba(loss_)])
    #         k_+=1
    #
    #
    # axs2.set_xlabel(r'$t$')
    # axs2.set_ylabel(r'$x$')
    # fig2.colorbar(m, ax=axs2)
    #
    # fig3 = plt.figure()
    # axs3 = fig3.add_subplot(projection='3d')
    #
    #
    # axs3.scatter(all_lx, all_lv, all_la, c= [m.to_rgba(losses[i]) for i in range(len(losses))])
    # fig3.colorbar(m, ax = axs3)
    # axs3.set_xlabel(r'$l_x$')
    # axs3.set_ylabel(r'$l_v$')
    # axs3.set_zlabel(r'$l_a$')
    # plt.show(block=True)
    return fig,ax

def plot_trajectories_only(simulation, phys_sim_params, plot_tr_params, units_translators,make_animation):
    '''
    condition_of_break = np.asarray([
        config.theta_range,
        config.omega_range,
        [-9999.0, 9999.0],
        [-9999.0, 9999.0]
    ])
    psi = make_psi(policy_func=p_func,
                   translators_units_of_measurement=config.translators_units_of_measurement)

    simulation = make_simulation_for_one_policy_function(
        psi=p_func,
        phys_sim_params=config.phys_sim_params,
        condition_of_break=condition_of_break,
        object_params=config.phys_params
    )
    plot_trajectories(simulation=simulation,
                      phys_sim_params=config.phys_sim_params,
                      plot_tr_params=config.plot_trajectories_params,
                      units_translators=config.translators_units_of_measurement)
    '''
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\theta$,V')
    ax.set_ylabel(r'$\omega,V$')

    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    n_x_for_sim = plot_tr_params['n_x_for_sim']
    n_y_for_sim = plot_tr_params['n_y_for_sim']
    x_1_range_v = plot_tr_params['x_1_range']
    x_2_range_v = plot_tr_params['x_2_range']

    y_0 = phys_sim_params['y_0']
    v_0 = phys_sim_params['v_0']
    t_0 = phys_sim_params['t_0']
    t_end = phys_sim_params['t_end']

    from_th_in_volt_to_th_in_si = copy.deepcopy(units_translators['from_th_in_volt_to_th_in_si'])
    from_omega_in_volt_to_omega_in_si = copy.deepcopy(units_translators['from_omega_in_volt_to_omega_in_si'])
    from_th_in_si_to_th_in_volt = copy.deepcopy(units_translators['from_th_in_si_to_th_in_volt'])
    from_omega_in_si_to_omega_in_volt = copy.deepcopy(units_translators['from_omega_in_si_to_omega_in_volt'])

    th_for_mul_sim = np.linspace(start=from_th_in_volt_to_th_in_si(x_1_range_v[0]),
                                 stop=from_th_in_volt_to_th_in_si(x_1_range_v[1]),
                                 num=n_x_for_sim)
    omega_for_mul_sim = np.linspace(start=from_omega_in_volt_to_omega_in_si(x_2_range_v[0]),
                                    stop=from_omega_in_volt_to_omega_in_si(x_2_range_v[1]),
                                    num=n_y_for_sim)

    sim_results = []
    times_of_sim = []

    n_bad = 0
    n_good = 0
    markers_for_sim_code = []
    # start_time = time.time()

    y_vecs = []
    v_vecs = []
    control_actions_vecs = []

    start_time = time.time()
    for i in range(n_x_for_sim):
        print('точка по theta {}/{}'.format(i+1, n_x_for_sim))
        for j in range(n_y_for_sim):
            th = from_th_in_si_to_th_in_volt(th_for_mul_sim[i])
            om = from_omega_in_si_to_omega_in_volt(omega_for_mul_sim[j])
            # if not (-0.5<=th<=0.5):
            #     continue
            # if th < -0.25 and om < -0.25:
            #     continue
            # if th > 0.25 and om > 0.25:
            #     continue
            if make_animation:
                code_of_sim, solution, time_of_simulation, control_actions = simulation(-0.13,
                                                                                        -0.5,
                                                                                        y_0, v_0)
                sim_results.append(solution)
                times_of_sim.append(time_of_simulation)
                markers_for_sim_code.append(code_of_sim)
                # code_of_sim == 0 - good try, 1-bad try
                if code_of_sim == 0:
                    n_good += 1
                elif code_of_sim == 1:
                    n_bad += 1
                make_and_play_animation(solution, 60, time_of_simulation, phys_sim_params['tau'], 0.25, control_actions,
                                        sim_results[-1],
                                        units_translators
                                        )
                raise SystemExit
            code_of_sim, solution, time_of_simulation, control_actions = simulation(th_for_mul_sim[i],
                                                                                    omega_for_mul_sim[j],
                                                                                    y_0, v_0)
            # plot_vec(np.linspace(start=t_0,stop=t_end,num=len(control_actions)),control_actions,title='',block=True)
            y_vecs.append(solution[:, 2])
            v_vecs.append(solution[:, 3])
            control_actions_vecs.append(control_actions)
            sim_results.append(solution)
            times_of_sim.append(time_of_simulation)
            markers_for_sim_code.append(code_of_sim)
            # code_of_sim == 0 - good try, 1-bad try
            if code_of_sim == 0:
                n_good += 1
            elif code_of_sim == 1:
                n_bad += 1
    print(time.time()-start_time)

    mean_time_of_sim = np.mean(times_of_sim)
    min_time_of_sim = np.min(times_of_sim)
    max_tim_of_sim = np.max(times_of_sim)
    target_sim_time = t_end - t_0

    for i in range(len(sim_results)):
        a = np.copy(sim_results[i][:, 0])
        b = np.copy(sim_results[i][:, 1])
        for j in range(len(a)):
            a[j] = from_th_in_si_to_th_in_volt(a[j])
            b[j] = from_omega_in_si_to_omega_in_volt(b[j])
        if markers_for_sim_code[i] == 0:
            ax.plot(a, b, color='b', alpha=0.5)
        if markers_for_sim_code[i] == 1:
            ax.plot(a, b, color='r', alpha=0.5)
        if markers_for_sim_code[i] == 2:
            ax.plot(a, b, color='g', alpha=0.5)

    for i in range(len(sim_results)):
        val_1 = from_th_in_si_to_th_in_volt(sim_results[i][0][0])
        val_2 = from_omega_in_si_to_omega_in_volt(sim_results[i][0][1])
        if markers_for_sim_code[i] == 0:
            ax.scatter(val_1, val_2, color='b')
        if markers_for_sim_code[i] == 1:
            ax.scatter(val_1, val_2, color='r')
        if markers_for_sim_code[i] == 2:
            ax.scatter(val_1, val_2, color='g')

    return fig,ax

def get_sim_results(simulation, phys_sim_params, plot_tr_params, units_translators):
    '''
    condition_of_break = np.asarray([
        config.theta_range,
        config.omega_range,
        [-9999.0, 9999.0],
        [-9999.0, 9999.0]
    ])
    psi = make_psi(policy_func=p_func,
                   translators_units_of_measurement=config.translators_units_of_measurement)

    simulation = make_simulation_for_one_policy_function(
        psi=p_func,
        phys_sim_params=config.phys_sim_params,
        condition_of_break=condition_of_break,
        object_params=config.phys_params
    )
    get_sim_results(simulation=simulation,
                      phys_sim_params=config.phys_sim_params,
                      plot_tr_params=config.plot_trajectories_params,
                      units_translators=config.translators_units_of_measurement)
    '''


    n_x_for_sim = plot_tr_params['n_x_for_sim']
    n_y_for_sim = plot_tr_params['n_y_for_sim']
    x_1_range_v = plot_tr_params['x_1_range']
    x_2_range_v = plot_tr_params['x_2_range']

    y_0 = phys_sim_params['y_0']
    v_0 = phys_sim_params['v_0']

    from_th_in_volt_to_th_in_si = copy.deepcopy(units_translators['from_th_in_volt_to_th_in_si'])
    from_omega_in_volt_to_omega_in_si = copy.deepcopy(units_translators['from_omega_in_volt_to_omega_in_si'])
    from_th_in_si_to_th_in_volt = copy.deepcopy(units_translators['from_th_in_si_to_th_in_volt'])
    from_omega_in_si_to_omega_in_volt = copy.deepcopy(units_translators['from_omega_in_si_to_omega_in_volt'])

    th_for_mul_sim = np.linspace(start=from_th_in_volt_to_th_in_si(x_1_range_v[0]),
                                 stop=from_th_in_volt_to_th_in_si(x_1_range_v[1]),
                                 num=n_x_for_sim)
    omega_for_mul_sim = np.linspace(start=from_omega_in_volt_to_omega_in_si(x_2_range_v[0]),
                                    stop=from_omega_in_volt_to_omega_in_si(x_2_range_v[1]),
                                    num=n_y_for_sim)


    times_of_sim = []
    n_bad = 0
    n_good = 0
    n_early_stopped = 0
    markers = []
    tau = phys_sim_params['tau']
    solutions = []
    for i in range(n_x_for_sim):
        for j in range(n_y_for_sim):
            code_of_sim, solution, time_of_simulation, control_actions = simulation(th_for_mul_sim[i],
                                                                                    omega_for_mul_sim[j],
                                                                                    y_0, v_0)
            solutions.append(solution)

            times_of_sim.append(time_of_simulation)
            if code_of_sim == 0:
                n_good +=1
            if code_of_sim == 1:
                n_bad +=1
            if code_of_sim == 2:
                n_early_stopped += 1
            markers.append(code_of_sim)



    result = {
        'n_good': n_good,
        'n_bad': n_bad,
        'n_early': n_early_stopped,
        'markers': markers,
        'solutions': solutions,
        'times': times_of_sim
    }

    return result
