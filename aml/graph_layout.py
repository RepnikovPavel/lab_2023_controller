import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm

def cluster_plot(fig_size, positions, colors_of_points, color_for_labels,labels_for_colors):
    fig,ax = plt.subplots()
    fig.set_size_inches(fig_size[0],fig_size[1])
    x = [el[0] for el in positions]
    y = [el[1] for el in positions]
    ax.scatter(x,y,c=colors_of_points)
    a_x = np.min(x)
    b_x = np.max(x)
    a_y = np.min(y)
    b_y = np.max(y)
    h_x = b_x-a_x
    h_y = b_y-a_y
    ax.set_xlim([a_x-h_x*0.01,b_x+h_x*0.01])
    ax.set_ylim([a_y-h_y*0.01,b_y+h_y*0.01])
    for i in range(len(labels_for_colors)):
        ax.scatter([-2*a_x],[-2*a_y],c=[color_for_labels[i]],label= labels_for_colors[i])
    ax.set_facecolor("black")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper right")
    return fig,ax

def cluster_plot3d(fig_size, positions, colors_of_points, color_for_labels,labels_for_colors):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    x = [el[0] for el in positions]
    y = [el[1] for el in positions]
    z = [el[2] for el in positions]
    ax.scatter(x,y,z,c=colors_of_points)
    a_x = np.min(x)
    b_x = np.max(x)
    a_y = np.min(y)
    b_y = np.max(y)
    a_z = np.min(z)
    b_z = np.max(z)

    h_x = b_x-a_x
    h_y = b_y-a_y
    h_z = b_z-a_z
    ax.set_xlim([a_x-h_x*0.01,b_x+h_x*0.01])
    ax.set_ylim([a_y-h_y*0.01,b_y+h_y*0.01])
    ax.set_zlim([a_z-h_z*0.01,b_z+h_z*0.01])
    for i in range(len(labels_for_colors)):
        ax.scatter([-99*a_x],[-99*a_y],[-99*a_z],c=[color_for_labels[i]],label= labels_for_colors[i])
    # ax.set_facecolor("black")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper right")
    return fig,ax

def rho(x1, x2):
    return torch.sqrt(torch.sum(torch.square(x1-x2)))

def l_i(x, x_vec, d_i_square,N):
    rho_vec = torch.sum(torch.square(x_vec - x),dim=1)
    l_i = torch.sum(torch.square(rho_vec - d_i_square))
    return l_i


def loss(x, d_m, N):
    # loss = 0.0
    # for i in range(N-1):
        # for j in range(i+1,N):
            # loss += torch.square(rho(x[i],x[j]) - d_m[i][j])
    loss = 0.0
    for i in range(N):
        loss += l_i(x[i],x, d_m[i],N)
    return loss/(N*(N-1))

class GraphOnAPlane:
    d_m: torch.tensor
    current_pos: torch.tensor
    out_pos: np.array
    loss_vec: np.array
    device: str
    N: int
    def __init__(self, distance_matrix: np.array, size_of_output_space):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        self.device = device
        print('using device {}'.format(self.device))
        self.N = len(distance_matrix)
        N = self.N
        self.d_m = torch.square(torch.tensor(distance_matrix,dtype=torch.float32)).to(device).requires_grad_(False)
        start_pos =  np.zeros(shape=(N, size_of_output_space))
        for i in range(size_of_output_space):
            start_pos[:,i] = np.random.uniform(low=0.0,high=1.0,size=N)
        self.current_pos =  torch.tensor(start_pos, dtype=torch.float32).to(self.device).requires_grad_(True)

    def fit(self):
        optimizer = torch.optim.Adam([self.current_pos],betas=[0.5,0.7])
        EPOCH = 550
        loss_vec = np.zeros(EPOCH,)
        def ep_lr(ep_index):
            if ep_index < 125:
                return 0.1
            elif ep_index >= 125 and ep_index < 250:
                return 0.2
            elif ep_index >= 250 and ep_index < 350:
                return 0.1
            elif ep_index >= 350 and ep_index < 450:
                return 0.01
            elif ep_index >= 450 and ep_index < 550:
                return 0.001
            
        for i in tqdm(range(EPOCH)):
            # print('\r{}/{}'.format(i+1,EPOCH),end='')
            for g in optimizer.param_groups:
                g['lr'] = ep_lr(i)
            optimizer.zero_grad()
            loss_ = loss(self.current_pos,self.d_m,self.N)
            loss_for_plot = float(loss_.cpu().detach().numpy())
            loss_vec[i] = loss_for_plot
            loss_.backward()
            optimizer.step()
        print('')
        self.loss_vec = loss_vec
        self.out_pos = self.current_pos.cpu().detach().numpy()
    def get_pos(self):
        return self.out_pos
    def plot_loss(self):
        fig,ax = plt.subplots()
        fig.set_size_inches(16,9)
        ax.plot(self.loss_vec)
        ax.set_yscale('log')
        ax.set_title(r'$loss = \frac{1}{N(N-1)} \sum_{i}\sum_{j}(\rho_{ij}^{2}-d_{ij}^{2})^{2} \: during \: grad \: descent$')
        plt.show()

# if __name__ == '__main__':

#     D = np.array([
#         [0,1,3,4],
#         [1,0,2,4],
#         [3,2,0,4],
#         [4,4,4,0]
#     ])

#     GM = GraphOnAPlane(D)
#     GM.fit()
#     GM.plot_loss()
#     positions = GM.get_pos()
#     N_ = len(positions)
#     cmap = matplotlib.colors.ListedColormap(sns.color_palette("bright", N_).as_hex())
#     colors_ = [cmap(i) for i in range(N_)]
#     fig,ax = cluster_plot3d((16,9),positions,colors_of_points=colors_,color_for_labels=[cmap(el) for el in range(4)],labels_for_colors=[0,1,2,3])
#     plt.show()