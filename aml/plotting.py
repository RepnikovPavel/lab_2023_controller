import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import os 
import numpy as np
import time 
from datetime import datetime
import keras
import os
from typing import List, Dict,Tuple, Any
import io


def plot_float_distribution(data,
                            x_label=None,
                            y_label = None, 
                            title=None,
                            font_size = None,
                            fig_size=(4,3)):

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size[0],fig_size[1])
    x = []
    for i in range(len(data)):
        if np.isnan(data[i]):
            continue
        else:
            x.append(data[i])
    u_vs = np.unique(x)

    if len(x) == 0:
        if title is None:
            ax.set_title('data is empty')
    elif len(u_vs)==1:
        if title is None:
            ax.set_title('all data is repeated with value: {}'.format(u_vs[0]))
    else:
        x = np.asarray(x)
        q25, q75 = np.percentile(x, [25, 75])
        bins = 0
        if q25==q75:
            bins = np.minimum(100,len(u_vs))
        else:
            bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
            bins = np.minimum(100, round((np.max(x) - np.min(x)) / bin_width))
        nan_rate = np.sum(np.isnan(data))/len(data)
        if title is None:
            ax.set_title('n of unique values {}'.format(len(u_vs)))
        else:
            ax.set_title(title)

        if x_label is None:
            ax.set_xlabel('nan rate {}'.format(nan_rate))
        else:
            ax.set_xlabel(x_label)
        
        if y_label is not None:
            ax.set_ylabel(y_label)

        density,bins = np.histogram(x,bins=bins,density=True)
        unity_density = density / density.sum()
        widths = bins[:-1] - bins[1:]
        ax.bar(bins[1:], unity_density,width=widths)

    if font_size is not None:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size)

    return fig,ax

def bar_plot(objects, heights, fig_size=(4,3),title=''):
    fig, ax = plt.subplots()
    fig.set_size_inches(*fig_size)
    ax.bar(objects, heights)
    ax.set_title(title)
    return fig,ax


def simple_plot(x,y,title='',size= (16,9)):
    fig, ax = plt.subplots()
    fig.set_size_inches(*size)
    ax.plot(x,y)
    ax.set_title(title)
    ax.grid()
    ax.minorticks_on()
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    return fig,ax


def simple_plot_y1y2(x,y1,y2,label1='',label2='',title=''):
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    ax.plot(x,y1,label=label1)
    ax.plot(x,y2,label=label2)
    ax.set_title(title)
    ax.grid(which='both')
    ax.legend()
    return fig,ax

def simple_plot_y1y2y3y4(x,y1,y2,y3,y4,label1='',label2='',label3='',label4='',title=''):
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    ax.plot(x,y1,label=label1)
    ax.plot(x,y2,label=label2)
    ax.plot(x,y3,label=label3)
    ax.plot(x,y4,label=label4)
    ax.set_title(title)
    ax.grid(which='both')

    ax.legend()
    return fig,ax

def get_time():
    return datetime.now().strftime("%H_%M_%S")

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

class TensorBoard:
    # windows support only 
    logdir: str
    TBPath: str
    LabelWriter: Dict[str, Any]
    port: str
    def __init__(self,
                 tensorboard_exe_path: str,
                 logdir:str, port: str,
                 mode='linux'):
        
        if mode=='win':
            if not os.path.exists(logdir):
                os.makedirs(logdir)        
            self.logdir =  logdir
            self.TBPath = tensorboard_exe_path
            self.LabelWriter = {}
            self.port = port
            cmd_ = 'start '+ tensorboard_exe_path  + ' --logdir {} --port {}'.format(
                logdir, port)
            os.system(cmd_)
            print('tensorboard http://localhost:{}/'.format(port))
        if mode =='linux':
            if not os.path.exists(logdir):
                os.makedirs(logdir)        
            self.logdir =  logdir
            self.TBPath = tensorboard_exe_path
            self.LabelWriter = {}
            self.port = port
            cmd_ = tensorboard_exe_path  + ' --logdir {} --port {} &'.format(
                logdir, port)
            os.system(cmd_)
            print('tensorboard http://localhost:{}/'.format(port))

    def InitExperiment(self, experiment_metadata: str):
        subdir_ = os.path.join(self.logdir, experiment_metadata)
        if not os.path.exists(subdir_):
            os.makedirs(subdir_)        
        wr_ = tf.summary.create_file_writer(subdir_)
        self.LabelWriter.update({experiment_metadata:wr_})

    def Push(self, experiment_metadata,  x, y, label):
        with self.LabelWriter[experiment_metadata].as_default():
            tf.summary.scalar(label, data = y , step = x)
    def PlotImage(self,experiment_metadata, img, label,step=0):
        with self.LabelWriter[experiment_metadata].as_default():
            tf.summary.image(label, img, step=step)