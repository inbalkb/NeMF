# This file contains util routines for NeMF.
# It is based on VIP-CT source code ('https://github.com/ronenroi/VIPCT') by Roi Ronen
# Copyright (c) Roi Ronen et al.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Inbal Kom Betzer, Roi Ronen, Vadim Holodovsky, Yoav. Y. Schechner and Ilan Koren, 
# "NeMF: Neural Microphysics Fields",
# TBD TPAMI 2024.
#
# Copyright (c) Inbal Kom Betzer. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# author. The author is not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def show_scatter_plot(gt_param, est_param, param_name='parameter',
                      colorbar_param = None, colorbar_name = None):
    gt_param = gt_param.detach().cpu().numpy().ravel()
    est_param = est_param.detach().cpu().numpy().ravel()
    max_val = max(gt_param.max(), est_param.max())
    fig, ax = plt.subplots()
    if colorbar_param is None:
        ax.scatter(gt_param, est_param, facecolors='none', edgecolors='b')
        ax.set_title('Estimated vs. True Values of the ' + param_name)
    else:
        colorbar_param = colorbar_param.cpu().detach().numpy().ravel()
        scat = ax.scatter(gt_param, est_param, c=colorbar_param, cmap=plt.cm.jet)
        plt.colorbar(scat)
        ax.set_title('Estimated vs. True Values of the ' + param_name + f'\ncolor-coded by' + colorbar_name)

    ax.set_xlim([0, 1.1 * max_val])
    ax.set_ylim([0, 1.1 * max_val])
    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
    ax.set_ylabel('Estimated', fontsize=14)
    ax.set_xlabel('True', fontsize=14)
    ax.set_aspect('equal')

    plt.show()

def show_scatter_plot_altitute(gt_param, est_param):
    import matplotlib.cm as cm
    gt_param = gt_param.detach().cpu().numpy()
    est_param = est_param.detach().cpu().numpy()
    colors = cm.rainbow(np.linspace(0, 1, gt_param.shape[-1]))

    max_val = max(gt_param.max(), est_param.max())
    fig, ax = plt.subplots()
    for i, c in enumerate(colors):
        if i>10:
            ax.scatter(gt_param[...,i].ravel(), est_param[...,i].ravel(),
                   facecolors='none', edgecolors=c, label=i)
        else:
            ax.scatter(gt_param[..., i].ravel(), est_param[..., i].ravel(),
                       facecolors='none', edgecolors=c)
    ax.set_xlim([0, 1.1 * max_val])
    ax.set_ylim([0, 1.1 * max_val])
    ax.legend(loc='best',fontsize='small')
    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
    ax.set_ylabel('Estimated', fontsize=14)
    ax.set_xlabel('True', fontsize=14)
    ax.set_aspect('equal')
    plt.show()

def volume_plot(gt_param, est_param):
    gt_param = gt_param.detach().cpu().numpy()
    est_param = est_param.detach().cpu().numpy()
    ax = plt.figure().add_subplot(projection='3d')
    plt.title("GT")
    ax.voxels(gt_param)
    plt.show()
    ax = plt.figure().add_subplot(projection='3d')
    plt.title("Est.")
    ax.voxels(est_param)
    plt.show()


