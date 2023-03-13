from vblinear import VBLinear
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
from collections import defaultdict
import argparse
from scipy.stats import norm

#####################################
### MPL SETTINGS ###

from matplotlib import rc
from matplotlib import rcParams

FONTSIZE=15
rc('font',**{'family':'serif','serif':['Helvetica'],'size':FONTSIZE})
rc('text', usetex=True);
rc('xtick', labelsize=FONTSIZE);
rc('ytick', labelsize=FONTSIZE)
rc('text.latex', preamble=r'\usepackage{amsmath}')

rcParams['legend.loc']="upper right"
rcParams['legend.frameon']=False
rcParams["errorbar.capsize"] = 8.0
rcParams['lines.linewidth'] = 2.




def tanhone(input):
    return 3 * ( torch.tanh(input) + 1 )

class TanhPlusOne(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return tanhone(input) # simply apply already implemented SiLU

class BNN(nn.Module):
    def __init__(self, training_size, inner_layers, input_dim, activation_inner='tanh', activation_last=None):
        super(BNN, self).__init__()

        self.training_size = training_size
        self.all_layers = []
        self.vb_layers = []

        # add first and last layer
        inner_layers = [input_dim] + inner_layers
        inner_layers.append(2)

        for i in range(len(inner_layers)-1):
            vb_layer = VBLinear(inner_layers[i], inner_layers[i+1])
            self.vb_layers.append(vb_layer)
            self.all_layers.append(vb_layer)
            if i < (len(inner_layers)-2): # skipping last layer
                if activation_inner.lower() == "tanh":
                    # self.all_layers.append(nn.Tanh())
                    self.all_layers.append(TanhPlusOne())
                elif activation_inner.lower() == "relu":
                    self.all_layers.append(nn.ReLU())
                elif activation_inner.lower() == "softplus":
                    self.all_layers.append(nn.Softplus())
                else:
                    raise NotImplementedError("Option for activation function of inner layers is not implemented! Given: {}".format(
                        activation_inner))

        # last layer activation function
        self.last_activation = None
        if activation_last is not None:
            if activation_last.lower() == "tanh":
                self.last_activation = nn.Tanh()
            elif activation_last.lower() == "relu":
                self.last_activation = nn.ReLU()
            elif activation_last.lower() == "softplus":
                self.last_activation = nn.Softplus()
            elif activation_last.lower() == "none":
                 self.last_activation = None
            else:
                raise NotImplementedError("Option for lactivation function of last layer is not implemented! Given {}".format(
                    activation_last))

        self.model = nn.Sequential(*self.all_layers)

    def set_num_training(self, training_size):
        self.training_size = training_size

    def forward(self, x):
        y = self.model(x)
        y0, y1 = y[:, 0], y[:, 1]
        if self.last_activation is not None:
            y0 = self.last_activation(y0)
        return torch.stack((y0, y1), axis=1)

    def set_map(self, map):
        for vb_layer in self.vb_layers:
            vb_layer.map = map

    def reset_random(self):
        '''
        reset random values used to sample network weights in each layer
        '''

        for layer in self.vb_layers:
            layer.reset_random()
        return

    def KL(self):
        '''
        return KL Loss summed from each layer
        '''

        kl = 0
        for vb_layer in self.vb_layers:
            kl += vb_layer.KL()
        return kl / self.training_size


class NN(nn.Module):
    def __init__(self, inner_layers, input_dim, activation_inner='tanh', activation_last=None, out_dim=2):
        super(NN, self).__init__()

        self.out_dim = out_dim
        if self.out_dim > 2:
            raise NotImplementedError("Option with out_dim > 2 not implemented!")

        self.all_layers = []
        self.l_layers = []

        # add first and last layer
        inner_layers = [input_dim] + inner_layers
        inner_layers.append(out_dim)

        for i in range(len(inner_layers)-1):
            l_layer = nn.Linear(inner_layers[i], inner_layers[i+1])
            self.l_layers.append(l_layer)
            self.all_layers.append(l_layer)
            if i < (len(inner_layers)-2): # skipping last layer
                if activation_inner.lower() == "tanh":
                    self.all_layers.append(nn.Tanh())
                elif activation_inner.lower() == "relu":
                    self.all_layers.append(nn.ReLU())
                elif activation_inner.lower() == "softplus":
                    self.all_layers.append(nn.Softplus())
                else:
                    raise NotImplementedError(
                        "Option for activation function of inner layers is not implemented! Given: {}".format(
                        activation_inner))

        # last layer activation function
        self.last_activation = None
        if activation_last is not None:
            if activation_last.lower() == "tanh":
                self.last_activation = nn.tanh()
            elif activation_last.lower() == "relu":
                self.last_activation = nn.ReLU()
            elif activation_last.lower() == "softplus":
                self.last_activation = nn.Softplus()
            elif activation_last.lower() == "none":
                 self.last_activation = None
            else:
                raise NotImplementedError(
                    "Option for lactivation function of last layer is not implemented! Given {}".format(
                    activation_last))

        self.model = nn.Sequential(*self.all_layers)

    def forward(self, x):
        if self.out_dim == 2:
            y = self.model(x)
            y0, y1 = y[:, 0], y[:, 1]
            if self.last_activation is not None:
                y0 = self.last_activation(y0)
            return torch.stack((y0, y1), axis=1)
        elif self.out_dim == 1:
            return self.model(x)
