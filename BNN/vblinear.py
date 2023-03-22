######################
### Imports ###

import math
import sys
import torch
import torch.nn as nn
import numpy as np

######################

class VBLinear(nn.Module):
    '''Linear layer of a Bayesian neural network (BNN). 
       The BNN represented here is a variational mean-field
       Bayesian neural network implemented via the local
       reparameterization trick.

       The linear layer works similar to the standard linear layer of pytorch.
       The only difference is that each trainable weight is represented by
       a trainable normal distribution. Each normal distribution comes with
       two trainable parameters:
            self.mu_w      = vector of mean-values
            self.logsig2_w = vector of log(sigma^2) values
       Learning log(sigma^2) is numerically more stable than learning sigma^2.

       The prior is set to a normal distribution as well. This assumption
       enters the explicit formula given in the KL() method.

       To sample different outputs the reset_random() method should be used:
            num_weight_samples = 50
            layer = VBLinear(...)
            for i in range(num_weight_samples):
                layer.reset_random()
                prediction = layer(input)

       @args:
            in_features: number of input features
            out_features: number of output features
            prior_prec: hyperparameter, defined as:
                prior_prec = 1 / sigma_{prior}^2
                where sigma_{prior} is the width of
                the normal prior distribution N(mu=0, sigma=sigma_{prior})
    '''
   
    def __init__(self, in_features, out_features, prior_prec=1.0):
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = False
        self.prior_prec = prior_prec
        self.random = None
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        '''Reset / initialize trainable parameters randomlly'''
        stdv = 1. / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(-9, 0.001)
        self.bias.data.zero_()

    def reset_random(self):
        '''Force resampling from the variational distributions.'''
        self.random = None

    def KL(self, loguniform=False):
        '''KL-diergence of two normal distributions:
              KL(
                N(mu = self.mu_w, sigma^2 = self.logsig2_2.exp()) |
                N(mu = 0,         sigma^2 = 1/prior_prec)
              )
        # TODO: is the order correct in formula given above?
        '''
        logsig2_w = self.logsig2_w.clamp(-20, 11)
        kl = 0.5 * (self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                    - logsig2_w - 1 - np.log(self.prior_prec)).sum()
        return kl

    def forward(self, input):
        if self.training:
            # local reparameterization trick is more efficient and leads to
            # an estimate of the gradient with smaller variance.
            # https://arxiv.org/pdf/1506.02557.pdf
            mu_out = nn.functional.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-20, 11)
            s2_w = logsig2_w.exp()
            var_out = nn.functional.linear(input.pow(2), s2_w) #+ 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:

            # MAP: Maximum a posteriori method
            if self.map:
                return nn.functional.linear(input, self.mu_w, self.bias)

            # usual evaluation via reparameterization trick (NOT local)
            logsig2_w = self.logsig2_w.clamp(-20, 11)
            if self.random is None:
                self.random = torch.randn_like(self.logsig2_w)
            s2_w = logsig2_w.exp()
            weight = self.mu_w + s2_w.sqrt() * self.random
            return nn.functional.linear(input, weight, self.bias) #+ 1e-8

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.n_in}) -> ({self.n_out})"
