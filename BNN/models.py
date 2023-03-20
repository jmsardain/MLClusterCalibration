#########################################
### Imports ###

from vblinear import VBLinear
import torch
from torch import nn
import numpy as np

#########################################

def tanhone(input):
    return 3 * ( torch.tanh(input) + 1 )

class TanhPlusOne(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return tanhone(input) # simply apply already implemented SiLU

class BNN(nn.Module):

    def __init__(self, training_size, inner_layers, input_dim, activation_inner='tanh', out_dim=2, activation_last=None):
        super(BNN, self).__init__()

        self.out_dim = out_dim
        self.training_size = training_size
        self.all_layers = []
        self.vb_layers = []

        # add first and last layer
        inner_layers = [input_dim] + inner_layers
        inner_layers.append(out_dim)

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

    # TODO: do we need this?
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


# TODO: make all of this more efficient:
# only call _compute_predictions once!
class BNN_normal(BNN):

    def __init__(self, training_size, inner_layers, input_dim, activation_inner='tanh', activation_last=None):
        super(BNN_normal, self).__init__(
            training_size,
            inner_layers,
            input_dim,
            activation_inner=activation_inner,
            # need two output-dimensions for mu and sigma of a normal dist
            out_dim=2,
            activation_last=activation_last
        )

        self._mean_values = None
        self._sigma_stoch2 = None
        self._sigma_pred2 = None

    def neg_log_likelihood(self, x, targets):

        outputs = self.forward(x)
        mu = outputs[:, 0]
        logsigma2 = outputs[:, 1]

        out = torch.pow(mu - targets, 2) / (2 * logsigma2.exp()) + 1./2. * logsigma2
        out += 1./2.*np.log(2.*np.pi) # let's add constant to get proper neg log likelihood
        return torch.mean(out)

    def total_loss(self, x, targets):
        loss = neg_log_likelihood(x, targets) + self.KL()
        return loss

    def _compute_predictions(self, x, n_monte=50):

        outputs = []
        for i in range(n_monte):
            # extract prediction for each weight sample
            self.reset_random()
            output = self.forward(x)
            outputs.append(output)

        # dim = (n_monte, batch-size, network-output-dim)
        outputs = torch.stack(outputs, axis=0)
        self._mean_values = torch.mean(outputs[:, :, 0], axis=0)
        self._sigma_stoch2 = torch.mean(outputs[:, :, 1].exp(), axis=0)
        self._sigma_pred2 = torch.var(outputs[:, :, 0], axis=0)

    def mean(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        return self._mean_values

    def mode(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        return self._mean_values

    def median(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        return self._mean_values

    def sigma_stoch2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        return self._sigma_stoch2

    def sigma_pred2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        return self._sigma_pred2

    def sigma_tot2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        sigma_tot2 = self._sigma_pred2**2 + self._sigma_stoch2**2
        return sigma_tot2

    def distribution(self, x, n_monte=50):
        # TODO
        raise NotImplemented("Not yet implemented!")


# TODO: make all of this more efficient:
# only call _compute_predictions once!
class BNN_lognormal(BNN):

    def __init__(self, training_size, inner_layers, input_dim, activation_inner='tanh', activation_last=None):
        super(BNN_lognormal, self).__init__(
            training_size,
            inner_layers,
            input_dim,
            activation_inner=activation_inner,
            out_dim=2,
            activation_last=activation_last
        )

        self._mu = None
        self._sigma2 = None

    def neg_log_likelihood(self, x, targets):

        outputs = self.forward(x)
        mu = outputs[:, 0]
        logsigma2 = outputs[:, 1]

        # parameterization of a log-normal via mu and logsigma2
        # IMPORTANT: mu and sigma are not equivalent to Mean() and Std() of a log-normal
        out = torch.pow(torch.log(targets) - mu, 2) / (2 * logsigma2.exp()) 
        out += 1./2.*logsigma2 + torch.log(targets) 
        out += 1./2.*np.log(2.*np.pi)

        return torch.mean(out)

    def total_loss(self, x, targets):
        loss = neg_log_likelihood(x, targets) + self.KL()
        return loss

    def _compute_predictions(self, x, n_monte=50):

        outputs = []
        for i in range(n_monte):
            self.reset_random()
            output = self.forward(x)
            outputs.append(output)

        # dim = (n_monte, batch-size, network-output-dim)
        outputs = torch.stack(outputs, axis=0)

        self._mu = outputs[:, :, 0]
        self._sigma2 = outputs[:, :, 1].exp()

    def mean(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        mean = torch.exp(self._mu + self._sigma2 / 2.)

        # mean over Monte-Carlo weight samples
        mean = torch.mean(mean, axis=0)
        return mean

    def mode(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        mode = torch.exp(self._mu - self._sigma2)

        # mean over Monte-Carlo weight samples
        # TODO: think about if this is a good idea, avergae median?
        mode = torch.mean(mode, axis=0)
        return mode

    def median(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        median = torch.exp(self._mu)

        # mean over Monte-Carlo weight samples
        # TODO: think about if this is a good idea, avergae median?
        median = torch.mean(median, axis=0)
        return median

    def sigma_stoch2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        # variance of a log-normal dist, from Wikipedia
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        sigma_stoch2 = (torch.exp(self._sigma2) - 1.) * torch.exp(2. * self._mu + self._sigma2)
        sigma_stoch2 = torch.mean(sigma_stoch2, axis=0) # mean over Monte-Carlo weight samples
        return sigma_stoch2

    def sigma_pred2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        # TODO: we could also compute the std of the modes or medians?
        mean = torch.exp(self._mu + self._sigma2 / 2.)
        sigma_pred2 = torch.var(mean, axis=0)

        return sigma_pred2

    def sigma_tot2(self, x, n_monte=50):
        sigma_tot2 = self.sigma_pred2(x, n_monte=n_monte)**2 + self.sigma_stoch2(x, n_monte=n_monte)**2 
        return sigma_tot2

    def distribution(self, x, n_monte=50):
        # TODO
        raise NotImplemented("Not yet implemented!")


# TODO: make all of this more efficient:
# only call _compute_predictions once!
class BNN_normal_mixture(BNN):

    def __init__(self, training_size, inner_layers, input_dim, activation_inner='tanh', n_mixtures=3, activation_last=None):
        super(BNN_normal_mixture, self).__init__(
            training_size,
            inner_layers,
            input_dim,
            activation_inner=activation_inner,
            out_dim=3*n_mixtures,
            activation_last=None
        )

        self.n_mixtures = n_mixtures

        self._mus = None
        self._sigma2s = None
        self._alphas = None

    # overwrite, this is not nice, TODO
    def forward(self, x):
        y = self.model(x)
        y, alphas = y[:, :self.n_mixtures*2], y[:, self.n_mixtures*2:]
        alphas = nn.Softmax(dim=-1)(alphas)
        return torch.cat((y, alphas), axis=1)

    def neg_log_likelihood(self, x, targets):

        outputs = self.forward(x)

        x = targets[:, None]
        mus = outputs[:, :self.n_mixtures] 
        logsigma2s = outputs[:, self.n_mixtures:self.n_mixtures*2]
        alphas = outputs[:, 2*self.n_mixtures:] # have to be between 0 and 1! TODO

        # compute log-gauss for each component
        log_components = -self._neg_log_gauss(x, mus, logsigma2s) + torch.log(alphas)
        neg_log_likelihood = -torch.logsumexp(log_components, dim=-1)

        return torch.mean(neg_log_likelihood)

    def _neg_log_gauss(self,  x, mu, logsigma2):
        ''' 1d log-gauss in logsigma2 parameterisation'''
        out = torch.pow(mu - x, 2) / (2 * logsigma2.exp()) + 1./2. * logsigma2
        out += 1./2.*np.log(2.*np.pi) # constant
        return out

    def total_loss(self, x, targets):
        loss = neg_log_likelihood(x, targets) + self.KL()
        return loss

    def _compute_predictions(self, x, n_monte=50):

        outputs = []
        for i in range(n_monte):
            self.reset_random()
            output = self.forward(x)
            outputs.append(output)

        # dim = (n_monte, batch-size, network-output-dim)
        outputs = torch.stack(outputs, axis=0)

        self._mus = outputs[:, :, 0:self.n_mixtures]
        self._sigma2s = outputs[:, :, self.n_mixtures:2*self.n_mixtures].exp()
        self._alphas = outputs[:, :, self.n_mixtures*2:]

    def mean(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        mean = torch.sum(self._mus * self._alphas, axis=-1)

        # mean over monte-carlo weight samples
        mean = torch.mean(mean, axis=0)
        return mean

    def mode(self, x, n_monte=50, approximation=False):
        self._compute_predictions(x, n_monte=n_monte)

        # compute average parameters
        alphas = torch.mean(self._alphas, axis=0)
        sigma2s = torch.mean(self._sigma2s, axis=0)
        mus = torch.mean(self._mus, axis=0)

        if approximation:
            # check which peak of normal components is the largest:
            #   peak-height = Normal(x=mu) * alpha
            #               = 1/sqrt(2 pi sigma^2) * alpha_i
            norm = alphas * 1. / torch.sqrt(2. * np.pi * sigma2s)
            idxs = torch.argmax(norm, axis=-1)
            mode = mus[(range(len(idxs)), idxs)]
        else:
            # numerical method to check for maxima: just evaluate likelihood for set of points and choose largest

            # select range to check for maxima
            x_min, _ = torch.min(mus, axis=-1)
            x_max, _ = torch.max(mus, axis=-1)
            x_min *= 0.9
            x_max*= 1.1
            
            # create array of ranges to check for maxima
            x_tests = []
            for i in range(x_min.shape[0]):
                device = x_min.get_device()
                x_test = torch.linspace(x_min[i].item(), x_max[i].item(), 1000).to(device)
                x_tests.append(x_test)

            x_test = torch.stack(x_tests, axis=0)

            # reshape parameters for likelihood computations
            x_test_reshaped = x_test[:, :, None]
            mus_reshaped = mus[:, None, :]
            sigma2s_reshaped = sigma2s[:, None, :]
            alphas_reshaped = alphas[:, None, :]

            # compute likelihood, TODO: code dubplication
            log_components = -self._neg_log_gauss(x_test_reshaped, mus_reshaped, torch.log(sigma2s_reshaped))
            log_components += torch.log(alphas_reshaped)
            neg_log_likelihood = -torch.logsumexp(log_components, dim=-1)

            # choose x-point with largest likelihood value
            idxs = torch.argmin(neg_log_likelihood, axis=-1)
            mode = x_test[range(len(idxs)), idxs]
                
        return mode

    def median(self, x, n_monte=50):
        #aself._compute_predictions(n_monte=n_monte)
        raise NotImplemented("Not implemented yet! Have to think about how to computed efficiently!")

    def sigma_stoch2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        # Wikipedia: https://en.wikipedia.org/wiki/Mixture_distribution
        # variance(mixture) = sum_i var_i * alpha_i + sum_i mu_i^2 * alpha_i + sum_i mu_i * alpha_i
        #                   = sum_i var_i * alpha_i + sum_i mu_i^2 * alpha_i + mean
        sigma_stoch2 = torch.sum(self._sigma2s * self._alphas, axis=-1)
        sigma_stoch2 += torch.sum(torch.pow(self._mus, 2) * self._alphas, axis=-1)
        sigma_stoch2 -= torch.pow(torch.sum(self._mus * self._alphas, axis=-1), 2)

        sigma_stoch2 = torch.mean(sigma_stoch2, axis=0)

        return sigma_stoch2

    def sigma_pred2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        # TODO: think about it
        mean = torch.sum(self._mus * self._alphas, axis=-1)

        # var over monte-carlo weight samples
        sigma_pred2 = torch.var(mean, axis=0)

        return sigma_pred2

    def sigma_tot2(self, x, n_monte=50):
        sigma_tot2 = self.sigma_pred2(x, n_monte=n_monte)**2 + self.sigma_stoch2(x, n_monte=n_monte)**2 
        return sigma_tot2

    def distributions(self, x_single_event, ax1, ax2, x_range_in_sigma=3, n_monte=50):
        x_single_event_reshaped = x_single_event[None, :]

        self._compute_predictions(x_single_event_reshaped, n_monte=n_monte)
        
        # compute plotting range
        mean = self.mean(x_single_event_reshaped, n_monte=n_monte)
        var = self.sigma_stoch2(x_single_event_reshaped, n_monte=n_monte)
        x_min = mean - x_range_in_sigma * torch.sqrt(var)
        x_max = mean + (x_range_in_sigma+1) * torch.sqrt(var) # +1 to have more space for the legend
        device = x_min.get_device()
        x_test = torch.linspace(x_min.item(), x_max.item(), 1000).to(device)

        # reshape arrays so likelihood computations work
        x_test_reshaped = x_test[None, :, None]
        mus_reshaped = self._mus
        sigma2s_reshaped = self._sigma2s
        alphas_reshaped = self._alphas

        # compute likelihood
        log_components = -self._neg_log_gauss(x_test_reshaped, mus_reshaped, torch.log(sigma2s_reshaped))
        log_components += torch.log(alphas_reshaped)    
        log_likelihood = torch.logsumexp(log_components, dim=-1)
        likelihood = torch.exp(log_likelihood)

        # compute mode and mean values for plotting
        mode_numerical = self.mode(x_single_event_reshaped, n_monte=n_monte, approximation=False)
        mode_approx = self.mode(x_single_event_reshaped, n_monte=n_monte, approximation=True)
        mean = self.mean(x_single_event_reshaped, n_monte=n_monte)

        # TODO: can this be done nicer?
        mode_numerical = mode_numerical.cpu().detach().numpy()
        mode_approx = mode_approx.cpu().detach().numpy()
        mean = mean.cpu().detach().numpy()

        # average over Bayesian weight samples
        likelihood_avg = likelihood.mean(axis=0).cpu().detach().numpy()
        likelihood_components_avg = torch.exp(log_components).mean(axis=0).cpu().detach().numpy()
        likelihood = likelihood.cpu().detach().numpy()
        x_test = x_test.cpu().detach().numpy()

        # draw indivdual weight samples
        max_draw_plots = np.min([20, likelihood.shape[0]])
        for i in range(max_draw_plots):
            if i == 0:
                ax2.plot(x_test, likelihood[i, :], color="C0", label="Bayesian\nsamples")
            else:
                ax2.plot(x_test, likelihood[i, :], color="C0")

        # plot bayesian averaged distributions
        ax1.plot(x_test, likelihood_avg, color="C1", label="Mixture")
        for i in range(likelihood_components_avg.shape[-1]):
            if i == 0:
                ax1.plot(x_test, likelihood_components_avg[:, i], color="C2", label=" Comp.")
            else:
                ax1.plot(x_test, likelihood_components_avg[:, i], color="C2") # without label

        # plot mode and mean
        ax2.axvline(mode_numerical, label="Mode\n(num.)", linestyle="-", color="C3")
        ax2.axvline(mode_approx, label="Mode\n(approx.)", linestyle=":", color="C3")
        ax2.axvline(mean, label="Mean", linestyle="-", color="C4")

        return x_test, likelihood


# TODO: this has to be rewritten
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



