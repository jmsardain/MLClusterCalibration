''' Implementation of (Bayesian) Neural networks.

There is a standard BNN class and several classes build on top of it
for different types of regression. The main logic for Bayesian neural
networks can be found in vblinear.py. The Bayesian neural networks
implemented here are variational mean-field BNNs. The main assumption
is that the variational distribution (distribution over weights) is a
product of independent Gaussian distributions which is also referred to
as the mean-field approximation.

The interpretation of the BNN output is bound to the loss function
(likelihood) chosen for training.
For instance, if using a normal likelihood, the first output of the
BNN represent the mean value and the second output represent the
logarithm of the Gaussian width. To make it easier to extract the
correct prediction for a specific likelihood function, there are
several subclasses build on top of the BNN class, each connected
with a certain likelihood.
The different types are:
    - BNN_normal: BNN for regression using a normal likelihood
    - BNN_normal_mixture: BNN for regression with normal mixture likelihood.
         Number of mixtures is free parameter.
    - BNN_log_normal: BNN for regression with log-normal likelihood.
    - BNN_log_normal_mixture: BNN for regression with log-normal mixture
        likelihood. Number of mixtures is free parameter.

@athors = Michel Luchmann, 
          Jad Mathieu Sardain
'''

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
    '''Standard BNN class.
    This class only comes with the KL() method which has to be added to the loss
    function. It doesn't have a log_likelihood method like the other BNN subclasses.
    Thus, the loss function has to be constructed outside like in the following
    use case example:

    Code exmaple for training:
        model = BNN(...)
        model.train()
        for input, labels in dataloader:
            prediction = model(input)

            # loss_function = negative log-likelihood
            loss = loss_function(prediction, labels)
            loss += model.KL()

            loss.backward()
            optimizer.step()

    Code example for testing phase:
        num_weight_samples = 50
        for i in range(num_weight_samples):
            model.reset_random()
            prediction = model(input)
        
        torch.mean(prediction, axis=0)

    @args:
        training_size = total number of traning examples (NOT batch size).
            Needed for correct prefactor for KL-divergence
        inner_layers = list of integers to represent number of nodes per layer.
            E.g. [64, 64, 32] -> 3 inner layers with nodes 64, 64 and 32
        input_dim = number of features/dimensions of network input. Shape of input data is
            (batch_size, input_dim)
        activation_inner = activation function of inner layers. Currently implemented are:
            tanh, relu, softplus
        out_dim = number of output dimensions
        activaton_last = optional activation function for last layer. Only applied to first
            output dimension. First output dimension is usually asscociated with the mean prediction.
    '''

    def __init__(self, training_size, inner_layers, input_dim, activation_inner='tanh', out_dim=2, activation_last=None):
        super(BNN, self).__init__()

        self.out_dim = out_dim
        self.training_size = training_size
        self.all_layers = []
        self.vb_layers = []

        # add first and last layer to list with all layers
        inner_layers = [input_dim] + inner_layers
        inner_layers.append(out_dim)

        # construct all layers
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
                    raise NotImplementedError(
                        "Option for activation function of inner layers is not implemented! Given: {}".format(
                        activation_inner))

        # set layer activation function
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
                raise NotImplementedError(
                    "Option for lactivation function of last layer is not implemented! Given {}".format(
                    activation_last))

        self.model = nn.Sequential(*self.all_layers)

    def forward(self, x):
        y = self.model(x)
        if self.last_activation is not None:
            y0, y1 = y[:, 0:1], y[:, 1:]
            y0 = self.last_activation(y0)
            return torch.cat((y0, y1), axis=1)
        else:
            return y

    def set_map(self, map):
        '''MAP=Maximum A Posteriori -> Set each weight to its mean-value, turns off weights
            sampling.
            @args: map = boolean, True -> use MAP method for computing predictions
        '''
        for vb_layer in self.vb_layers:
            vb_layer.map = map

    def reset_random(self):
        ''' Reset random values used to sample network weights in each layer.
        '''
        for layer in self.vb_layers:
            layer.reset_random()
        return

    def KL(self):
        ''' Return total KL-divergence between variational dist and prior.
        '''

        kl = 0
        for vb_layer in self.vb_layers:
            kl += vb_layer.KL()
        return kl / self.training_size


class BNN_normal(BNN):
    '''BNN class specific for regression using a normal / Gaussian likelihood.
       Output of BNN is 2 dimensional. First component is mean value of
       a Normal distribution and second output is log(sigma^2).

       Loss function can be constructed by calling the two methods 
       neg_log_likelihood() and KL() or by just calling total_loss().
       E.g.:
            model = BNN_normal()
            for input, label in dataloader:
                neg_log_likelihood = model.neg_log_likelihood(x, y)
                kl = model.KL()
                loss = neg_log_likelihood + kl
                loss.backward()
                optimizer.step()

       Use Mean / Mode / Meadian method to compute predictions for given input vector x.
       E.g.:
            model = BNN_normal()
            ...
            model.Mean(input_test_data, n_monte=50)
       In the Gaussian case all of these methods result into the same output.

       Uncertainty outputs can be extracted by calling:
            - sigma_stoch2: averaged Gaussian variance. Also referred to as aleatoric uncertainty.
                Formula: E_{q(omega)} (Var) with q(omega) being the variational distribution.
                Should capture noise on labels and input data.
            - sigma_pred2: uncertainty extracted from weight sampling (epistemic uncertainty). 
                Specific to Bayesian neural networks.
                Formula: Var_{q(omega)} (Mean) with q(omega) being the variational distribution.
                Should be highly training size dependent. Captures statistical limitations
                of trainin data.

       @args:
            See BNN class
    '''

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

        self._mus = None
        self._sigma2s = None

    def neg_log_likelihood(self, x, targets):
        ''' Computes average negative log likelihood:
            -log likelihood = 1/M sum_i -log p(y_i | x_i)
            @args:
                x: input data for neural network
                targets: training labels
        '''

        # evaluate network outputs
        outputs = self.forward(x)
        mu = outputs[:, 0]
        log_sigma2 = outputs[:, 1]

        neg_log_prob = -self._log_prob_func(targets, mu, log_sigma2)
        return torch.mean(neg_log_prob) # mean over batch of data

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
        self._mus = outputs[:, :, 0]
        self._sigma2s = outputs[:, :, 1].exp()

    def log_probs(self, y_evaluate, x, n_monte=50):
        '''Computes log_probability for each input data point x and label y_evaluated
        The BNN predicts p(y | x) per event. The log-liklihood is given by:
            log_likelihood = sum_i log( p(y_i | x_i) )
        This method evaluates instead:
            log_probs_{i, j} = log p(y_i | x_j)
        So x and y can be choosen independenlty. The output tensor has one more dimension 
        because no explicit sum over the weight samples is performed:

        @args:
            y_evaluate: 1d array with label values
            x: 1d array with input point values, can have different length then y_evaluate
            n_monte: number of weight samples
        @returns:
            log_probs tensor with shape (len(y_evaluate), n_monte, len(x))
        '''

        self._compute_predictions(x, n_monte=n_monte)

        # self._mus.shape = (n_monte, len(x))
        mu_reshaped = self._mus[None, :, :]
        log_sigma2_reshaped = torch.log(self._sigma2s[None, :, :])
        y_evaluate_reshaped = y_evaluate[:, None, None]

        log_probs = self._log_prob_func(y_evaluate_reshaped, mu_reshaped, log_sigma2_reshaped)

        # log_probs.shape = (len(y_evaluate), n_monte, len(x))
        return log_probs

    def _log_prob_func(self, x, mu, log_sigma2):
        ''' Logarithm of probability density of normal distribution:
            log N_{mu, sigma2}(x)
        '''
        neg_log_probs = torch.pow(x - mu, 2) / (2 * log_sigma2.exp()) 
        neg_log_probs += 1./2.*log_sigma2
        neg_log_probs += 1./2.*np.log(2.*np.pi)
        return -neg_log_probs

    def mean(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        # mean over weight samples
        mean = torch.mean(self._mus, axis=0)
        return mean

    def mode(self, x, n_monte=50):
        return self.mean(x, n_monte=n_monte)

    def median(self, x, n_monte=50):
        return self.mean(x, n_monte=n_monte)

    def sigma_stoch2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        # mean over weight samples
        sigma_stoch2 = torch.mean(self._sigma2s, axis=0)
        return sigma_stoch2

    def sigma_pred2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        sigma_pred2 = torch.var(self._mus, axis=0)
        return sigma_pred2

    def sigma_tot2(self, x, n_monte=50):
        # inefficient, but who cares
        return self.sigma_pred2(x, n_monte=n_monte) + self.sigma_stoch2(x, n_monte=n_monte)

    def draw_distribution(self, x_single_event, ax1, ax2, x_range_in_sigma=3, n_monte=50):
        '''Draw for a single input data point the full predicted distribtuion.

        @args:
            'x_single_event': single data point to draw distribution for. Should have dimensions
                x_single_event.shape = (n_features)
            ax1, ax2: matpltolib.axes objects to draw distributions into
            x_range_in_sigma: x-axis range given in Standard deviatons of the distribution,
                e.g. x_range_in_sigma = 3 -> Draws distribution in range[mean - 3*std, mean + (3+1)*sigma].
                There is more space to the right side for a possible legend.
        '''
        x_single_event_reshaped = x_single_event[None, :]

        # compute plotting range
        mean = self.mean(x_single_event_reshaped, n_monte=n_monte)
        var = self.sigma_stoch2(x_single_event_reshaped, n_monte=n_monte)
        x_min = mean - x_range_in_sigma * torch.sqrt(var)
        x_max = mean + (x_range_in_sigma+1) * torch.sqrt(var) # +1 to have more space for the legend
        device = x_min.get_device()
        x_test = torch.linspace(x_min.item(), x_max.item(), 1000).to(device)

        # reshape arrays so likelihood computations work
        self._compute_predictions(x_single_event_reshaped, n_monte=n_monte)
        x_test_reshaped = x_test[None, :]
        mus_reshaped = self._mus            # shape = (n_monte, 1)
        sigma2s_reshaped = self._sigma2s    # shape = (n_monte, 1)

        # compute likelihood
        log_prob = self._log_prob_func(x_test_reshaped, mus_reshaped, torch.log(sigma2s_reshaped))
        likelihood = torch.exp(log_prob)

        # compute mean
        mean = self.mean(x_single_event_reshaped, n_monte=n_monte)
        mean = mean.cpu().detach().numpy()

        # average over Bayesian weight samples
        likelihood_avg = likelihood.mean(axis=0).cpu().detach().numpy()
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
        ax1.plot(x_test, likelihood_avg, color="C1", label="Average")

        # plot mean
        ax2.axvline(mean, label="Mean", linestyle="-", color="C4")
        ax1.axvline(mean, label="Mean", linestyle="-", color="C4")

        return x_test, likelihood


class BNN_lognormal(BNN):
    '''BNN class specific for regression using a log-normal likelihood.
       Output of BNN is 2 dimensional. First component is the mu parameter
       and the second output is the log(sigma^2) parameter. Be careful, we use
       the standard parameterization. These parameters don't have the
       interpretation of the mean and the variance of the log-normal
       distribution. These parameter represent the mean and the variance
       in 'log-space'. For more information look into [1]. There is a table
       explaining the connection between these parameters and the moments
       of the log-normal distribution.

       Loss function can be constructed by calling the two methods 
       neg_log_likelihood() and KL() or by just calling total_loss().
       E.g.:
            model = BNN_lognormal()
            for input, label in dataloader:
                neg_log_likelihood = model.neg_log_likelihood(x, y)
                kl = model.KL()
                loss = neg_log_likelihood + kl
                loss.backward()
                optimizer.step()

       Use Mean / Mode / Meadian method to compute predictions for given input vector x.
       E.g.:
            model = BNN_lognormal()
            ...
            model.Mean(input_test_data, n_monte=50)
       Be careful, in the log-normal these methods result into different predictions.

       Uncertainty outputs can be extracted by calling:
            - sigma_stoch2: averaged variance. Also referred to as aleatoric uncertainty.
                Formula: E_{q(omega)} (Var) with q(omega) being the variational distribution.
                Should capture noise on labels and input data.
            - sigma_pred2: uncertainty extracted from weight sampling (epistemic uncertainty). 
                Specific to Bayesian neural networks.
                Formula: Var_{q(omega)} (Mean) with q(omega) being the variational distribution.
                Should be highly training size dependent. Captures statistical limitations
                of trainin data.

        [1]: https://en.wikipedia.org/wiki/Log-normal_distribution

       @args:
            See BNN class
    '''

    def __init__(self, training_size, inner_layers, input_dim, activation_inner='tanh', activation_last=None):
        super(BNN_lognormal, self).__init__(
            training_size,
            inner_layers,
            input_dim,
            activation_inner=activation_inner,
            out_dim=2,
            activation_last=activation_last
        )

        self._mus = None
        self._sigma2s = None

    def neg_log_likelihood(self, x, targets):
        ''' Computes average negative log likelihood:
            -log likelihood = 1/M sum_i -log p(y_i | x_i)
            @args:
                x: input data for neural network
                targets: training labels
        '''

        # compute outputs
        outputs = self.forward(x)
        mu = outputs[:, 0]
        log_sigma2 = outputs[:, 1]

        neg_log_prob = -self._log_prob_func(targets, mu, log_sigma2)
        return torch.mean(neg_log_prob) # mean over batch of data

    def total_loss(self, x, targets):
        loss = neg_log_likelihood(x, targets) + self.KL()
        return loss

    def _compute_predictions(self, x, n_monte=50):

        outputs = []
        for i in range(n_monte):
            self.reset_random()
            output = self.forward(x)
            outputs.append(output)

        # outputs.shape = (n_monte, batch-size, network-output-dim)
        outputs = torch.stack(outputs, axis=0)
        self._mus = outputs[:, :, 0]
        self._sigma2s = outputs[:, :, 1].exp()

    def _log_prob_func(self, x, mu, log_sigma2):
        ''' Logarithm of probability density of log-normal distribution:
            log f_{mu, sigma2}(x) with f = density of a log-normal dist
        '''
        # parameterization of a log-normal via mu and logsigma2
        # IMPORTANT: mu and sigma are not equivalent to Mean() and Std() of a log-normal
        neg_log_prob = torch.pow(torch.log(x) - mu, 2) / (2 * log_sigma2.exp()) 
        neg_log_prob += 1./2.*log_sigma2 + torch.log(x) 
        neg_log_prob += 1./2.*np.log(2.*np.pi)
        return -neg_log_prob

    def mean(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        # https://en.wikipedia.org/wiki/Log-normal_distribution
        mean = torch.exp(self._mus + self._sigma2s / 2.)

        # mean over Monte-Carlo weight samples
        mean = torch.mean(mean, axis=0)
        return mean

    def mode(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        # https://en.wikipedia.org/wiki/Log-normal_distribution
        mode = torch.exp(self._mus - self._sigma2s)

        # TODO: This is in principle not correct to do, only an approximation for narrow q(omega)
        # mean over Monte-Carlo weight samples
        mode = torch.mean(mode, axis=0)
        return mode

    def median(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        median = torch.exp(self._mus)

        # TODO: This is in principle not correct to do, only an approximation for narrow q(omega)
        median = torch.mean(median, axis=0)
        return median

    def log_probs(self, y_evaluate, x, n_monte=50):
        '''Computes log_probability for each input data point x and label y_evaluated
        The BNN predicts p(y | x) per event. The log-liklihood is given by:
            log_likelihood = sum_i log( p(y_i | x_i) )
        This method evaluates instead:
            log_probs_{i, j} = log p(y_i | x_j)
        So x and y can be choosen independenlty. The output tensor has one more dimension 
        because no explicit sum over the weight samples is performed:

        @args:
            y_evaluate: 1d array with label values
            x: 1d array with input point values, can have different length then y_evaluate
            n_monte: number of weight samples
        @returns:
            log_probs tensor with shape (len(y_evaluate), n_monte, len(x))
        '''

        self._compute_predictions(x, n_monte=n_monte)

        # self._mus.shape = (n_monte, len(x))
        mu_reshaped = self._mus[None, :, :]
        log_sigma2_reshaped = torch.log(self._sigma2s[None, :, :])
        y_evaluate_reshaped = y_evaluate[:, None, None]

        log_probs = self._log_prob_func(y_evaluate_reshaped, mu_reshaped, log_sigma2_reshaped)

        # log_probs.shape = (len(y_evaluate), n_monte, len(x))
        return log_probs

    def sigma_stoch2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        # variance of a log-normal dist, from Wikipedia
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        sigma_stoch2 = (torch.exp(self._sigma2s) - 1.) * torch.exp(2. * self._mus + self._sigma2s)
        sigma_stoch2 = torch.mean(sigma_stoch2, axis=0) # mean over Monte-Carlo weight samples
        return sigma_stoch2

    def sigma_pred2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        # TODO: we could also compute the std of the modes or medians?
        mean = torch.exp(self._mus + self._sigma2s / 2.)
        sigma_pred2 = torch.var(mean, axis=0)

        return sigma_pred2

    def sigma_tot2(self, x, n_monte=50):
        # inefficient, but who cares
        return self.sigma_pred2(x, n_monte=n_monte) + self.sigma_stoch2(x, n_monte=n_monte)

    def draw_distribution(self, x_single_event, ax1, ax2, x_range_in_sigma=3, n_monte=50):
        '''Draw for a single input data point the full predicted distribtuion.

        @args:
            'x_single_event': single data point to draw distribution for. Should have dimensions
                x_single_event.shape = (n_features)
            ax1, ax2: matpltolib.axes objects to draw distributions into
            x_range_in_sigma: x-axis range given in Standard deviatons of the distribution,
                e.g. x_range_in_sigma = 3 -> Draws distribution in range[mean - 3*std, mean + (3+1)*sigma].
                There is more space to the right side for a possible legend.
        '''
        x_single_event_reshaped = x_single_event[None, :]

        # compute plotting range
        mean = self.mean(x_single_event_reshaped, n_monte=n_monte)
        var = self.sigma_stoch2(x_single_event_reshaped, n_monte=n_monte)
        x_min = mean - x_range_in_sigma * torch.sqrt(var)
        x_max = mean + (x_range_in_sigma+1) * torch.sqrt(var) # +1 to have more space for the legend
        device = x_min.get_device()
        x_test = torch.linspace(x_min.item(), x_max.item(), 1000).to(device)

        # reshape arrays so likelihood computations work
        self._compute_predictions(x_single_event_reshaped, n_monte=n_monte)
        x_test_reshaped = x_test[None, :]
        mus_reshaped = self._mus            # shape = (n_monte, 1)
        sigma2s_reshaped = self._sigma2s    # shape = (n_monte, 1)

        # compute likelihood
        log_prob = self._log_prob_func(x_test_reshaped, mus_reshaped, torch.log(sigma2s_reshaped))
        likelihood = torch.exp(log_prob)

        # compute mean and mode
        mean = self.mean(x_single_event_reshaped, n_monte=n_monte)
        mode = self.mode(x_single_event_reshaped, n_monte=n_monte)
        mean = mean.cpu().detach().numpy()
        mode = mode.cpu().detach().numpy()

        # average over Bayesian weight samples
        likelihood_avg = likelihood.mean(axis=0).cpu().detach().numpy()
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
        ax1.plot(x_test, likelihood_avg, color="C1", label="Average")

        # plot mean and mode
        ax2.axvline(mean, label="Mean", linestyle="-", color="C4")
        ax1.axvline(mean, label="Mean", linestyle="-", color="C4")
        ax2.axvline(mode, label="Mode", linestyle="-", color="C3")
        ax1.axvline(mode, label="Mode", linestyle="-", color="C3")

        return x_test, likelihood


class BNN_normal_mixture(BNN):
    '''BNN class specific for regression using a normal mixture likelihood.
       Output of BNN is (num_mixtures * 3) dimensional:
            out = BNN_normal_mixture(input)
            mus = out[:n_mixtures]
            sigma2s = out[n_mixtures:2*n_mixtures]
            alphas = out[n_mixtures*2:]

            where the PDF is given by:
            Normal-Mixture = Sum_i alphas[i] * Normal(mus[i], sigmas[i])

       Loss function can be constructed by calling the two methods 
       neg_log_likelihood() and KL() or by just calling total_loss().
       E.g.:
            model = BNN_normal_mixture(n_mixtures=3)
            for input, label in dataloader:
                neg_log_likelihood = model.neg_log_likelihood(x, y)
                kl = model.KL()
                loss = neg_log_likelihood + kl
                loss.backward()
                optimizer.step()

       Use Mean / Mode / Meadian method to compute predictions for given input vector x.
       E.g.:
            model = BNN_normal_mixture()
            ...
            model.Mean(input_test_data, n_monte=50)
       Be careful, in a normal mixture model these methods result into different predictions.

       Uncertainty outputs can be extracted by calling:
            - sigma_stoch2: averaged variance. Also referred to as aleatoric uncertainty.
                Formula: E_{q(omega)} (Var) with q(omega) being the variational distribution.
                Should capture noise on labels and input data.
            - sigma_pred2: uncertainty extracted from weight sampling (epistemic uncertainty). 
                Specific to Bayesian neural networks.
                Formula: Var_{q(omega)} (Mean) with q(omega) being the variational distribution.
                Should be highly training size dependent. Captures statistical limitations
                of trainin data.

       @args:
            See BNN class,
            n_mixtures = Number of normal distributions to construct mixture distribution with.
    '''

    def __init__(self, training_size, inner_layers, input_dim, activation_inner='tanh', n_mixtures=3, activation_last=None):

        # not needed here
        del activation_last

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

    def forward(self, x):
        y = self.model(x)
        y, alphas = y[:, :self.n_mixtures*2], y[:, self.n_mixtures*2:]
        alphas = nn.Softmax(dim=-1)(alphas)
        return torch.cat((y, alphas), axis=1)

    def neg_log_likelihood(self, x, targets):
        ''' Computes average negative log likelihood:
            -log likelihood = 1/M sum_i -log p(y_i | x_i)
            @args:
                x: input data for neural network
                targets: training labels
        '''

        # compute output of BNN
        outputs = self.forward(x)
        mus = outputs[:, :self.n_mixtures] 
        log_sigma2s = outputs[:, self.n_mixtures:self.n_mixtures*2]
        alphas = outputs[:, 2*self.n_mixtures:] # have to be between 0 and 1! TODO
        targets_reshaped = targets[:, None]

        neg_log_likelihood = -self._log_prob_func(targets_reshaped, mus, log_sigma2s, alphas)
        return torch.mean(neg_log_likelihood) # mean over batch of data

    def _neg_log_gauss(self,  x, mu, logsigma2):
        ''' 1d log-gauss in logsigma2 parameterisation'''
        out = torch.pow(mu - x, 2) / (2 * logsigma2.exp()) + 1./2. * logsigma2
        out += 1./2.*np.log(2.*np.pi) # constant
        return out

    def _log_prob_func(self, x, mus, log_sigma2s, alphas):
        ''' Logarithm of probability density of log-normal distribution:
            log f_{mu, sigma2}(x) with f = density of a log-normal dist
        '''

        # log_prob is computed as:
        # log(prob) = log(sum_i Exp( log(Normal(mu_i, sigma_i)) + log(alpha_i))).
        # This construction uses the logsumexp method from pytorch which is numerically
        # more stable then just computing: 
        # log(likelihood) = log(sum_i alpha_i * Normal(mu_i, sigma_i) )
        log_components = -self._neg_log_gauss(x, mus, log_sigma2s) + torch.log(alphas)
        log_prob = torch.logsumexp(log_components, dim=-1)

        return log_prob

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

        # mean of mixture distribution is given by weighted mean of each individual dist
        mean = torch.sum(self._mus * self._alphas, axis=-1)

        # mean over monte-carlo weight samples
        mean = torch.mean(mean, axis=0)
        return mean

    def mode(self, x, n_monte=50, approximation=False):
        self._compute_predictions(x, n_monte=n_monte)

        # First computing the Bayesian average is actually only an approximation
        # TODO: Is there a better alternative?
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
            x_max *= 1.1
            
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

            neg_log_prob = -self._log_prob_func(x_test_reshaped, mus_reshaped, torch.log(sigma2s_reshaped), alphas_reshaped)

            # choose x-point with largest likelihood value
            idxs = torch.argmin(neg_log_prob, axis=-1)
            mode = x_test[range(len(idxs)), idxs]
                
        return mode

    def median(self, x, n_monte=50):
        #aself._compute_predictions(n_monte=n_monte)
        raise NotImplementedError("Not implemented yet! Have to think about how to computed efficiently!")

    def sigma_stoch2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        # Variance of a mixture model:
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

        # mean of mixture distribution is given by weighted mean of each individual dist
        mean = torch.sum(self._mus * self._alphas, axis=-1)
        # var over monte-carlo weight samples
        sigma_pred2 = torch.var(mean, axis=0)

        return sigma_pred2

    def sigma_tot2(self, x, n_monte=50):
        sigma_tot2 = self.sigma_pred2(x, n_monte=n_monte) + self.sigma_stoch2(x, n_monte=n_monte)
        return sigma_tot2


    def log_probs(self, y_evaluate, x, n_monte=50):
        '''Computes log_probability for each input data point x and label y_evaluated
        The BNN predicts p(y | x) per event. The log-liklihood is given by:
            log_likelihood = sum_i log( p(y_i | x_i) )
        This method evaluates instead:
            log_probs_{i, j} = log p(y_i | x_j)
        So x and y can be choosen independenlty. The output tensor has one more dimension 
        because no explicit sum over the weight samples is performed:

        @args:
            y_evaluate: 1d array with label values
            x: 1d array with input point values, can have different length then y_evaluate
            n_monte: number of weight samples
        @returns:
            log_probs tensor with shape (len(y_evaluate), n_monte, len(x))
        '''

        self._compute_predictions(x, n_monte=n_monte)

        # self._mus.shape = (n_monte, len(x), 3)
        mus_reshaped = self._mus[None, :, :, :]
        alphas_reshaped = self._alphas[None, :, :, :]
        sigma2s_reshaped = self._sigma2s[None, :, :, :]
        y_evaluate_reshaped = y_evaluate[:, None, None, None]

        log_probs = self._log_prob_func(y_evaluate_reshaped, mus_reshaped, torch.log(sigma2s_reshaped), alphas_reshaped)

        # log_likelihood.shape = (len(y_evaluate), n_monte, len(x))
        return log_probs


    def draw_distribution(self, x_single_event, ax1, ax2, x_range_in_sigma=3, n_monte=50):
        '''Draw for a single input data point the full predicted distribtuions.

        @args:
            'x_single_event': single data point to draw distribution for. Should have dimensions
                x_single_event.shape = (n_features)
            ax1, ax2: matpltolib.axes objects to draw distributions into
            x_range_in_sigma: x-axis range given in Standard deviatons of the distribution,
                e.g. x_range_in_sigma = 3 -> Draws distribution in range[mean - 3*std, mean + (3+1)*sigma].
                There is more space to the right side for a possible legend.
        '''
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
        mus_reshaped = self._mus            # shape = (n_monte, 1, 3)
        sigma2s_reshaped = self._sigma2s    # shape = (n_monte, 1, 3)
        alphas_reshaped = self._alphas      # shape = (n_monte, 1, 3)

        # compute likelihood
        # not calling self._log_prob_func() here to get individual components as well
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

class BNN_log_normal_mixture(BNN):
    '''BNN class specific for regression using a log-normal mixture likelihood.
       Output of BNN is (num_mixtures * 3) dimensional:
            out = BNN_log_normal_mixture(input)
            mus = out[:n_mixtures]
            sigma2s = out[n_mixtures:2*n_mixtures]
            alphas = out[n_mixtures*2:]

            where the PDF is given by:
            Normal-Mixture = Sum_i alphas[i] * LogNormal(mus[i], sigmas[i])

       Loss function can be constructed by calling the two methods 
       neg_log_likelihood() and KL() or by just calling total_loss().
       E.g.:
            model = BNN_log_normal_mixture(n_mixtures=3)
            for input, label in dataloader:
                neg_log_likelihood = model.neg_log_likelihood(x, y)
                kl = model.KL()
                loss = neg_log_likelihood + kl
                loss.backward()
                optimizer.step()

       Use Mean / Mode / Meadian method to compute predictions for given input vector x.
       E.g.:
            model = BNN_log_normal_mixture()
            ...
            model.Mean(input_test_data, n_monte=50)
       Be careful, for a log-normal mixture model these methods result into different predictions.

       Uncertainty outputs can be extracted by calling:
            - sigma_stoch2: averaged variance. Also referred to as aleatoric uncertainty.
                Formula: E_{q(omega)} (Var) with q(omega) being the variational distribution.
                Should capture noise on labels and input data.
            - sigma_pred2: uncertainty extracted from weight sampling (epistemic uncertainty). 
                Specific to Bayesian neural networks.
                Formula: Var_{q(omega)} (Mean) with q(omega) being the variational distribution.
                Should be highly training size dependent. Captures statistical limitations
                of trainin data.

        [1]: https://en.wikipedia.org/wiki/Log-normal_distribution
        [2]: https://en.wikipedia.org/wiki/Mixture_model

       @args:
            See BNN class,
            n_mixtures = Number of normal distributions to construct mixture distribution with.
    '''

    def __init__(self, training_size, inner_layers, input_dim, activation_inner='tanh', n_mixtures=3, activation_last=None):

        # not needed here
        del activation_last

        super(BNN_log_normal_mixture, self).__init__(
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

    def forward(self, x):
        y = self.model(x)
        y, alphas = y[:, :self.n_mixtures*2], y[:, self.n_mixtures*2:]
        alphas = nn.Softmax(dim=-1)(alphas)
        return torch.cat((y, alphas), axis=1)

    def neg_log_likelihood(self, x, targets):
        ''' Computes average negative log likelihood:
            -log likelihood = 1/M sum_i -log p(y_i | x_i)
            @args:
                x: input data for neural network
                targets: training labels
        '''

        # compute output of BNN
        outputs = self.forward(x)
        mus = outputs[:, :self.n_mixtures] 
        log_sigma2s = outputs[:, self.n_mixtures:self.n_mixtures*2]
        alphas = outputs[:, 2*self.n_mixtures:] # have to be between 0 and 1! TODO
        targets_reshaped = targets[:, None]

        neg_log_likelihood = -self._log_prob_func(targets_reshaped, mus, log_sigma2s, alphas)
        return torch.mean(neg_log_likelihood) # mean over batch of data

    def _log_prob_func(self, x, mus, log_sigma2s, alphas):
        ''' Logarithm of probability density of log-normal distribution:
            log f_{mu, sigma2}(x) with f = density of a log-normal dist
        '''

        # log_prob is computed as:
        # log(prob) = log(sum_i Exp( log(Normal(mu_i, sigma_i)) + log(alpha_i))).
        # This construction uses the logsumexp method from pytorch which is numerically
        # more stable then just computing: 
        # log(likelihood) = log(sum_i alpha_i * Log-Normal(mu_i, sigma_i) )
        log_components = -self._neg_log_log_normal(x, mus, log_sigma2s) + torch.log(alphas)
        log_prob = torch.logsumexp(log_components, dim=-1)

        return log_prob


    def _neg_log_log_normal(self, x, mu, log_sigma2):
        # parameterization of a log-normal via mu and logsigma2
        # IMPORTANT: mu and sigma are not equivalent to Mean() and Std() of a log-normal
        neg_log_prob = torch.pow(torch.log(x) - mu, 2) / (2 * log_sigma2.exp()) 
        neg_log_prob += 1./2.*log_sigma2 + torch.log(x) 
        neg_log_prob += 1./2.*np.log(2.*np.pi)
        return neg_log_prob


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


    def _means(self):
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        mean = torch.exp(self._mus + self._sigma2s / 2.)
        return mean

    def _modes(self):
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        mode = torch.exp(self._mus - self._sigma2s)
        return mode

    def _medians(self):
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        median = torch.exp(self._mus)
        return median

    def _var(self):
        # variance of a log-normal dist, from Wikipedia
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        var = (torch.exp(self._sigma2s) - 1.) * torch.exp(2. * self._mus + self._sigma2s)
        return var


    def mean(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)

        means = self._means()

        # mean of mixture distribution is given by weighted mean of each individual dist
        mean = torch.sum(means * self._alphas, axis=-1)

        # mean over monte-carlo weight samples
        mean = torch.mean(mean, axis=0)
        return mean

    def mode(self, x, n_monte=50, approximation=False):

        self._compute_predictions(x, n_monte=n_monte)

        # First computing the Bayesian average is actually only an approximation
        # TODO: Is there a better alternative?
        # compute average parameters
        alphas = torch.mean(self._alphas, axis=0)
        sigma2s = torch.mean(self._sigma2s, axis=0)
        mus = torch.mean(self._mus, axis=0)
        modes = torch.mean(self._modes(), axis=0)

        if approximation:
            # check which peak of lognormal components is the largest:
            #   peak-height = LogNormal(x=mode) * alpha
            norm = torch.exp(
                      -self._neg_log_log_normal(
                          modes,
                          mus,
                          torch.log(sigma2s)))
            norm *= alphas
            idxs = torch.argmax(norm, axis=-1)
            mode = modes[(range(len(idxs)), idxs)]
        else:
            # numerical method to check for maxima: just evaluate likelihood for set of points and choose largest

            # select range to check for maxima
            means = self.mean(x, n_monte=n_monte)
            std = torch.sqrt(self.sigma_stoch2(x, n_monte=n_monte))
            # +epsilon=1e-4 to not pass zero which would give NaN
            x_min, _ = torch.max(torch.stack([means - 4.*std, torch.zeros_like(means) + 1e-4], axis=0), axis=0)
            x_max = means + 4.*std
            
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

            neg_log_prob = -self._log_prob_func(x_test_reshaped, mus_reshaped, torch.log(sigma2s_reshaped), alphas_reshaped)

            # choose x-point with largest likelihood value
            idxs = torch.argmin(neg_log_prob, axis=-1)
            mode = x_test[range(len(idxs)), idxs]
                
        return mode

    def median(self, x, n_monte=50):
        #aself._compute_predictions(n_monte=n_monte)
        raise NotImplementedError("Not implemented yet! Have to think about how to computed efficiently!")

    def sigma_stoch2(self, x, n_monte=50):

        self._compute_predictions(x, n_monte=n_monte)
        var = self._var()

        # Variance of a mixture model:
        # Wikipedia: https://en.wikipedia.org/wiki/Mixture_distribution
        # variance(mixture) = sum_i var_i * alpha_i + sum_i mu_i^2 * alpha_i + sum_i mu_i * alpha_i
        #                   = sum_i var_i * alpha_i + sum_i mu_i^2 * alpha_i + mean
        sigma_stoch2 = torch.sum(var * self._alphas, axis=-1)
        sigma_stoch2 += torch.sum(torch.pow(self._mus, 2) * self._alphas, axis=-1)
        sigma_stoch2 -= torch.pow(torch.sum(self._mus * self._alphas, axis=-1), 2)

        sigma_stoch2 = torch.mean(sigma_stoch2, axis=0)

        return sigma_stoch2

    def sigma_pred2(self, x, n_monte=50):
        self._compute_predictions(x, n_monte=n_monte)
        means = self._means()

        # mean of mixture distribution is given by weighted mean of each individual dist
        mean = torch.sum(means * self._alphas, axis=-1)
        # var over monte-carlo weight samples
        sigma_pred2 = torch.var(mean, axis=0)

        return sigma_pred2

    def sigma_tot2(self, x, n_monte=50):
        sigma_tot2 = self.sigma_pred2(x, n_monte=n_monte) + self.sigma_stoch2(x, n_monte=n_monte)
        return sigma_tot2


    def log_probs(self, y_evaluate, x, n_monte=50):
        '''Computes log_probability for each input data point x and label y_evaluated
        The BNN predicts p(y | x) per event. The log-liklihood is given by:
            log_likelihood = sum_i log( p(y_i | x_i) )
        This method evaluates instead:
            log_probs_{i, j} = log p(y_i | x_j)
        So x and y can be choosen independenlty. The output tensor has one more dimension 
        because no explicit sum over the weight samples is performed:

        @args:
            y_evaluate: 1d array with label values
            x: 1d array with input point values, can have different length then y_evaluate
            n_monte: number of weight samples
        @returns:
            log_probs tensor with shape (len(y_evaluate), n_monte, len(x))
        '''

        self._compute_predictions(x, n_monte=n_monte)

        # self._mus.shape = (n_monte, len(x), 3)
        mus_reshaped = self._mus[None, :, :, :]
        alphas_reshaped = self._alphas[None, :, :, :]
        sigma2s_reshaped = self._sigma2s[None, :, :, :]
        y_evaluate_reshaped = y_evaluate[:, None, None, None]

        log_probs = self._log_prob_func(y_evaluate_reshaped, mus_reshaped, torch.log(sigma2s_reshaped), alphas_reshaped)

        # log_likelihood.shape = (len(y_evaluate), n_monte, len(x))
        return log_probs


    def draw_distribution(self, x_single_event, ax1, ax2, x_range_in_sigma=3, n_monte=50):
        '''Draw for a single input data point the full predicted distribtuions.

        @args:
            'x_single_event': single data point to draw distribution for. Should have dimensions
                x_single_event.shape = (n_features)
            ax1, ax2: matpltolib.axes objects to draw distributions into
            x_range_in_sigma: x-axis range given in Standard deviatons of the distribution,
                e.g. x_range_in_sigma = 3 -> Draws distribution in range[mean - 3*std, mean + (3+1)*sigma].
                There is more space to the right side for a possible legend.
        '''
        x_single_event_reshaped = x_single_event[None, :]

        self._compute_predictions(x_single_event_reshaped, n_monte=n_monte)
        
        # compute plotting range
        x_range_in_sigma = 2
        mean = self.mean(x_single_event_reshaped, n_monte=n_monte)
        var = self.sigma_stoch2(x_single_event_reshaped, n_monte=n_monte)
        x_min = mean - x_range_in_sigma * torch.sqrt(var)
        x_max = mean + x_range_in_sigma * torch.sqrt(var)
        device = x_min.get_device()
        x_test = torch.linspace(x_min.item(), x_max.item(), 1000).to(device)

        # reshape arrays so likelihood computations work
        x_test_reshaped = x_test[None, :, None]
        mus_reshaped = self._mus            # shape = (n_monte, 1, 3)
        sigma2s_reshaped = self._sigma2s    # shape = (n_monte, 1, 3)
        alphas_reshaped = self._alphas      # shape = (n_monte, 1, 3)

        # compute likelihood
        # not calling self._log_prob_func() here to get individual components as well
        log_components = -self._neg_log_log_normal(x_test_reshaped, mus_reshaped, torch.log(sigma2s_reshaped))
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



