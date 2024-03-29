''' Collection of functions used for training and other things.

@athors = Michel Luchmann, 
          Jad Mathieu Sardain
'''


#####################################
### Imports ###

import torch
from torch import nn
import numpy as np
import os

####################################

def train_loop(dataloader, model, optimizer, loss_dict):
    '''Train loop for training a Bayesian neural network.
    
    @args:
        dataloader: needs to be iterable and provide pytorch
          tensors as an output. Shape of tensors should be
          data.shape = (batch_size, num_input_featues + 1) where
          the last entry represents the truth label
          data[:, -1] = label / trainings-target
        model: Bayesian neural network mode. Needs
          neg_log_likelihood() and KL() method to construct
          loss function
        optimizer: torch.optim object
        loss_dict: dictonrary to save loss-values in. Should
           be initialized with:
             from collections import defaultdict
             loss_dict = defaultdict(list)

    @returns:
        loss_dict
    '''

    size = len(dataloader.dataset)
    model.train()

    loss_tot, kl_tot, neg_log_tot, mse_tot = 0, 0, 0, 0
    mse_norm_tot = 0
    n_batches = 0 # there has to be a more pythonic way
    for batch, data in enumerate(dataloader):

        # Compute prediction and loss
        x = data[:, :-1].float()
        y = data[:, -1]

        neg_log = model.neg_log_likelihood(x, y)
        kl = model.KL()
        loss = neg_log + kl

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute separately, this could be removed for efficieny
        # TODO: implement option to not resample?
        prediction = model.mean(x, n_monte=1)
        sigma2 = model.sigma_stoch2(x, n_monte=1)
        mse = torch.pow(prediction - y, 2).mean()
        mse_normalized = torch.mean(torch.pow((prediction - y), 2) / sigma2)

        # save losses for later plotting
        loss_tot += loss.item()
        kl_tot += kl.item()
        neg_log_tot += neg_log.item()
        mse_tot += mse.item()
        mse_norm_tot += mse_normalized.item()
        n_batches += 1

    loss_tot /= n_batches
    kl_tot /= n_batches
    neg_log_tot /= n_batches
    mse_tot /= n_batches
    mse_norm_tot /= n_batches

    print(f"loss: {loss_tot:>7f} KL: {kl_tot} Neg-log {neg_log_tot} MSE: {mse_tot} MSE-Norm {mse_norm_tot}")

    loss_dict['kl'].append(kl_tot)
    loss_dict['loss_tot'].append(loss_tot)
    loss_dict['neg_log'].append(neg_log_tot)
    loss_dict['mse'].append(mse_tot)
    loss_dict['mse_norm'].append(mse_norm_tot)

    return loss_dict


def val_pass(dataloader, model, loss_dict):
    '''Validation pass for training a Bayesian neural network.
    Should be called after each epoch.
    
    @args:
        dataloader: needs to be iterable and provide pytorch
          tensors as an output. Shape of tensors should be
          data.shape = (batch_size, num_input_featues + 1) where
          the last entry represents the truth label
          data[:, -1] = label / trainings-target
        model: Bayesian neural network mode. Needs
          neg_log_likelihood() and KL() method to construct
          loss function
        loss_dict: dictonrary to save loss-values in. Should
           be initialized with:
             from collections import defaultdict
             loss_dict = defaultdict(list)

    @returns:
        loss_dict
    '''


    loss_tot, kl_tot, neg_log_tot, mse_tot = 0, 0, 0, 0
    mse_norm_tot = 0
    n_batches = 0 # there has to be a more pythonic way

    for batch, data in enumerate(dataloader):

        with torch.no_grad():
            # Compute prediction and loss
            x = data[:, :-1].float()
            y = data[:, -1]

            neg_log = model.neg_log_likelihood(x, y)
            kl = model.KL()
            loss = neg_log + kl

            prediction = model.mean(x, n_monte=1) # inefficient!
            sigma2 = model.sigma_stoch2(x, n_monte=1)
            mse = torch.pow(prediction - y, 2).mean()
            mse_normalized = torch.mean(torch.pow((prediction - y), 2) / sigma2)

            # save losses for later plotting
            loss_tot += loss.item()
            kl_tot += kl.item()
            neg_log_tot += neg_log.item()
            mse_tot += mse.item()
            mse_norm_tot += mse_normalized.item()
            n_batches += 1

    loss_tot /= n_batches
    kl_tot /= n_batches
    neg_log_tot /= n_batches
    mse_tot /= n_batches
    mse_norm_tot /= n_batches

    print(f"validation loss: {loss_tot:>7f} KL: {kl_tot} Neg-log {neg_log_tot} MSE: {mse_tot} MSE-Norm {mse_norm_tot}")

    loss_dict['kl'].append(kl_tot)
    loss_dict['loss_tot'].append(loss_tot)
    loss_dict['neg_log'].append(neg_log_tot)
    loss_dict['mse'].append(mse_tot)
    loss_dict['mse_norm'].append(mse_norm_tot)

    return loss_dict

# TODO
def train_loop_mse(dataloader, model, optimizer, loss_dict):

    size = len(dataloader.dataset)
    model.train()

    loss_val = 0
    n_batches = 0 # there has to be a more pythonic way
    for batch, data in enumerate(dataloader):

        #print(data.shape)

        # Compute prediction and loss
        x = data[:, :-1].float()
        y = data[:, -1]

        pred = model(x)[:, 0]
        loss = torch.pow(pred - y, 2).mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val += loss.item()
        n_batches += 1

    loss_val /= n_batches

    print(f"loss: {loss_val}")

    loss_dict['mse'].append(loss_val)

    return loss_dict


# TODO
def val_pass_mse(dataloader, model, loss_dict):

    size = len(dataloader.dataset)

    loss_val = 0
    n_batches = 0 # there has to be a more pythonic way
    for batch, data in enumerate(dataloader):
        with torch.no_grad():

            # Compute prediction and loss
            x = data[:, :-1].float()
            y = data[:, -1]

            #print(x.shape, y.shape)

            pred = model(x)
            loss = torch.pow(pred[:, 0] - y, 2).mean()
            loss_val += loss.item()
            n_batches += 1


    loss_val /= n_batches

    print(f"validation loss: {loss_val}")
    loss_dict['mse'].append(loss_val)

    return loss_dict


def train_loop_det(dataloader, model, optimizer, loss_dict):

    size = len(dataloader.dataset)
    model.train()

    neg_log_tot, mse_tot, mse_norm_tot = 0, 0, 0
    n_batches = 0 # there has to be a more pythonic way
    for batch, data in enumerate(dataloader):

        # Compute prediction and loss
        x = data[:, :-1].float()
        y = data[:, -1]

        neg_log = model.neg_log_likelihood(x, y)
        loss = neg_log + kl

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = model(x)
        mse = torch.pow(pred[:, 0] - y, 2).mean()
        mse_normalized = torch.mean(torch.pow((pred[:, 0] - y), 2) / pred[:, 1].exp())

        # save losses for later plotting
        neg_log_tot += neg_log.item()
        mse_tot += mse.item()
        mse_norm_tot += mse_normalized.item()
        n_batches += 1

    neg_log_tot /= n_batches
    mse_tot /= n_batches
    mse_norm_tot /= n_batches

    print(f"Neg-log {neg_log_tot} MSE: {mse_tot} MSE-Norm {mse_norm_tot}")

    loss_dict['neg_log'].append(neg_log_tot)
    loss_dict['mse'].append(mse_tot)
    loss_dict['mse_norm'].append(mse_norm_tot)

    return loss_dict


def val_pass_det(dataloader, model, loss_fn, loss_dict):

    size = len(dataloader.dataset)

    neg_log_tot, mse_tot, mse_norm_tot = 0, 0, 0
    n_batches = 0 # there has to be a more pythonic way
    for batch, data in enumerate(dataloader):

        with torch.no_grad():

            # Compute prediction and loss
            x = data[:, :-1].float()
            y = data[:, -1]

            pred = model(x)
            neg_log = loss_fn(pred, y)

            mse = torch.pow(pred[:, 0] - y, 2).mean()
            mse_normalized = torch.mean(torch.pow((pred[:, 0] - y), 2) / pred[:, 1].exp())

            # save losses for later plotting
            neg_log_tot += neg_log.item()
            mse_tot += mse.item()
            mse_norm_tot += mse_normalized.item()
            n_batches += 1

    neg_log_tot /= n_batches
    mse_tot /= n_batches
    mse_norm_tot /= n_batches

    print(f"validation loss: Neg-log {neg_log_tot} MSE: {mse_tot} MSE-Norm {mse_norm_tot}")

    loss_dict['neg_log'].append(neg_log_tot)
    loss_dict['mse'].append(mse_tot)
    loss_dict['mse_norm'].append(mse_norm_tot)

    return loss_dict

def neg_log_gauss(outputs, targets):

    mu = outputs[:, 0]
    logsigma2 = outputs[:, 1]

    out = torch.pow(mu - targets, 2) / (2 * logsigma2.exp()) + 1./2. * logsigma2
    return torch.mean(out)

def create_directory(dir_path):
    '''Create directory with given path.
    If directory already exists, a number will be added to the name.
    If the directory name already ends with a number the next possible number
    which is not yet taken wil be choosen. E.g.:
        dir_path = /path/to/dir45
        #-> dir45 already exists -> creates /path/to/dir46 instead

    The maximu number of directory names this function tries to create 
        is restricted to MAX_TRIALS.
    '''

    # just for safity
    MAX_TRIALS = 30

    # create new directory with name dir_path
    try:
        # Create target Directory
        os.mkdir(dir_path)
        print(f"Directory {dir_path} Created!")
    except FileExistsError:

        # let's not put a while True loop here for safity reasons
        for tries in range(MAX_TRIALS):

            print(f"Directory {dir_path} already exists!")

            if tries == MAX_TRIALS - 1:
                raise AssertionError("Failed sevral times to create a new directory! Stopping the program!")

            # if directory name ends with a digit: Let's extract remove this digit and increase it by one
            # e.g. /path/to/dir23 -> /path/to/dir24
            end_digit = ""
            for i, char in enumerate(dir_path[::-1]):
                if not char.isdigit():
                    idx = i
                    break
                else:
                    print("Dir name ended with digit. Removing digit.")
                    end_digit += char

            if end_digit:
                end_digit = int(end_digit[::-1]) # reversing order
                dir_path = dir_path[:-idx]
            else:
                end_digit = 0

            # increase last digit by one
            new_digit = end_digit + 1
            dir_path = dir_path + str(new_digit)

            # try to create new directory: if fails -> try again with increased last digit
            # otherwise: break loop and return dir_path
            try:
                os.mkdir(dir_path)
                print(f"Creating directory {dir_path}")
                break
            except FileExistsError:
                print(f"New directory suggestion {dir_path} failed!")
                print("Directory already exists!")

    return dir_path


def remove_nans_and_inf(x, replace_value=None):
    '''Removes or replaced inf and NaN values in a given np.array'''

    mask = np.isfinite(x)
    if (~mask).sum() > 0:
        print("Detected {} probelmatic values (NaN or Inf)".format(mask.sum()))

    if replace_value is not None:
        x[~mask] = replace_value
        return x
    else:
        return x[mask]
