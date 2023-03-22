""" Training a BNN for regression.

Dataset represents a calibration task. Input data is a set of high level features
extracted from the detector simulations. Training target is the response defined as:
    R = E_truth / E_cluster
where E_cluster is the naive energy stored in a cluster and E_truth is the Monte
Carlo truth energy.

This code requires three (training, test, validation) statistically independent 
already preprocessed datasets.

@authors: Michel Luchmann
          Jad Mathieu Sardain
"""

####################################
### Imports ###

from vblinear import VBLinear
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
from collections import defaultdict
import argparse
from scipy.stats import norm

from utils import *
from models import *

###################################
### MPL SETTINGS ###

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

plt.style.use("./plotting.mplstyle")

#####################################

def main():

    ##########################################################
    ### Training parameters and hyperparameters ###

    parser = argparse.ArgumentParser('Train a BNN!')

    # TODO: add infos
    parser.add_argument(
        '--output_path',
        default="/remote/gpu03/luchmann/ML_Amplitudes/BNN/output/zjets_first_test_new_new_new_new_new",
        type=str
    )
    parser.add_argument(
        '--data_path_train',
        default="/remote/gpu03/luchmann/ML_Amplitudes/BNN/dataset/Momenta_2_6__G__G__e-__e+__G__G__d__db_2M_train_ampl_prep_log_removed_negative_events_new.npy",
        type=str
    )
    parser.add_argument(
        '--data_path_test',
        default="/remote/gpu03/luchmann/ML_Amplitudes/BNN/dataset/Momenta_2_6__G__G__e-__e+__G__G__d__db_2M_test_ampl_prep_log_removed_negative_events_new.npy",
        type=str
    )
    parser.add_argument(
        '--data_path_val',
        default="/remote/gpu03/luchmann/ML_Amplitudes/BNN/dataset/Momenta_2_6__G__G__e-__e+__G__G__d__db_2M_val_ampl_prep_log_removed_negative_events_new.npy",
        type=str
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float
    )
    parser.add_argument(
        '--batch_size', '--bs',
        default=128,
        type=int
    )
    parser.add_argument(
        '--test_batch_size',
        default=8192,
        type=int
    )
    parser.add_argument(
        '--epochs',
        default=50,
        type=int
    )
    parser.add_argument(
        '--n_monte',
        default=50,
        type=int
    )
    parser.add_argument(
        '--train_size',
        default=10000,
        type=int
    )
    parser.add_argument(
        '--val_size',
        default=1000,
        type=int
    )
    parser.add_argument(
        '--test_size',
        default=10000,
        type=int
    )
    parser.add_argument(
        '--save_weights_iter',
        default=-1,
        type=int
    )
    parser.add_argument(
        '--layer', '--layers',
        default=[50, 50, 50],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--activation_inner',
        default='tanh',
        type=str,
    )
    parser.add_argument(
        '--activation_last',
        default=None,
        type=str,
    )

    parser.add_argument(
        '--explanation',
       default="",
       nargs="+",
       type=str
    )

    parser.add_argument(
        '--target',
        default="resp", # ME, FULL,
        type=str,
    )
    parser.add_argument(
        '--bayesian',
        default=True,
        type=bool,
    )

    parser.add_argument(
        '--likelihood',
        default='normal',
        type=str,
    )

    parser.add_argument(
        '--prediction',
        default='mean',
        type=str,
    )

    args = parser.parse_args()

    ##############################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # If True console output is redirected to the txt files std.out and err.out
    # created inside of the output directory
    redirect_outputs = False

    # default number of mixtures of normal-mixture model
    n_mixtures_default = 2

    ########################################

    dir_path = create_directory(args.output_path)

    if (redirect_outputs):
        sys.stdout = open(dir_path + "/std.out", 'w')
        sys.stderr = open(dir_path + "/err.out", 'w')

    ##############################
    # set up datasets and dataloaders

    # train dataset
    dataset_train = np.load(args.data_path_train)
    x_train = dataset_train[:args.train_size, 1:]
    y_train = dataset_train[:args.train_size, 0]
    data_train = np.concatenate([x_train, y_train[:, None]], axis=-1)
    data_train = torch.from_numpy(data_train).to(device)
    print(f"Training dataset size {y_train.shape[0]}")

    # val dataset
    dataset_val = np.load(args.data_path_val)
    x_val = dataset_val[:args.val_size, 1:]
    y_val = dataset_val[:args.val_size, 0] 
    data_val = np.concatenate([x_val, y_val[:, None]], axis=-1)
    data_val = torch.from_numpy(data_val).to(device)
    print(f"Validation dataset size {y_val.shape[0]}")

    # test dataset
    dataset_test = np.load(args.data_path_test)
    x_test = dataset_test[:args.test_size, 1:]
    y_test = dataset_test[:args.test_size, 0]
    data_test = np.concatenate([x_test, y_test[:, None]], axis=-1)
    data_test = torch.from_numpy(data_test).to(device)
    print(f"Test dataset size {y_test.shape[0]}")

    # data loaders
    train_dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(data_val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(data_test, batch_size=args.test_batch_size, shuffle=False)

    input_dim = x_train.shape[1]
    print(f"Number of input features {input_dim}")

    ###############################
    # set up neural network for training

    if args.bayesian:
        if args.likelihood.lower() == "normal":
            model = BNN_normal(
                args.train_size,
                args.layer,
                input_dim,
                activation_inner=args.activation_inner,
                activation_last=args.activation_last
            ).to(device)

        elif args.likelihood.lower() in ["lognormal", "log-normal", "log_normal"]:
            model = BNN_lognormal(
                args.train_size,
                args.layer,
                input_dim,
                activation_inner=args.activation_inner,
                activation_last=args.activation_last
            ).to(device)

        elif (args.likelihood.lower().startswith("mixturenormal") or
          args.likelihood.lower().startswith("mixture_normal") or
          args.likelihood.lower().startswith("mixture-normal")):

            # extract number of gaussian mixtures from input string
            # e.g.: 'mixturenormal(3)' -> 3
            likelihood_string = args.likelihood.replace(" ", "") # remove white spave if any
            if likelihood_string.endswith(")"):
                digit = args.likelihood[-2]
                if digit.isdigit():
                    n_mixtures = int(digit)
                    print(f"Number of mixture components to be used {n_mixtures}")
                else:
                    print(f"Failed to infere number of mixture for Normal mixture from input string!"
                          f" Given {likelihood_string} -> {digit}"
                          f" Using default value of {n_mixtures_default}!")
                    n_mixtures = n_mixtures_default
            else:
                print(f"Number of mixture components not specified. Using default value of {n_mixtures_default}")
                n_mixtures = n_mixtures_default

            model = BNN_normal_mixture(
                args.train_size,
                args.layer,
                input_dim,
                n_mixtures=n_mixtures,
                activation_inner=args.activation_inner,
                activation_last=args.activation_last
            ).to(device)

        else:
            raise NotImplemented(f"Option for args.likelihood is not implemented! Given {args.likelihood}")

    else:
        # TODO: re-implement these options as well
        raise NotImplemented(f"This has to be change!")

        if args.likelihood.lower() == "normal":
            model = NN(
                args.layer,
                input_dim,
                activation_inner=args.activation_inner,
                activation_last=args.activation_last,
                out_dim=2,
            ).to(device)


        elif args.likelihood.lower() in ["lognormal", "log-normal", "log_normal"]:
            pass
        elif args.likelihood.lower() in ["mixture-normal", "mixturenormal", "mixture_normal"]:
            pass
        else:
            raise NotImplemented(f"Option for args.likelihood is not implemented! Given {args.likelihood}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print("Number of parameters of model: {}".format(n_params))

    ##################################
    # set up optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # save all hyperparameters in text file
    with open(dir_path + "/args.txt", mode="w") as f:
        print("-"*100)
        print("Arguments:")
        for key, item in vars(args).items():
            print("{:<40} {:<40}".format(key, str(item)))
            out = "{:<40} {:<40}\n".format(key, str(item))
            f.write(out)

    ################################
    # training

    print(f"Start training with {args.epochs} epochs...")
    loss_dict = defaultdict(list)
    loss_val_dict = defaultdict(list)
    for t in range(args.epochs):
        print(f"--------------------------------\nEpoch {t+1}")

        # TODO: this is ugly, remove if condition if possible
        if args.bayesian:
            loss_dict = train_loop(train_dataloader, model, optimizer, loss_dict)
            loss_val_dict = val_pass(val_dataloader, model, loss_val_dict)
        else:
            loss_dict = train_loop_det(train_dataloader, model, optimizer, loss_dict)
            loss_val_dict = val_pass_det(val_dataloader, model, loss_val_dict)

        # save model weights
        if (args.save_weights_iter > 0) and (t % args.save_weights_iter == 0):
            checkpoint_path = "{}/state_{}".format(dir_path, t)
            torch.save(
                {
                    'epoch': t+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                checkpoint_path
            )
            print("Saving model state of epoch {} into {}".format(t+1, checkpoint_path))

    ##################################
    # training finished, saving things

    print("Training finished!")

    # save model
    if (args.save_weights_iter > 0):
            checkpoint_path = "{}/final".format(dir_path)
            torch.save(
                {
                    'epoch': t+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                checkpoint_path
            )
            print("Saving final into {}".format(t+1, checkpoint_path))

    # save loss values
    with open(dir_path + "/losses.txt", mode="w") as f:

        header = "{:<5} {:<6}".format("", "epoch")
        for key in loss_dict.keys():
            header += "{:<15}".format(key)

        f.write(header + "\n")

        for i in range(len(loss_dict[list(loss_dict.keys())[0]])):
            out = "{:<5} {:<6}".format("trn", i)
            for key in loss_dict.keys():
                out += "{:<15.5f}".format(loss_dict[key][i])

            f.write(out + "\n")

            out = "{:<5} {:<6}".format("val", i)
            for key in loss_dict.keys():
                out += "{:<15.5f}".format(loss_val_dict[key][i])

            f.write(out + "\n")

    ############################################
    ### Plotting loss as function of epochs ###

    with PdfPages(dir_path + "/losses.pdf") as pdf:

        # loop over different contributions to the loss function
        for key in loss_dict.keys():

            loss_list = np.array(loss_dict[key])
            loss_val_list = np.array(loss_val_dict[key])

            ### linear scale plot
            fig = plt.figure(figsize=[5.5, 5])
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(1, len(loss_list)+1, 1), loss_list, label=key)
            ax.plot(np.arange(1, len(loss_val_list)+1, 1), loss_val_list, label=key + " (validation)")
            ax.legend(frameon=False)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss: " + key)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            ### log scale plot
            fig = plt.figure(figsize=[5.5, 5])
            ax = fig.add_subplot(1, 1, 1)

            # shift to positive value for putting it on a log-scale
            if np.min(loss_list) < 0:
                loss_list_upscaled = loss_list + np.abs(np.min(loss_list))*1.1
                loss_val_list_upscaled = loss_val_list + np.abs(np.min(loss_list))*1.1 # TODO
            else:
                loss_list_upscaled = loss_list
                loss_val_list_upscaled = loss_val_list

            ax.plot(np.arange(1, len(loss_list_upscaled)+1, 1), loss_list_upscaled, label=key + " (upscaled)")
            ax.plot(np.arange(1, len(loss_val_list_upscaled)+1, 1), loss_val_list_upscaled, label=key + " (upscaled, validation)")
            ax.legend(frameon=False)
            ax.set_yscale('log')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss: " + key)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    #####################################################
    ### Evaluate predictions and uncetrainties of BNN ###

    sigma_stochs, sigma_tots, sigma_preds = [], [], []
    predictions, y, x_tests = [], [], []
    log_probs = []
    for batch, data in enumerate(test_dataloader):

        x_test = data[:, :-1].float()
        y_test = data[:, -1]

        if args.bayesian:

            # compute predictions
            if args.prediction.lower() == "mean":
                prediction = model.mean(x_test, args.n_monte).cpu().detach().numpy()
            elif args.prediction.lower() == "mode":
                prediction = model.mode(x_test, args.n_monte).cpu().detach().numpy()
            elif args.prediction.lower() == "median":
                prediction = model.median(x_test, args.n_monte).cpu().detach().numpy()
            else:
                raise NotImplementedError(f"Option for args.prediction not implemented! Given {args.prediction}")
    
            # compute uncertainties
            sigma_stoch = np.sqrt(model.sigma_stoch2(x_test, args.n_monte).cpu().detach().numpy())
            sigma_pred = np.sqrt(model.sigma_pred2(x_test, args.n_monte).cpu().detach().numpy())
            sigma_tot = np.sqrt(model.sigma_tot2(x_test, args.n_monte).cpu().detach().numpy())

            # compute log_probabilities for plotting full predicted distributions
            device = data.get_device()
            y_draw = torch.linspace(0, 4, 100).to(device)
            log_prob = model.log_probs(y_draw, x_test, n_monte=50).cpu().detach().numpy()
            log_probs.append(log_prob)
            y_draw = y_draw.cpu().detach().numpy()

            # append everything into lists
            sigma_stochs.append(sigma_stoch)
            sigma_preds.append(sigma_pred)
            sigma_tots.append(sigma_tot)
            predictions.append(prediction)

            # this is a bit useless because we could instead use the original x_test without a data loader
            # however, this is saver in the case we use a dataloader which shuffles the batches
            y.append(y_test.cpu().detach().numpy())
            x_tests.append(x_test.cpu().detach().numpy())

        else:
            # TODO: rewrite
            raise NotImplementedError("Not implemented!")

    y = np.concatenate(y)
    sigma_stoch = np.concatenate(sigma_stochs)
    sigma_pred = np.concatenate(sigma_preds)
    sigma_tot = np.concatenate(sigma_tots)
    prediction = np.concatenate(predictions)
    x_test = np.concatenate(x_tests)

    log_prob = np.concatenate(log_probs, axis=-1)

    print(f"Minimum prediction {np.min(prediction)}")
    print(f"Mean sigma_stoch {np.mean(sigma_stoch)}")
    print(f"Mean sigma_pred {np.mean(sigma_pred)}")
    print(f"Mean sigma_tot {np.mean(sigma_tot)}")

    ########################################################
    ### Additional plot settings ###
     
    # labels
    label_r_truth = r"$R^{\mathrm{truth}}$"
    label_r_pred = r"$R^{\mathrm{BNN}}$"
    label_e_truth = r"$E^{\mathrm{truth}}$"
    label_e_pred = r"$E^{\mathrm{BNN}}$"
    label_yaxis = r"Frequency"
    units_energy = "[GeV]"

    # colors
    color_truth = "C1" 
    color_pred = "C0"

    if args.prediction.lower() == "mean":
        # [:-1] -> remove $ sign, very fine tuned
        label_r_pred = label_r_pred[:-1] + r"_{\mathrm{mean}}$"
        label_e_pred = label_e_pred[:-1] + r"_{\mathrm{mean}}$"
    elif args.prediction.lower() == "mode":
        label_r_pred = label_r_pred[:-1] + r"_{\mathrm{mode}}$"
        label_e_pred = label_e_pred[:-1] + r"_{\mathrm{mode}}$"
    elif args.prediction.lower() == "median":
        label_r_pred = label_r_pred[:-1] + r"_{\mathrm{median}}$"
        label_e_pred = label_e_pred[:-1] + r"_{\mathrm{median}}$"


    ########################################################
    ### Plots for predicted distributions ###
     
    # The BNN predicts an entire distribution over possible labels y for each input data point x
    # Let's plot this for a bunch of examples with the corresponding truth labels

    num_draw = 20
    plot_path_distributions = dir_path + "/" + "full_distribution_examples.pdf"
    if args.bayesian:

        # get one batch of test data
        data = next(iter(test_dataloader))
        x_one_batch = data[:, :-1].float()
        y_one_batch = data[:, -1]

        # select random indices
        idxs = np.arange(0, len(x_one_batch), 1)
        np.random.shuffle(idxs)
        idxs = idxs[:num_draw]

        with PdfPages(plot_path_distributions) as pdf:
            for i, idx in enumerate(idxs):
                print(f"Drawing full distribution for index {idx} ({i+1}/{len(idxs)})")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[5.5*2, 5])

                # draw single distribution
                # could be parallized. However, plotting has to be done non-parallel anyway
                _, _ = model.draw_distribution(x_one_batch[idx], ax1, ax2)
                y_single_event = y_one_batch[idx].cpu().detach().numpy()
                ax1.axvline(y_single_event, linestyle=":", color="black", label=label_r_truth)
                ax1.set_xlabel("R")
                ax1.set_ylabel("Normalized")
                ax1.legend(frameon=False)

                ax2.axvline(y_single_event, linestyle=":", color="black", label=label_r_truth)
                ax2.set_xlabel("R")
                ax2.set_ylabel("Normalized")
                ax2.legend(frameon=False)

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            print(f"Saved into {plot_path_distributions}!")

    ###########################################
    ### Plots for performance evaluation ###

    plot_path = dir_path + "/" + "performance.pdf"

    # binning
    n_bins = 50

    #####################
    ### Compute pulls ###

    # compute pulls and mse
    mse_i = np.where(y!=0, (prediction - y) / y, np.zeros_like(y))
    mse_norm_i_v1 = (prediction - y) / sigma_tot
    mse_norm_i_v2 = (prediction - y) / sigma_pred
    mse_norm_i_v3 = (prediction - y) / sigma_stoch

    print("MSE...")
    mse_i = remove_nans_and_inf(mse_i, replace_value=None)
    print("MSE norm v1...")
    mse_norm_i_v1 = remove_nans_and_inf(mse_norm_i_v1, replace_value=None)
    print("MSE norm v2 ...")
    mse_norm_i_v2 = remove_nans_and_inf(mse_norm_i_v2, replace_value=None)
    print("MSE norm v3...")
    mse_norm_i_v3 = remove_nans_and_inf(mse_norm_i_v3, replace_value=None)


    with PdfPages(plot_path) as pdf:

        ###########################
        ### r-value plots ###

        ### Plot: 1d histogram, filled=R-values, linear-scale
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        bins = np.linspace(0, 10, n_bins)
        xlim = [bins[0], bins[-1]]
        bin_width = bins[1] - bins[0]
        _, bin_edges, _ = ax.hist(
            np.clip(y, xlim[0] + bin_width*0.5, xlim[1] - bin_width*0.5),
            bins=bins, histtype="step", label=label_r_truth, color=color_truth
        )
        ax.hist(
            np.clip(prediction, xlim[0] + bin_width*0.5, xlim[1] - bin_width*0.5),
            bins=bin_edges, histtype="step", label=label_r_pred, color=color_pred
        )
        ax.set_title("Linear Scale")
        ax.set_xlabel("R")
        ax.set_ylabel(label_yaxis)
        ax.legend(frameon=False)
        ax.set_xlim(xlim)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ### Plot: 1d histogram, filled=R-values, log-scale
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        xlim = [np.min(y), np.max(y)]
        _, bin_edges, _ = ax.hist(
            np.clip(y, xlim[0], xlim[1]),
            bins=n_bins, histtype="step", label=label_r_truth, color=color_truth
        )
        ax.hist(
            np.clip(prediction, xlim[0], xlim[1]),
            bins=bin_edges, histtype="step", label=label_r_pred, color=color_pred
        )
        ax.set_title("Log Scale")
        ax.set_xlabel("R")
        ax.set_ylabel(label_yaxis)
        ax.set_yscale('log')
        ax.legend(frameon=False)
        ax.set_xlim(xlim)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ##############################
        ### Energy / r-value plots ###

        # TODO: make this better!
        # hardcoding inverse preprocessing
        std_scale = 1.4257378451544638
        mean_scale = 1.6009432921797704
        energy_log = x_test[:, 0] * std_scale + mean_scale
        energy = np.exp(energy_log)

        ### Plot: 2d histogram, filled = R_predicted vs E_truth
        n = 10000
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        energy_predicted = energy * 1. / prediction
        energy_true = energy * 1. / y
        ax.hist2d(energy_true, prediction, bins=[np.linspace(0, 100, 50), np.linspace(0, 5, 50)], norm=mpl.colors.LogNorm())
        ax.set_xlabel(label_e_truth + " " + units_energy)
        ax.set_ylabel(label_r_pred)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ### Plot: 1d histograms, filled = R-values, but conditioned in True energy bins
        energy_bins = np.linspace(0, 100, 30)

        # loop over true energy bins
        for i in range(energy_bins.shape[0]-1):

            print(f"Making distribution plot {i+1} / {len(energy_bins)-1}")
            low = energy_bins[i]
            high = energy_bins[i+1]
            fig = plt.figure(figsize=[5.5, 5])
            ax = fig.add_subplot(1, 1, 1)
            bins = np.linspace(0, 3, n_bins)
            bin_width = bins[1] - bins[0]
            xlim = [bins[0], bins[-1]]
            mask = np.all([energy > low, energy <=high], axis=0)
            y_selected = y[mask]
            prediction_selected = prediction[mask]

            if log_prob is not None:

                # let's plot the full predicted distribution:
                # The BNN gives us a distribution over possible R-values for each input data point x:
                #   p^{BNN}(R | x)
                # What we want to plot here is:
                #   p(R | true_energy)
                #
                # In general if we want to plot p(R | x_specific_feature) we need to marginalize over all
                # other directions. 
                #   p^{BNN}(R | x_specific_feature) = int dx p^{BNN}(R | x) * p(x | x_specific_feature)
                #                                   ~ 1/N sum_i p^{BNN}(R | x_i) with x_i~p(x | x_specific_feature)
                #                                   ~ Mean_over_test_samples( Predicted_BNN_dist )
                # We can take for x_i the test dataset. We only need to average over all predicted distributions
                # in this case x_specific_feature=E_truth

                # log_probs.shape = (len(y_draw), n_monte, len(x_test_dataset))
                log_prob_selected = log_prob[:, :, mask]
                prob_reduced = np.mean(np.exp(log_prob_selected), axis=-1)
                prob_avg = np.mean(prob_reduced, axis=-1) # mean over Bayesian weight samples
                ax.plot(y_draw, prob_avg, label="BNN predicted\ndistribution")

            # plotting histogram of truth labels and predictions
            _, bin_edges, _ = ax.hist(
                np.clip(y_selected, xlim[0] + bin_width*0.5, xlim[1] - bin_width*0.5),
                bins=bins, histtype="step", label=label_r_truth, density=True, color=color_truth
            )
            _, bin_edges, _ = ax.hist(
                np.clip(prediction_selected, xlim[0] + bin_width*0.5, xlim[1] - bin_width*0.5),
                bins=bins, histtype="step", label=label_r_pred, density=True, color=color_pred
            )
            ax.set_xlabel("R")
            ax.set_title(label_e_truth + f" = [{np.round(low, 1)}, {np.round(high, 1)}]" + " " + units_energy)
            ax.set_ylabel(label_yaxis)
            ax.legend(frameon=False)
            ax.set_xlim(xlim)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        ### Plot: 2d histogram, filled = predicted energy vs true energy
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        ax.hist2d(energy_predicted, energy_true, bins=[np.linspace(0, 100, 50), np.linspace(0, 100, 50)], norm=mpl.colors.LogNorm())
        ax.plot([y.min(), y.max()], [y.min(), y.max()], linestyle=":", color="black")
        ax.set_xlabel(label_e_truth + " " + units_energy)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.set_ylabel(label_e_pred + " " + units_energy)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ### Plot: 1d histogram, filled = energy, linear scale
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(energy_true, bins=np.linspace(-10, 100, 50), histtype="step", label=label_e_truth, color=color_truth)
        ax.hist(energy_predicted, bins=np.linspace(-10, 100, 50), histtype="step", label=label_e_pred, color=color_pred)
        ax.set_xlabel("Energy" + " " + units_energy)
        ax.set_yscale("log")
        ax.set_xlim([-30, 100])
        ax.set_ylabel("Frequency")
        ax.set_title("Linear Scale")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ### Plot: 1d histogram, filled = energy, log scale
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        bins = np.logspace(-2, 3, 100+1)
        ax.hist(energy_true,bins=bins, histtype="step", label=label_e_truth, color=color_truth)
        ax.hist(energy_predicted,bins=bins, histtype="step", label=label_e_pred, color=color_pred)
        ax.set_xlabel("Energy" + " " + units_energy)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("Log Scale")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        #################################
        ### Uncertainty plots ###

        ### Plot: 1d histogram, filled = sigma_pred
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        bins = np.linspace(0, 0.3, 50)
        xlim = [bins[0], bins[-1]]
        _, bin_edges, _ = ax.hist(
                np.clip(sigma_pred, xlim[0] + bin_width*0.5, xlim[1] - bin_width*0.5),
                bins=bins, histtype="step", color=color_pred
        )
        ax.set_xlabel("$\sigma_{\mathrm{pred}}$")
        ax.set_ylabel(label_yaxis)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ### Plot: 1d histogram, filled = sigma_stoch
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        bins = np.linspace(0, 10, 50)
        xlim = [bins[0], bins[-1]]
        _, bin_edges, _ = ax.hist(
                np.clip(sigma_stoch, xlim[0] + bin_width*0.5, xlim[1] - bin_width*0.5),
                bins=bins, histtype="step", color=color_pred
        )
        ax.set_xlabel("$\sigma_{\mathrm{model}}$")
        ax.set_ylabel(label_yaxis)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ### Plot: 1d histogram, filled = sigma_tot
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        bins = np.linspace(0, 10, 50)
        xlim = [bins[0], bins[-1]]
        _, bin_edges, _ = ax.hist(
                np.clip(sigma_tot, xlim[0] + bin_width*0.5, xlim[1] - bin_width*0.5),
                bins=bins, histtype="step", color=color_pred
        )
        ax.set_xlabel("$\sigma_{\mathrm{tot}}$")
        ax.set_ylabel(label_yaxis)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ###################################
        ### Pull plots ###

        ### Plot: 1d histogram, filled=pull_total
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)

        # gauss fit
        loc = np.mean(mse_norm_i_v1)
        scale = np.std(mse_norm_i_v1)
        x = np.linspace(loc - 3 * scale, loc + 3 * scale)
        y = norm.pdf(x, loc=loc, scale=scale)
        bins = np.linspace(loc - 3 * scale, loc + 3 * scale)
        bin_width = bins[1] - bins[0]
        _, bin_edges, _ = ax.hist(
            np.clip(mse_norm_i_v1, bins[0] + bin_width*0.5, bins[-1] - bin_width*0.5),
            bins=n_bins, histtype="step", label="BNN", density=True
        )
        ax.plot(x, y, label="Gauss ({:4.4f} {:4.4f})".format(loc, scale))
        ax.set_xlabel("$t_{\mathrm{tot}}$")
        ax.set_ylabel("Normalized")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ### Plot: 1d histogram, filled=pull_pred
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)

        # gauss fit
        loc = np.mean(mse_norm_i_v2)
        scale = np.std(mse_norm_i_v2)
        x = np.linspace(loc - 3 * scale, loc + 3 * scale)
        y = norm.pdf(x, loc=loc, scale=scale)
        ax.plot(x, y, label="Gauss ({:4.4f} {:4.4f})".format(loc, scale))
        bins = np.linspace(loc - 3 * scale, loc + 3 * scale)
        bin_width = bins[1] - bins[0]
        _, bin_edges, _ = ax.hist(
            np.clip(mse_norm_i_v2, bins[0] + bin_width*0.5, bins[-1] - bin_width*0.5),
            bins=n_bins, histtype="step", label="BNN", density=True
        )
        ax.set_xlabel("$t_{\mathrm{pred}}$")
        ax.set_ylabel("Normalized")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        ### Plot: 1d histogram, filled=pull_model
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)

        # gauss fit
        loc = np.mean(mse_norm_i_v3)
        scale = np.std(mse_norm_i_v3)
        x = np.linspace(loc - 3 * scale, loc + 3 * scale)
        y = norm.pdf(x, loc=loc, scale=scale)
        ax.plot(x, y, label="Gauss ({:4.4f} {:4.4f})".format(loc, scale))
        bins = np.linspace(loc - 3 * scale, loc + 3 * scale)
        bin_width = bins[1] - bins[0]
        _, bin_edges, _ = ax.hist(
            np.clip(mse_norm_i_v3, bins[0] + bin_width*0.5, bins[-1] - bin_width*0.5),
            bins=n_bins, histtype="step", label="BNN", density=True
        )
        ax.set_xlabel("$t_{\mathrm{model}}$")
        ax.set_ylabel("Normalized")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)


    ################################################################

    print(f"Directory: {dir_path}")
    sys.stdout.close()

####################################

if __name__ == "__main__":
    main()
