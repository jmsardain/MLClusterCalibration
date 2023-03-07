""" Training

This code was created for making some test trainnigs.
It was never really meant for sharing, so it's a bit messy.

@author: Michel Luchmann
"""

####################################
### Imports ###

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

from utils import *
from models import *
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

########################################

def main():

    ##########################################################
    ### Training parameters and hyperparameters ###

    parser = argparse.ArgumentParser('Train a BNN!')

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
        '--mode',
        default='bayesian',
        type=str,
    )

    args = parser.parse_args()

    ##############################
    ### Additional parameters ###

    me_label_index_rev = -2 # 32: phase weight, 33: matrix weight

    debug_path = '/home/jmsardain/BNN/debug'

    ##########################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ##############################
    input_dim = 15 ## number of features
    # extract preprocessing scales ## check with Michel
    # scales_path = args.data_path_train.replace("_train", "")
    # scales_path = scales_path.replace(".npy", "")
    # scales_path = scales_path + "_scales.npy"
    # print("File for scales of preprocessing {}".format(scales_path))
    # scales = np.load(args.data_path_train)

    # input_dim = scales.shape[0] - 3
    # me_label_index = scales.shape[0] + me_label_index_rev
    me_label_index = 0 ## first element in array is response

    # not actually used here! TODO
    #mean_p, scale_p = scales[me_label_index]

    ########################################

    # safe function to create new directory and don't overwrite any old files
    dir_path = create_directory(args.output_path)

    sys.stdout = open(dir_path + "/std.out", 'w')
    sys.stderr = open(dir_path + "/err.out", 'w')

    ###############################

    if args.mode.lower() in ['bayesian', 'bayes', 'bnn']:
        print("Initializing BNN...")
        model = BNN(
            args.train_size,
            args.layer,
            input_dim,
            activation_inner=args.activation_inner,
            activation_last=args.activation_last
        ).to(device)
        print(model)
    else:
        raise NotImplementedError("Option mode='{}' is not implemented".format(args.mode))


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters of model: {}".format(n_params))

    ##############################
    # set up datasets and dataloaders

    # train dataset
    dataset_train = np.load(args.data_path_train)
    x_train = dataset_train[:args.train_size, :input_dim]
    if args.target.lower() == "resp":
        y_train = dataset_train[:args.train_size, me_label_index] # matrix weight
        print(y_train)
    else:
        raise NotImplementedError("Option target={} is not implemented!".format(args.target))
    data_train = np.concatenate([x_train, y_train[:, None]], axis=-1)
    data_train = torch.from_numpy(data_train).to(device)

    print(f"Training dataset size {y_train.shape[0]}")

    # val dataset
    dataset_val = np.load(args.data_path_val)
    x_val = dataset_val[:args.val_size, :input_dim]
    if args.target.lower() == "resp":
        y_val = dataset_val[:args.val_size, me_label_index] # matrix weight
        print(y_val)
    else:
        raise NotImplementedError("Option target={} is not implemented!".format(args.target))
    data_val = np.concatenate([x_val, y_val[:, None]], axis=-1)
    data_val = torch.from_numpy(data_val).to(device)

    print(f"Validation dataset size {y_val.shape[0]}")

    # test dataset
    dataset_test = np.load(args.data_path_test)
    x_test = dataset_test[:args.test_size, :input_dim]
    if args.target.lower() == "resp":
       y_test = dataset_test[:args.test_size, me_label_index] # matrix weight
    else:
        raise NotImplementedError("Option target={} is not implemented!".format(args.target))
    #data = np.stack([x_test, y_test], axis=-1)
    #data = torch.from_numpy(data).to(device)
    y_test = torch.from_numpy(y_test).to(device)
    x_test = torch.from_numpy(x_test).to(device)

    print(f"Test dataset size {y_test.shape[0]}")

    # data loaders
    train_dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(data_val, batch_size=args.batch_size, shuffle=True)

    ##################################
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # save arguments
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

        # TODO: this is ugly and not generalizable, the method loss_fn doesn't make sense
        if args.mode.lower() in ['bayesian', 'bayes', 'bnn']:
            # gradient updates
            loss_dict = train_loop(train_dataloader, model, neg_log_gauss, optimizer, loss_dict)

            # validaiton pass
            loss_val_dict = val_pass(val_dataloader, model, neg_log_gauss, loss_val_dict)
        else:
           raise NotImplementedError("Option mode='{}' is not implemented".format(args.mode))

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
    # Create plots for trainings loss

    print(loss_dict)
    print(loss_val_dict)

    with PdfPages(dir_path + "/losses.pdf") as pdf:
        for key in loss_dict.keys():

            loss_list = np.array(loss_dict[key])
            loss_val_list = np.array(loss_val_dict[key])

            fig = plt.figure(figsize=[5.5, 5])
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(1, len(loss_list)+1, 1), loss_list, label=key)
            ax.plot(np.arange(1, len(loss_val_list)+1, 1), loss_val_list, label=key + " (validation)")

            ax.legend(frameon=False)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss: " + key)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=[5.5, 5])
            ax = fig.add_subplot(1, 1, 1)

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

    ###########################################
    # Evaluate predictions

    x_test = x_test.float()

    if args.mode.lower() in ['bnn', "bayesian", "bayes"]:

        outputs, sigmas2  = [], []
        for i in range(args.n_monte):
            print(f"Evaluating {i+1} of {args.n_monte} predictions")
            model.reset_random()
            y_eval = model(x_test)
            y_eval = y_eval.cpu().detach().numpy()
            output = y_eval[:, 0]
            sigma_stoch = y_eval[:, 1]
            outputs.append(output)
            sigmas2.append(np.exp(sigma_stoch))

        outputs = np.stack(outputs, axis=0)
        sigmas2 = np.stack(sigmas2, axis=0)
        mean = np.mean(outputs, axis=0)
        sigma_pred = np.std(outputs, axis=0)
        sigma_stoch = np.sqrt(np.mean(sigmas2, axis=0))
        sigma_tot = np.sqrt(sigma_pred**2 + sigma_stoch**2)

    print(f"Minimum prediction {np.min(mean)}")
    print(f"Mean sigma_stoch {np.mean(sigma_stoch)}")
    print(f"Mean sigma_pred {np.mean(sigma_pred)}")
    print(f"Mean sigma_tot {np.mean(sigma_tot)}")

    # truth label
    y = y_test.cpu().detach().numpy()

    ###########################################
    # create plots
    plot_path = dir_path + "/" + "eval_without_removing_prep.pdf"
    with PdfPages(plot_path) as pdf:

        # compute pulls and mse
        mse_i = np.where(y!=0, (mean - y) / y, np.zeros_like(y))
        mse_norm_i_v1 = (mean - y) / sigma_tot
        mse_norm_i_v2 = (mean - y) / sigma_pred
        mse_norm_i_v3 = (mean - y) / sigma_stoch
        mse_norm_i_v4 = np.mean((outputs - y) / np.sqrt(sigmas2), axis=0)

        n_bins = 50

        print("MSE...")
        mse_i = remove_nans_and_inf(mse_i, replace_value=None)
        print("MSE norm v1...")
        mse_norm_i_v1 = remove_nans_and_inf(mse_norm_i_v1, replace_value=None)
        print("MSE norm v2 ...")
        mse_norm_i_v2 = remove_nans_and_inf(mse_norm_i_v2, replace_value=None)
        print("MSE norm v3...")
        mse_norm_i_v3 = remove_nans_and_inf(mse_norm_i_v3, replace_value=None)
        print("MSE norm v4...")
        mse_norm_i_v4 = remove_nans_and_inf(mse_norm_i_v4, replace_value=None)

        #######################
        # amplitudes, zoomed in
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        #xlim = [-0.004, 0.01]
        xlim = [np.min(y), np.max(y)]
        bins = np.linspace(xlim[0], xlim[1], n_bins) # increase range a bit
        bin_width = bins[1] - bins[0]
        _, bin_edges, _ = ax.hist(
            np.clip(y, xlim[0] + bin_width*0.5, xlim[1] - bin_width*0.5),
            bins=bins, histtype="step", label="truth"
        )
        ax.hist(
            np.clip(mean, xlim[0] + bin_width*0.5, xlim[1] - bin_width*0.5),
            bins=bin_edges, histtype="step", label='prediction'
        )
        ax.set_xlabel("Ampltitudes")
        ax.set_ylabel("Events")
        ax.legend(frameon=False)
        ax.set_xlim(xlim)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # amplitudes log
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        xlim = [-0.01, 0.5]
        xlim = [np.min(y), np.max(y)]
        _, bin_edges, _ = ax.hist(
            np.clip(y, xlim[0], xlim[1]),
            bins=n_bins, histtype="step", label="truth"
        )
        ax.hist(
            np.clip(mean, xlim[0], xlim[1]),
            bins=bin_edges, histtype="step", label='prediction'
        )
        ax.set_xlabel("Ampltitudes")
        ax.set_ylabel("Events")
        ax.set_yscale('log')
        ax.legend(frameon=False)
        ax.set_xlim(xlim)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # amplitudes scatter, log plot
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        n = 10000
        mask = mean[:n] > 0
        print(mask.shape, n, mean.shape)
        ax.scatter(mean[:n][mask], y[:n][mask], s=0.2)
        ylim = [y[:n][mask].min(), mean[:n][mask].max()]
        xlim = ylim
        ax.plot(xlim, xlim, linestyle=":", color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Truth")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ampltiudes scatter
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        n = 10000
        mask = np.ones_like(mean[:n]).astype('bool')
        ax.scatter(mean[:n][mask], y[:n][mask], s=0.2)
        ylim = [-0.02, 0.5]
        xlim = ylim
        ax.plot(xlim, xlim, linestyle=":", color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Truth")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # amplitudes errorbarplot
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        n = 100
        mask = np.ones_like(mean[:n]).astype('bool')
        ax.errorbar(y[:n][mask], mean[:n][mask], yerr=sigma_tot[:n][mask], linestyle="", capsize=2.)
        ylim = [0, 0.2]
        xlim = ylim
        ax.plot(xlim, xlim, linestyle=":", color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel("Prediction")
        ax.set_xlabel("Truth")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # amplitudes 2d plot
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        mask = np.ones_like(mean[:n]).astype('bool')
        ylim = [0, 0.01]
        xlim = ylim
        ax.hist2d(y, mean, bins=n_bins, range=[xlim, ylim])
        ax.plot(xlim, xlim, linestyle=":", color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.set_ylabel("Prediction")
        ax.set_xlabel("Truth")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # uncerainties
        print("Plot uncertainties...")
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        _, bin_edges, _ = ax.hist(sigma_pred, bins=n_bins, histtype="step")
        ax.set_xlabel("$\sigma_{\mathrm{pred}}$")
        ax.set_ylabel("Events")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        _, bin_edges, _ = ax.hist(sigma_stoch, bins=n_bins, histtype="step")
        ax.set_xlabel("$\sigma_{\mathrm{model}}$")
        ax.set_ylabel("Events")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)
        _, bin_edges, _ = ax.hist(sigma_tot, bins=n_bins, histtype="step")
        ax.set_xlabel("$\sigma_{\mathrm{tot}}$")
        ax.set_ylabel("Events")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # performance
        print("Plot performance...")
        fig = plt.figure(figsize=[5.5, 5])
        ax = fig.add_subplot(1, 1, 1)

        # gauss fit
        loc = np.mean(mse_i)
        scale = np.std(mse_i)
        x = np.linspace(loc - 3 * scale, loc + 3 * scale)
        y = norm.pdf(x, loc=loc, scale=scale)
        bins = np.linspace(loc - 3 * scale, loc + 3 * scale)
        bin_width = bins[1] - bins[0]
        ax.plot(x, y, label="Gauss ({:4.4f} {:4.4f})".format(loc, scale))
        _, bin_edges, _ = ax.hist(
            np.clip(mse_i, bins[0] + bin_width*0.5, bins[-1] - bin_width*0.5),
            bins=bins, histtype="step", label="BNN", density=True
        )

        # second gauss fit
        loc = np.mean(mse_i)
        scale = np.std(mse_i[np.abs(mse_i) < 1])
        x = np.linspace(loc - 3 * scale, loc + 3 * scale)
        y = norm.pdf(x, loc=loc, scale=scale)
        ax.plot(x, y, label="Gauss central ({:4.4f} {:4.4f})".format(loc, scale))
        ax.set_xlabel("$\Delta_{\mathrm{test}}$")
        ax.set_ylabel("Events")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # pulls
        print("Plot pulls...")
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

        # gauss fit
        loc = np.mean(mse_norm_i_v4)
        scale = np.std(mse_norm_i_v4)
        x = np.linspace(loc - 3 * scale, loc + 3 * scale)
        y = norm.pdf(x, loc=loc, scale=scale)
        ax.plot(x, y, label="Gauss ({:4.4f} {:4.4f})".format(loc, scale))
        bins = np.linspace(loc - 3 * scale, loc + 3 * scale)
        bin_width = bins[1] - bins[0]
        _, bin_edges, _ = ax.hist(
            np.clip(mse_norm_i_v4, bins[0] + bin_width*0.5, bins[-1] - bin_width*0.5),
            bins=n_bins, histtype="step", label="BNN", density=True
        )
        ax.set_xlabel("$t_{\mathrm{model}}(\omega)$")
        ax.set_ylabel("Normalized")
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    ################################################################

    print(f"Directory: {dir_path}")
    sys.stdout.close()

####################################

if __name__ == "__main__":
    main()
