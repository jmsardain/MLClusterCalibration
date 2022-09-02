import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy.interpolate import make_interp_spline, BSpline
import datetime, os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def plotRealVsPredict(predictions, real):
    fig, ax = plt.subplots()
    bins = np.linspace(0, 2, 21, endpoint=True)
    a = ax.hist2d(real, predictions, bins=[bins,bins], cmap = 'jet')
    fig.colorbar(a[3], ax=ax)
    ax.set_xlabel('real')
    ax.set_ylabel('predicted')
    plt.savefig('plotRealVsPredict.png')
	#plt.show()

def plotRealPredict(real, predictions):
    fig, ax = plt.subplots()
    bins = np.linspace(0, 2, 21, endpoint=True)
    ax.hist(predictions, bins=bins, color = 'r', label='r_e_pred', alpha=0.4)
    ax.hist(real,        bins=bins, color = 'b', label='r_e_calc', alpha=0.4)
    ax.set_xlabel('Response')
    plt.legend()
    plt.savefig('histoResp.png')
	#plt.show()


def plotRealPredictFit(real, predictions):
    fig, ax = plt.subplots(2, 1)
    bins = np.linspace(0, 2, 21, endpoint=True)
    # -- Up: distribution before transformation
    mean_pred, std_pred = norm.fit(predictions)
    ax[0].hist(predictions, bins=bins, color = 'r', label='r_e_pred', alpha=0.4, density=True)
    xmin, xmax = ax[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean_pred, std_pred)
    ax[0].plot(x, y)
    #ax[0].set_yscale('log')

    # -- Down: distribution after transformation
    mean_cal, std_cal = norm.fit(real)
    ax[1].hist(real, bins=bins, color = 'b', label='r_e_cal', alpha=0.4, density=True)
    xmin, xmax = ax[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean_cal, std_cal)
    ax[1].plot(x, y)
    #ax[1].set_yscale('log')
    # Show the graph
    fig.tight_layout()
    plt.savefig('histoRespFit.png')
    plt.clf()

def plot_loss(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig("Losses.png")
    plt.clf()


def plot_metrics(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['mae'], label='mae')
    ax.plot(history.history['val_mae'], label='val_mae')
    #plt.ylim([0, 2])
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig("MAE.png")
    plt.clf()

def finalplot(testName, resultName):
    df_test  = pd.read_csv(testName)
    df_res   = pd.read_csv(resultName)

    # -- Get all common columns
    # col = ['clusterE', 'clusterEtaCalib', 'cluster_CENTER_LAMBDA', 'cluster_ENG_FRAC_EM',
    #     'cluster_FIRST_ENG_DENS', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
    #     'cluster_PTD', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
    #     'nPrimVtx', 'avgMu', 'r_e_calculated']
    # col = ['clusterE', 'r_e_calculated',
    #                 'cluster_nCells_tot', 'cluster_SIGNIFICANCE',
    #                 'cluster_time', 'cluster_sumCellE', 'cluster_nCells', 'cluster_ENG_POS',
    #                 'cluster_FIRST_ENG_DENS', 'clusterPt', 'cluster_CENTER_LAMBDA',
    #                 'cluster_PTD', 'cluster_CELL_SIG_SAMPLING',
    #                 'cluster_SECOND_ENG_DENS', 'cluster_CELL_SIGNIFICANCE', 'cluster_MASS',
    #                 #'cluster_ENG_FRAC_MAX', 'cluster_ENG_FRAC_EM'
    #                 ]
    col = ['r_e_calculated',
           'nPrimVtx', 'avgMu',
           'clusterE', 'clusterPt', 'clusterPhi', 'cluster_MASS', 'cluster_sumCellE',
           'cluster_time', 'cluster_fracE', 'cluster_PTD', 'cluster_ISOLATION',
           'cluster_FIRST_ETA', 'cluster_FIRST_PHI', 'cluster_FIRST_ENG_DENS',
           'cluster_SECOND_TIME', 'cluster_SECOND_R', 'cluster_SECOND_LAMBDA', 'cluster_SECOND_ENG_DENS',
           'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG', 'cluster_CENTER_X', 'cluster_CENTER_Y', 'cluster_CENTER_Z',
           'cluster_ENG_BAD_CELLS', 'cluster_ENG_BAD_HV_CELLS', 'cluster_ENG_FRAC_EM', 'cluster_ENG_FRAC_MAX', 'cluster_ENG_FRAC_CORE', 'cluster_ENG_POS',
           'cluster_DELTA_THETA', 'cluster_DELTA_PHI',
           'cluster_CELL_SIGNIFICANCE', 'cluster_CELL_SIG_SAMPLING',
           'cluster_N_BAD_CELLS', 'cluster_BAD_CELLS_CORR_E',
           'cluster_LONGITUDINAL', 'cluster_LATERAL', 'cluster_SIGNIFICANCE',
           'nCluster', 'cluster_N_BAD_HV_CELLS', 'cluster_nCells', 'cluster_nCells_tot',
           ]
    # -- Check if the values are the same (they should be, this is just a protection)
    # for icol in col:
    #     print(icol)
    #     arr_test = np.array(df_test[icol].values)
    #     arr_res  = np.array(df_res[icol].values)
    #     flag = np.array_equal(arr_test, arr_res)
    #     print(flag)
    #     if flag==False:
    #         print("{} is false. Return.")
    #         return

    # -- Create new dataframe that will be used for final plots
    columnsname = col + ['r_e_predec', 'cluster_ENG_CALIB_TOT', 'cluster_ENG_TOT_frompred', 'clusterECalib']
    df_plot = pd.DataFrame(columns=columnsname)

    for icolumn in columnsname:
        print(icolumn)
        if icolumn == "cluster_ENG_CALIB_TOT":
            df_plot[icolumn] = df_test[icolumn]
        elif icolumn == "cluster_ENG_TOT_frompred":
            df_plot[icolumn] = df_test["clusterE"] / df_res["r_e_predec"]
        elif icolumn =="r_e_predec":
            df_plot[icolumn] = df_res["r_e_predec"]
        else:
            df_plot[icolumn] = df_test[icolumn]


    # r_e_calc     = np.array(df_res["r_e_calculated"].values)
    # r_e_pred     = np.array(df_res["r_e_predec"].values)
    # clusE        = np.array(df_res["clusterE"].values)
    #
    #
    # clusTOT      = np.array(df_test["cluster_ENG_CALIB_TOT"].values)
    # clusTOT_pred = np.array(df_res["clusterE"].values) / np.array(df_res["r_e_predec"].values)
    #
    # df_plot["r_e_calculated"]           = r_e_calc
    # df_plot["r_e_predec"]               = r_e_pred
    # df_plot["clusterE"]                 = clusE
    # df_plot["cluster_ENG_CALIB_TOT"]    = clusTOT
    # df_plot["cluster_ENG_TOT_frompred"] = clusTOT_pred

    return df_plot

#['r_e_calculated', 'r_e_predec', 'TOT_frompred', 'clusterE',  'cluster_ENG_CALIB_TOT', 'clusterEtaCalib', 'cluster_CENTER_LAMBDA',
#'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
#'cluster_PTD', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE', 'nPrimVtx', 'avgMu']
