'''
Author: Olivia Pitcl
Collaborators: Jad Sardain

Description:
From a root file, processes individual cluster information and trains
a DNN to modify the values of the energy response to match a desired
ratio encoded in the cluster information. Assume the data is run on
all true energies above 0 GeV.

During training, the following parameters can be tailored to change the
design of the neural net:
activation function
loss function
data transformation function
input features

This file contains tab-wise indentation

To run:
source setup.sh
python training.py --nentries 1000000
Should do everything up until plotting!
'''


import uproot as ur
import pandas as pd
import numpy  as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

from functions_new import *
from sklearn.preprocessing import  QuantileTransformer
from sklearn.model_selection import train_test_split
from utils import *



def get_dataframe(nentries, cut):
	print("Retrieving the data...\n\n")
	filename = "/data1/atlng02/loch/Summer2022/MLTopoCluster/data/Akt4EMTopo.topo_cluster.root"
	# filename = "/home/loch/Summer2022/MLTopoCluster/data/Akt4LCTopo.inclusive_topo_cluster.root"
	file = ur.open(filename)
	tree = file["ClusterTree"]
	
	df = tree.arrays(library="pd")
	if nentries > 0:
	    df = df.sample(n = nentries)

	# -- Add response
	resp = np.array( df.clusterE.values ) /  np.array( df.cluster_ENG_CALIB_TOT.values )
	df["r_e_calculated"] = resp

	# -- change clusterECalib to clusterECalib_old, but don't put old back in
	vals = df.pop('clusterECalib')
	# df["clusterECalib_old"] = vals

	# -- add in recalculated clusterECalib (clusterECalib new)
	vals = np.array( df.cluster_HAD_WEIGHT.values ) * np.array( df.clusterE.values )
	df["clusterECalib"] = vals

	column_names = ['r_e_calculated', 'clusterE', 'clusterEtaCalib', 'cluster_CENTER_MAG',
			'cluster_CENTER_LAMBDA', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
			'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_time',
			'cluster_ISOLATION', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
			'nPrimVtx', 'avgMu', 'cluster_ENG_CALIB_TOT', 'clusterECalib']
	df  = df[column_names]

	# -- Sanity cuts (if not done, this variable gives inf when logged)
	df = df[df["cluster_ENG_CALIB_TOT"] >= cut]
	df = df[(df["clusterE"] > 0) & (df["cluster_FIRST_ENG_DENS"] > 0) & (df["cluster_CENTER_LAMBDA"] > 0)]

	# -- Split dataframe where train is 80% and test is 20%
	labels = df['r_e_calculated']
	df_train, df_test, train_target, test_target = train_test_split(df, labels, test_size=0.2, random_state=2)


	print("Saving dataframes...")
	df_train.to_csv("train_all.csv")
	df_test.to_csv("test_all.csv")
	
	df_train = cleanData(df_train)
	df_test = cleanData(df_test)
	return df_train, df_test

def main():
	parser = argparse.ArgumentParser(description='Prepare CSV files for MLClusterCalibration')
	parser.add_argument('--nentries', dest='nentries', type=int, default=0, help='random selection of events from df')
	args = parser.parse_args()

	# -- Process dataframe
	cut = input("Truth energy cut (0 or 0.3): ")
	while cut != "0" and cut != "0.3":
		cut = input("Truth energy cut (0 or 0.3): ")
	cut = float(cut)
	df_train, df_test = get_dataframe(args.nentries, cut)
	
	path = "/home/opitcl/calo-jad/MLClusterCalibration/"
	

	# -- Training
	print("Training...")
	history, model = train(df_train)
	utils.plot_loss(history)
	
	# -- Testing
	print("Testing...")
	r_e_calc, pred = test(df_test, model)
	utils.plotRealVsPredict(r_e_calc, pred)
	utils.plotRealPredict(r_e_calc, pred)
	utils.plotRealPredictFit(r_e_calc, pred)

	# -- Plotting
	print("Saving plotting csv...")
	test_file = path + "test_all.csv"
	result_file = path + "results_all.csv"
	df_plot = utils.finalplot(test_file, result_file)
	df_plot.to_csv("/home/opitcl/calo-jad/MLClusterCalibration/FinalPlots/plot_all.csv", sep=' ', index=False)


if __name__ == '__main__':
	main()
			
