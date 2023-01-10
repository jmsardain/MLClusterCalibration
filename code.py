import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
import uproot as ur
import datetime, os
import tensorflow as tf
from sklearn.preprocessing import  QuantileTransformer, StandardScaler

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn import ensemble
parser = argparse.ArgumentParser(description='Perform signal injection test.')
parser.add_argument('--train',  dest='train',  action='store_const', const=True, default=False, help='Train NN  (default: False)')
parser.add_argument('--test',   dest='test',   action='store_const', const=True, default=False, help='Test NN   (default: False)')
parser.add_argument('--plot',   dest='plot',   action='store_const', const=True, default=False, help='Save plot (default: False)')
parser.add_argument('--path',   dest='path',   type=str, default='', help='Path to model')
parser.add_argument('--rangeE', dest='rangeE', type=str, default='', help='range in energy')
parser.add_argument('--rank',   dest='rank',   action='store_const', const=True, default=False, help='Do feature ranking, outputs are print on terminal and plot')
args = parser.parse_args()


def transformData(data, dict):
	a = -1
	b = 1
	data["clusterE"] = np.log(data["clusterE"].values)
	data["cluster_FIRST_ENG_DENS"] = np.log(data["cluster_FIRST_ENG_DENS"].values)
	data["cluster_CENTER_LAMBDA"] = np.log(data["cluster_CENTER_LAMBDA"].values)
	for icol in data.columns:
		arr = np.array(data[icol].values)
		# minValue = np.quantile(arr, 0.01)
		# maxValue = np.quantile(arr, 0.99)
		minValue = dict[icol][0]
		maxValue = dict[icol][1]

		s = (b-a) / (maxValue - minValue)
		newcol = a + s * (arr - minValue)
		data[icol] = newcol

	return data

def transformInvData(df, dict):
	a = -1
	b = 1
	for icol in df.columns:
		arr = np.array(df[icol].values)
		minValue = dict[icol][0]
		maxValue = dict[icol][1]
		s = (b-a) / (maxValue - minValue)
		newcol = minValue + (arr - a) / s
		if icol=="clusterE" or icol=="cluster_FIRST_ENG_DENS" or icol=="cluster_CENTER_LAMBDA":
			df[icol] = np.exp(newcol)
		else:
			df[icol] = newcol

	return df



def lgk_loss_function(y_true, y_pred): ## https://arxiv.org/pdf/1910.03773.pdf
	alpha = tf.constant(0.05)
	bandwith = tf.constant(0.5)
	pi = tf.constant(math.pi)
	## LGK (h and alpha are hyperparameters)
	norm = -1/(bandwith*tf.math.sqrt(2*pi))
	gaussian_kernel  = norm * tf.math.exp( -(y_true - y_pred)**2 / (2*(bandwith**2)))
	leakiness = alpha*tf.math.abs(y_true - y_pred)
	lgk_loss = gaussian_kernel + leakiness
	return lgk_loss

def build_and_compile_model(X_train, lr):
	model = keras.Sequential([layers.Flatten(input_shape=(X_train.shape[1],)),
									layers.Dropout(.2),
									layers.Dense(64,  activation='tanh'),
									layers.Dense(64,  activation='tanh'),
									layers.Dense(128, activation='tanh'),
									layers.Dense(256, activation='tanh'),
									layers.Dense(1,   activation='linear')])
	model.compile(loss=lgk_loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
	return model


# Main function.
def main():
	filename="/data1/atlng02/loch/Summer2022/MLTopoCluster/data/Akt4EMTopo.topo_cluster.root"

	file = ur.open(filename)
	tree = file["ClusterTree"]
	df = tree.arrays(library="pd", entry_stop=1000000)
	# df = tree.arrays(library="pd")
	print("Read tree")

	df = df[df["cluster_ENG_CALIB_TOT"] >= 0.3]
	df = df[(df["clusterE"] > 0) & (df["cluster_FIRST_ENG_DENS"] > 0) & (df["cluster_CENTER_LAMBDA"] > 0)]

	# -- Add response
	resp = np.array( df.clusterE.values ) /  np.array( df.cluster_ENG_CALIB_TOT.values )
	df["r_e_calculated"] = resp
	clusterECal =  np.array( df.cluster_HAD_WEIGHT.values ) *  np.array( df.clusterE.values )
	df["clusterECalib"] = clusterECal

	column_names = [ 'r_e_calculated', 'clusterE', 'clusterEtaCalib',
                        'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
                        'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_time', 'cluster_ISOLATION',
                        'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE', 'nPrimVtx', 'avgMu',
                        'cluster_ENG_CALIB_TOT', 'clusterECalib'
                        ]
	#
	# column_names = ['r_e_calculated',
	# 				'nPrimVtx', 'avgMu','clusterEtaCalib',
	# 				'clusterE', 'clusterPt', 'clusterPhi', 'cluster_MASS', 'cluster_sumCellE',
	# 				'cluster_time', 'cluster_fracE', 'cluster_PTD', 'cluster_ISOLATION',
	# 				'cluster_FIRST_ETA', 'cluster_FIRST_PHI', 'cluster_FIRST_ENG_DENS',
	# 				'cluster_SECOND_TIME', 'cluster_SECOND_R', 'cluster_SECOND_LAMBDA', 'cluster_SECOND_ENG_DENS',
	# 				'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG', 'cluster_CENTER_X', 'cluster_CENTER_Y', 'cluster_CENTER_Z',
	# 				'cluster_ENG_BAD_CELLS', 'cluster_ENG_BAD_HV_CELLS', 'cluster_ENG_FRAC_EM', 'cluster_ENG_FRAC_MAX', 'cluster_ENG_FRAC_CORE', 'cluster_ENG_POS',
	# 				'cluster_DELTA_THETA', 'cluster_DELTA_PHI',
	# 				'cluster_CELL_SIGNIFICANCE', 'cluster_CELL_SIG_SAMPLING',
	# 				'cluster_N_BAD_CELLS', 'cluster_BAD_CELLS_CORR_E',
	# 				'cluster_LONGITUDINAL', 'cluster_LATERAL', 'cluster_SIGNIFICANCE',
	# 				'nCluster', 'cluster_N_BAD_HV_CELLS', 'cluster_nCells', 'cluster_nCells_tot',
	# 				'cluster_ENG_CALIB_TOT', 'clusterECalib',
	# 				]

	dict_min_max = {}
	print("{} {}".format(np.percentile(df["clusterE"], 0) , np.percentile(df["clusterE"], 100)))
	print("{} {}".format(np.percentile(df["r_e_calculated"], 0) , np.percentile(df["r_e_calculated"], 100)))
	for i in column_names:
		if i=="clusterE" or i=="cluster_FIRST_ENG_DENS" or i=="cluster_CENTER_LAMBDA":
			dict_min_max[i] = [np.percentile(np.log(df[i]), 0) , np.percentile(np.log(df[i]), 100)]
		else:
			dict_min_max[i] = [np.percentile(df[i], 0) , np.percentile(df[i], 100)]

	var_to_check = 'r_e_calculated'
	print(dict_min_max[var_to_check])

	## original dataframe
	df = df[column_names]

	## make a copy of the dataframe and transform the input features
	df1 = df.copy()
	df1 = df1.drop(['cluster_ENG_CALIB_TOT', 'clusterECalib', 'r_e_calculated'],axis=1)
	df1 = transformData(df1, dict_min_max)
	df2 = df1.copy()
	df2 = transformInvData(df2, dict_min_max)


	labels = df['r_e_calculated']
	# train1 = df.drop(['cluster_ENG_CALIB_TOT', 'clusterECalib', 'r_e_calculated'],axis=1)
	train1 = df1.copy()
	print(train1.columns)

	x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.20,random_state =2)
	nepochs = 50
	dnn_model = build_and_compile_model(x_train, lr=1e-6)
	history = dnn_model.fit( x_train, y_train, validation_split=0.3, epochs=nepochs, batch_size=1024) # batch_size=1024

	fig, ax = plt.subplots()
	ax.plot(history.history['loss'], label='loss')
	ax.plot(history.history['val_loss'], label='val_loss')
	ax.set_xlabel('Epoch')
	plt.legend()
	plt.grid(True)
	plt.savefig("Losses.png")
	plt.clf()

	savehistory = {'loss': history.history['loss'], 'valloss': history.history['val_loss']}
	df_savehistory = pd.DataFrame(data=savehistory)
	df_savehistory.to_csv("history.csv", sep=' ', index=False)

	## testing
	y_pred = dnn_model.predict(x_test).flatten()


	## y_test
	# print(y_test.values)
	## y_pred
	# print(y_pred)

	# print(x_test.head())
	plot = x_test.copy()
	plot = transformInvData(plot, dict_min_max)
	plot['cluster_ENG_CALIB_TOT']    = df['cluster_ENG_CALIB_TOT']
	plot['clusterECalib']            = df['clusterECalib']
	plot['cluster_ENG_TOT_frompred'] = plot["clusterE"] / y_pred
	plot["r_e_calculated"]           = y_test
	plot["r_e_predec"]               = y_pred
	print(plot.head())
	print(plot.tail())
	plot.to_csv("plot.csv", sep=' ', index=False)




	return
# Main function call.
if __name__ == '__main__':
    main()
    pass
