import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns

import datetime, os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import utils
from utils import *
from functions import *
from sklearn.preprocessing import  QuantileTransformer, StandardScaler
import tensorflow_probability as tfp
import eli5
from eli5.sklearn import PermutationImportance


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
'''
def getData(filename):
	res = pd.read_csv(filename)
	df = res.drop(columns=['cluster_ENG_CALIB_TOT', 'clusterECalib_new'])
	# df = res.drop(columns=['cluster_ENG_CALIB_TOT', 'clusterECalib_new', 'clusterECalib_old'])

	dict_min_max = {}
	print("{} {}".format(np.percentile(df["clusterE"], 0) , np.percentile(df["clusterE"], 100)))
	print("{} {}".format(np.percentile(df["r_e_calculated"], 0) , np.percentile(df["r_e_calculated"], 100)))
	for i in df.columns:
		if i=="clusterE" or i=="cluster_FIRST_ENG_DENS" or i=="cluster_CENTER_LAMBDA":
			dict_min_max[i] = [np.percentile(np.log(df[i]), 0) , np.percentile(np.log(df[i]), 100)]
		else:
			dict_min_max[i] = [np.percentile(df[i], 0) , np.percentile(df[i], 100)]

	# for icol in df.columns:
	#	if icol== "r_e_calculated":
	#		continue
	#	arr = np.array(df[icol].values)
	#	if icol == "clusterE" or icol == "cluster_FIRST_ENG_DENS":
	#	 	arr = np.log(arr)
	#	brr = [[i] for i in arr]
		# quantile = QuantileTransformer(random_state=0, output_distribution='normal')
		# quantile = StandardScaler()
		# data_trans = quantile.fit_transform(brr)
		# newcol = data_trans.flatten()
	#	df[icol] = newcol
	
	df1 = transformData(df, dict_min_max)
	return df1
'''

def getData(filename):
        res = pd.read_csv(filename)
        # df = res.drop(columns=['cluster_ENG_CALIB_TOT', 'clusterECalib_new'])
        df = res.drop(columns=['cluster_ENG_CALIB_TOT', 'clusterECalib_new', 'clusterECalib_old'])
        for icol in df.columns:
                if icol== "r_e_calculated":
                        continue
                arr = np.array(df[icol].values)
                if icol == "clusterE" or icol == "cluster_FIRST_ENG_DENS":
                        arr = np.log(arr)
                brr = [[i] for i in arr]
                quantile = QuantileTransformer(random_state=0, output_distribution='normal')
                # quantile = StandardScaler()
                data_trans = quantile.fit_transform(brr)
                newcol = data_trans.flatten()
                df[icol] = newcol

        return df

def train(filename, epochs, batch_size, i):
	dataset = getData(filename)
	train_dataset = dataset.sample(frac=0.8, random_state=0)
	test_dataset = dataset.drop(train_dataset.index)


	train_features = train_dataset.copy()
	test_features = test_dataset.copy()


	train_labels = train_features.pop('r_e_calculated')
	test_labels = test_features.pop('r_e_calculated')

	# Do not train on old, new clusterECalib

	print("input features: {}".format(train_features.columns))
	print("target features: {}".format(train_labels.name))
	dnn_model = build_and_compile_model(train_features)
	dnn_model.summary()
	

	history = dnn_model.fit( train_features, train_labels, validation_split=0.20, epochs=epochs, batch_size=batch_size) # old batch_size=1024, 100 epochs

	test_results = dict()
	test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
	test_predictions = dnn_model.predict(test_features).flatten()
	saveModel(dnn_model, i)

	doImportance = False
	if doImportance:
		perm = PermutationImportance(dnn_model, random_state=1, scoring='neg_mean_absolute_percentage_error').fit(train_features,train_labels)
		w = eli5.show_weights(perm, feature_names = list(train_features.columns.values))
		resultImportance = pd.read_html(w.data)[0]

		featureImportance = np.array(list(zip(list(train_features.columns.values), perm.feature_importances_)), dtype=[('featureName', 'S100'), ('tot', float)])
		featureImportance.sort(order='tot')

		for i in featureImportance[::-1]:
		    print("{}: {}".format(i[0], i[1]))

	return history



def lgk_loss_function(y_true, y_pred):
	alpha = tf.constant(0.05)
	bandwith = tf.constant(0.5)
	pi = tf.constant(math.pi)

	## LGK (h and alpha are hyperparameters)
	norm = -1/(bandwith*tf.math.sqrt(2*pi))
	gaussian_kernel  = norm * tf.math.exp( -(y_true - y_pred)**2 / (2*(bandwith**2)))
	leakiness = alpha*tf.math.abs(y_true - y_pred)
	lgk_loss = gaussian_kernel + leakiness
	return lgk_loss

def build_and_compile_model(X_train):
	activation = tf.nn.silu
	#model = keras.Sequential([norm, layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(1)])
	# Original: four layers, tanh activation
	model = keras.Sequential([layers.Flatten(input_shape=(X_train.shape[1],)),
									layers.Dense(64, activation=activation),
									layers.Dense(64, activation=activation),
									layers.Dense(128, activation=activation),
									layers.Dense(256, activation=activation),
									layers.Dense(1, activation='linear')])
	#model.compile(loss=custom_loss_function, optimizer=tf.keras.optimizers.Adam(0.001))
	model.compile(loss=lgk_loss_function, optimizer=tf.keras.optimizers.Adam(0.001))
	#model.compile(loss='mean_absolute_percentage_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), metrics=['mae'])
	#model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
	#model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
	return model


def saveModel(model, itera):

	Date = datetime.datetime.now().strftime('%m-%d-%Y')
	path = os.getcwd()+'/TrainedModels/'+Date+'/'
	try:
		os.mkdir(path)
	except:
		pass
	model.save(path)


def loadModel(path):
	try:
		return tf.keras.models.load_model(path, compile=False)
		#return tf.keras.models.load_model(path)


	except:
		print("File doesn't exist")
		return None

def test(filename, path, rangeE):
	dnn_model = loadModel(path)
	if dnn_model == None:
		return
	dataset = getData(filename)
	r_e_calc = dataset.pop('r_e_calculated')
	# clusNew = dataset.pop('clusterECalib_new')

	print(dataset.iloc[0:50,:])
	col_names = dataset.columns
	print(len(col_names))

	dnn_model.summary()

	test_predictions = dnn_model.predict(dataset).flatten()
	dataset['r_e_calculated'] = r_e_calc
	# dataset['clusterECalib_new'] = clusNew
	dataset['r_e_predec'] = test_predictions

	print(dataset.iloc[0:50,:])
	dataset.to_csv('results_{}.csv'.format(rangeE), index=False)
	print(test_predictions)
	
	return r_e_calc, test_predictions
