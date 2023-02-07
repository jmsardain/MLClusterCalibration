'''
New set of functions designed to allow us to 
toggle hyperparameters, transformations, activation
functions, and loss functions from the command line.

The dataframe is also given directly to the test and train functions.


To do next:
debug
see if runs smoothly

plot Jad's transformation metrics
plot loss
in the future, throw error if file not formatted correctly
'''

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
from keras.utils import get_custom_objects
from keras.layers.core import Activation

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


def getCustom(df):
	'''
	Grooms data for custom transformation from Peter and Felix
	'''
	dict_min_max = {}
	print("{} {}".format(np.percentile(df["clusterE"], 0) , np.percentile(df["clusterE"], 100)))
	print("{} {}".format(np.percentile(df["r_e_calculated"], 0) , np.percentile(df["r_e_calculated"], 100)))

	for i in df.columns:
		if i=="clusterE" or i=="cluster_FIRST_ENG_DENS" or i=="cluster_CENTER_LAMBDA":
			dict_min_max[i] = [np.percentile(np.log(df[i]), 0) , np.percentile(np.log(df[i]), 100)]
		else:
			dict_min_max[i] = [np.percentile(df[i], 0) , np.percentile(df[i], 100)]
	
	df1 = transformData(df, dict_min_max)
	return df1




def cleanData(df):
	'''
	Prepares data for training, removing trueE and LCW calibrations, then transforming the
	data
	'''
	# -- Do not train or test with the following data columns
	df = df.drop(columns=['cluster_ENG_CALIB_TOT', 'clusterECalib'])
	# df = df.drop(columns=['cluster_ENG_CALIB_TOT', 'clusterECalib', 'clusterECalib_old'])


	# Now choose transformation
	transf = input("Enter transformation (quantile, custom, standard) ")
	while transf != "quantile" and transf != "custom" and transf != "standard":
		 transf = input("Enter transformation (quantile, custom, standard) ")
	if transf == "custom":
		return getCustom(df)

	for icol in df.columns:
		# -- Do not clean r_e_calc
		if icol== "r_e_calculated":
			continue
		arr = np.array(df[icol].values)

		# -- This is why these inputs were cleaned in getDataframe
		if icol == "clusterE" or icol == "cluster_FIRST_ENG_DENS":
			arr = np.log(arr)
		brr = [[i] for i in arr]

		quantile = []
		if transf == "quantile":
			quantile = QuantileTransformer(random_state=0, output_distribution='normal')
		elif transf == "standard":
			quantile = StandardScaler()
		data_trans = quantile.fit_transform(brr)
		newcol = data_trans.flatten()
		df[icol] = newcol

	return df

def train(df):
	'''
	Taking in 0.8 of the data, already cleaned
	'''
	# dataset = cleanData(df)
	dataset = df.copy()
	dataset.describe()

	# Commented out because it is already split
	# train_dataset = dataset.sample(frac=0.8, random_state=0)
	# test_dataset = dataset.drop(train_dataset.index)

	# train_features = train_dataset.copy()
	# test_features = test_dataset.copy()

	# train_labels = train_features.pop('r_e_calculated')
	# test_labels = test_features.pop('r_e_calculated')

	train_labels = dataset.pop('r_e_calculated')

	# print("input features: {}".format(train_features.columns))
	# print("target features: {}".format(train_labels.name))

	# -- Build the model
	dnn_model = build_and_compile_model(dataset)
	dnn_model.summary()
	
	# -- Training
	history = dnn_model.fit(dataset, train_labels, validation_split=0.20, epochs=100, batch_size=1024)


	# -- TODO: PLOT HISTORY

	# -- Returns loss and metrics
	# test_results = dnn_model.evaluate(test_features, test_labels, verbose=0)
	# test_names = dnn_model.metrics_names
	
	# -- Do we test the model now? -> I will say no for now
	# test_predictions = dnn_model.predict(test_features).flatten()
	saveModel(dnn_model)

	return history, dnn_model


def tanhPlus(x):
	return tf.tanh(x) + 1

def swish(x, beta=0.3):
	return 1/(1+tf.exp(-x))

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
	activation = input("Activation function: (tanh, swish, tanh+1) ")
	while activation != "tanh" and activation != "swish" and activation != "tanh+1":
		activation = input ("tanh or swish or tanh+1? ")


	# dnn_model.add(Activation(act_function))
	get_custom_objects().update({'tanh+1': Activation(tanhPlus), 'swish': Activation(swish)})
	# Original: four layers, tanh activation
	model = keras.Sequential([layers.Flatten(input_shape=(X_train.shape[1],)),
									layers.Dense(64, activation=activation),
									layers.Dense(64, activation=activation),
									layers.Dense(128, activation=activation),
									layers.Dense(256, activation=activation),
									layers.Dense(1, activation='linear')])
	loss_function = input("Loss function (mae, lgk) ")
	while loss_function != "mae" and loss_function != "lgk":
		loss_function = input("mae or lgk? ")
	if loss_function == "lgk":
		model.compile(loss=lgk_loss_function, optimizer=tf.keras.optimizers.Adam(0.001))
	else:
		model.compile(loss='mean_absolute_percentage_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), metrics=['mae'])
	return model


def saveModel(model):

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

	except:
		print("File doesn't exist")
		return None

def test(dataset, dnn_model):
	'''
	dnn_model = loadModel(path)
	if dnn_model == None:
		return
	'''

	r_e_calc = dataset.pop('r_e_calculated')
	
	# print(dataset.iloc[0:50,:])
	col_names = dataset.columns
	# print(len(col_names))

	dnn_model.summary()

	test_predictions = dnn_model.predict(dataset).flatten()
	# test_predictions = dnn_model.predict(dataset)
	dataset['r_e_calculated'] = r_e_calc
	dataset['r_e_predec'] = test_predictions

	print(dataset.iloc[0:50,:])
	dataset.to_csv('results_all.csv', index=False)
	print(test_predictions)
	
	return r_e_calc, test_predictions
