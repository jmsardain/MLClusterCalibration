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

def getData(filename):
	res = pd.read_csv(filename)
	df = res.drop(columns=['cluster_ENG_CALIB_TOT', 'clusterECalib'])
	for icol in df.columns:
		if icol== "r_e_calculated":
			continue
		arr = np.array(df[icol].values)
		if icol == "clusterE" or icol == "cluster_FIRST_ENG_DENS":
			arr = np.log(arr)
		brr = [[i] for i in arr]
		#quantile = QuantileTransformer(random_state=0, output_distribution='normal')
		quantile = StandardScaler()
		data_trans = quantile.fit_transform(brr)
		newcol = data_trans.flatten()
		df[icol] = newcol

	return df


def train(filename):
	dataset = getData(filename)
	train_dataset = dataset.sample(frac=0.8, random_state=0)
	test_dataset = dataset.drop(train_dataset.index)


	train_features = train_dataset.copy()
	test_features = test_dataset.copy()


	train_labels = train_features.pop('r_e_calculated')
	test_labels = test_features.pop('r_e_calculated')


	dnn_model = build_and_compile_model(train_features)
	dnn_model.summary()

	history = dnn_model.fit( train_features, train_labels, validation_split=0.2, epochs=10, batch_size=2048) # batch_size=1024


	test_results = dict()
	test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
	test_predictions = dnn_model.predict(test_features).flatten()
	saveModel(dnn_model)


	perm = PermutationImportance(dnn_model, random_state=1, scoring='neg_mean_absolute_percentage_error').fit(train_features,train_labels)
	w = eli5.show_weights(perm, feature_names = list(train_features.columns.values))
	resultImportance = pd.read_html(w.data)[0]
	print(list(zip(list(train_features.columns.values), perm.feature_importances_)))
	resultImportance.to_csv("featureImportance.csv", index=False)

	return history



def custom_loss_function(y_true, y_pred):
	medianloss = tfp.stats.percentile(tf.math.abs(y_true - y_pred), q=50.)
	#return tf.reduce_mean(medianloss)
	return medianloss

#def build_and_compile_model(norm, X_train):
def build_and_compile_model(X_train):
	#model = keras.Sequential([norm, layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(1)])
	model = keras.Sequential([layers.Flatten(input_shape=(X_train.shape[1],)),
									layers.Dense(64, activation='tanh'),
									layers.Dense(64, activation='tanh'),
									layers.Dense(128, activation='tanh'),
									layers.Dense(256, activation='tanh'),
									layers.Dense(1, activation='linear')])
	#model.compile(loss=custom_loss_function, optimizer=tf.keras.optimizers.Adam(0.001))
	model.compile(loss='mean_absolute_percentage_error', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mae'])
	#model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
	#model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
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

	print(dataset.iloc[0:50,:])
	col_names = dataset.columns
	print(len(col_names))

	dnn_model.summary()

	test_predictions = dnn_model.predict(dataset).flatten()
	dataset['r_e_calculated'] = r_e_calc
	dataset['r_e_predec'] = test_predictions

	print(dataset.iloc[0:50,:])
	dataset.to_csv('results_{}.csv'.format(rangeE), index=False)
	print(test_predictions)

	return r_e_calc, test_predictions
