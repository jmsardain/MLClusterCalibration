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



def getData(filename):
	res = pd.read_csv(filename)
	df = res.drop(columns=['cluster_ENG_CALIB_TOT'])
	print(df.columns)
	print(len(df.index))
	return df


def train(filename):
    dataset = getData(filename)

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('r_e_calculated')
    test_labels = test_features.pop('r_e_calculated')

    normalizer = tf.keras.layers.Normalization(axis=-1)

    normalizer.adapt(np.array(train_features))

    dnn_model = build_and_compile_model(normalizer)
    dnn_model.summary()

    history = dnn_model.fit( train_features, train_labels, validation_split=0.2, epochs=20)
    test_results = dict()
    test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
    test_predictions = dnn_model.predict(test_features).flatten()
    saveModel(dnn_model)

    # print(pd.DataFrame(test_results, index=['Mean absolute error [r_e_calculated]']).T)
    # print(len(test_predictions))

    return history


def build_and_compile_model(norm):
	#model = keras.Sequential([norm, layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(1)])
	model = keras.Sequential([norm, layers.Dense(64, activation='tanh'), layers.Dense(64, activation='tanh'), layers.Dense(1)])
	model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
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
		return tf.keras.models.load_model(path)

	except:
		print("File doesn't exist")
		return None

def test(filename, path):
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
	dataset.to_csv('results.csv')
	print(test_predictions)

	return r_e_calc, test_predictions
