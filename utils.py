import matplotlib.pyplot as plt
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns

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

def plotRealPredict(predictions, real):
    fig, ax = plt.subplots()
    bins = np.linspace(0, 2, 21, endpoint=True)
    ax.hist(predictions, bins=bins, color = 'r', label='r_e_pred', alpha=0.4)
    ax.hist(real, bins=bins, color = 'b', label='r_e_calc', alpha=0.4)
    ax.set_xlabel('Response')
    plt.legend()
    plt.savefig('histoResp.png')
	#plt.show()


def plot_loss(history):
	plt.plot(history.history['loss'], label='loss')
	plt.plot(history.history['val_loss'], label='val_loss')
	#plt.ylim([0, 2])
	plt.xlabel('Epoch')
	plt.legend()
	plt.grid(True)
	plt.savefig("Losses.png")

# def plotError(predictions, real):
# 	error = predictions - real
# 	fig, ax = plt.subplots()
# 	ax.hist(error, bins=100, range=(-2,2))
# 	_ = ax.set_ylabel('Count')
# 	#plt.show()
#     plt.savefig('plotError.png')
#
# def plotGraphs(data):
# 	fig, ax = plt.subplots(3,4)
# 	index = 0
# 	data_transpose = data.transpose()
#
# 	for i in range(len(column_names)-1):
#
# 		if i%4 == 0 and i != 0:
# 			index+=1
# 		ax[index, i%4].plot(data_transpose[i], data_transpose[-1], 'o')
# 		ax[index, i%4].set_xlabel('r_e_calculated')
# 		ax[index, i%4].set_ylabel(column_names[i])
#
# 	plt.show()
#
#
#
#
#
#
