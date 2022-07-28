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


parser = argparse.ArgumentParser(description='Perform signal injection test.')

parser.add_argument('--train', dest='train', action='store_const', const=True, default=False, help='Train NN (default: False)')
parser.add_argument('--test', dest='test', action='store_const', const=True, default=False, help='Test NN (default: False)')
parser.add_argument('--path', dest='path', type=str, default='', help='Path to model')
parser.add_argument('--plot', dest='plot', action='store_const', const=True, default=False, help='Save plots (default: False)')

args = parser.parse_args()


## how to run
## python traintest.py --train
## python traintest.py --test --path PathToModelDirectory
## python traintest.py --test --path TrainedModels/07-27-2022/
## python traintest.py --plot

# Main function.
def main():

	## path to train: /home/jmsardain/JetCalib/train.csv
	## path to test: /home/jmsardain/JetCalib/test.csv

	pathToCSVFile = "/home/jmsardain/JetCalib/"
	# -- Train using train dataset
	if args.train:
		filename = pathToCSVFile+"train.csv"
		history = train(filename)
		utils.plot_loss(history)

	# -- Get prediction on test dataset
	if args.test:
		filename = pathToCSVFile+"test.csv"
		r_e_calc, test_predictions = test(filename, args.path)
		utils.plotRealVsPredict(r_e_calc, test_predictions)
		utils.plotRealPredict(r_e_calc, test_predictions)
		#utils.plotError(r_e_calc, test_predictions)
		#utils.plotError(r_e_calc, test_predictions)

	# -- Get 2D plot
	if args.plot:
		filename = pathToCSVFile+"results.csv"
		utils.finalplot(filename)
		print("Under construction")


	return

# Main function call.
if __name__ == '__main__':
    main()
    pass
