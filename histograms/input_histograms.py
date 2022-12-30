
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
'''
Receives range and two 2d lists: one which corresponds to a more
linearized data set and another which corresponds to the outlying
data set.

Creates histogram of input values
'''
def plot(rangeE, values_in, values_out, filenames, norm):
	# create 4, one for each file
	if values_out[1] == None:
		print("No outlying values to graph")
		values_out = [1] * len(values_in)
		values_out = np.array(values_out)
		
	fig, ax = plt.subplots(2,2)
	num = 4
	path = "/home/opitcl/calo-jad/MLClusterCalibration/histograms/results/all/"
	if rangeE:
		path = "/home/opitcl/calo-jad/MLClusterCalibration/histograms/results/" + rangeE + "/"
		fig, ax = plt.subplots()
		num = 1
	for i in range(num):
		maximum_in = np.max(values_in[i+1])
		minimum_in = np.min(values_in[i+1])
		maximum_out = np.max(values_out[i+1])
		minimum_out = np.min(values_out[i+1])
		
		# bins = np.linspace(minimum_in, maximum_in, 101, endpoint=True)
		# bins_out = np.linspace(minimum_out, maximum_out, 101, endpoint=True)
		bins = np.linspace(minimum_in, maximum_in, 101, endpoint=True)
		bins_out = np.linspace(minimum_out, maximum_out, 101, endpoint=True)
		
		row = 0
		col = 0
		if i == 1:
			col = 1
		elif i == 2:
			row = 1
		elif i == 3:
			row = 1
			col = 1
		# Normalize values in the array to 1
		# normalized_in = normalize(values_in[i+1], minimum_in, maximum_in)
		# normalized_out = normalize(values_out[i+1], minimum_out, maximum_out)
		# normalized_in = np.linalg.norm(values_in[i+1])
		# normalized_out = np.linalg.norm(values_out[i+1])
		normalized_in = values_in[i+1]
		normalized_out = values_out[i+1]
		if rangeE:
			ax.hist(normalized_in, bins=bins, histtype=u'step', color='b', density=norm)
			ax.hist(normalized_out, bins=bins_out, histtype=u'step', color='g', density=norm)
			ax.set_xlabel(filenames[i])
			ax.set_ylabel('Count')
		else:
			ax[row, col].hist(normalized_in, bins=bins, histtype=u'step', color='b', density=norm)
			ax[row, col].hist(normalized_out, bins=bins_out, histtype=u'step', color='g', density=norm)
			ax[row, col].set_xlabel(filenames[i])
			ax[row, col].set_ylabel('Count')

	fig.tight_layout()
	# plt.text(200, 200, r'-- linear', color='b')
	# plt.text(200, 210, r'-- ETrue < 0.1GeV', color='g')
	if rangeE:
		plt.savefig(path + rangeE + "_" + values_in[0] + ".png")
	else:
		plt.savefig(path + values_in[0] + ".png")
	plt.clf()
	plt.close()

def normalize(arr, minimum, maximum):
	new_arr = []
	diff = maximum - minimum
	for elem in arr:
		new_arr.append((elem - minimum) / diff)
	new_arr = np.array(new_arr)
	# print(new_arr)
	return new_arr

'''
Each element in the list will look like this by the end:
['cluster_ENG_CALIB_TOT', array([1.8745068e-01, 2.4229003e-02, 5.7890780e-03, ..., 5.4504395e-05,
       5.1734350e-04, 2.2101013e-02]), array([2.8097640e-01, 1.5818007e-01, 9.2299200e-03, ..., 7.2164537e-04,
       7.1507770e-02, 5.4504395e-05]), array([0.0329346...]) ]
Multiple arrays will correspond to each input, one for each range.
elem 0 - input name
elem 1 - input values
'''
def makeList(path, filenames):	
	# Load all info from CSV files
	# 2d list, first level for each input, second for each file with given input
	# first element in each subarray will have name of input
	# should end up with 4 subarrays, one for each file
	data_in = []
	data_out = []
	# data = []
	
	# This is in case each file has a different order of inputs
	old_input = []
	second = True

	for filename in filenames:
		print("Reading file", filename)
		file = open(path + filename)
			
		# One histogram per input per test
		inputs = file.readline().strip().split()

		values = []
		data = []
		if filename[0] == "o":
			data = data_out
		elif second:
			old_input = []
			data = data_in
			second = False
		else:
			data = data_in

		# make values 2d list of decimal values instead of list of strings
		for line in file:
			float_values = []
			line = line.strip().split()
			for j in range(len(line)):
				float_values.append(float(line[j]))
			values.append(float_values)
		
		values = np.array(values)

		# Place input values as the first element in each interior list
		# which needs to be created upon first run   
		if not old_input: 	
			for i in range(len(inputs)):
				# Place input name in array
				input_array = [inputs[i]]
				# data_in.append(input_array)
				# data_out.append(input_array)
				data.append(input_array)
			
		# Place input values from linearized and outlying data
		for i in range(len(inputs)):
			if filename[0] == "o":
				data[i].append(None)
				continue

			# data_in[i].append(values_in[:, i])
			# data_out[i].append(values_out[:, i])
			data[i].append(values[:,i])
		
		old_input = inputs
	return data_in, data_out
	# return data

'''
Sift through the 2D list for two sets of results:
	- one where the clus_ENG_CALIB_TOT vs. clusterECalib is linearized
	- one where these relations have truth energy < 0.1 GeV

Produce two csv files with these selections for lowE, midE, highE, and all.

Plot the histograms for the input value spread from each selection of each energy range.
'''
def sift(inputs, file_info):
	difference_range	= 0.4 # Choose all values +/- 0.4 from the line
	clus_ENG_CALIB_TOT	= 0
	clusterECalib		= 0

	for i in range(len(inputs)):
		if inputs[i] == "cluster_ENG_CALIB_TOT":
			clus_ENG_CALIB_TOT = i
		if inputs[i] == "clusterECalib":
			clusterECalib = i


	# Sift through the data and only add it back if it meets the requirements
	linearized	= []
	outliers	= []
	for info in file_info:
		if abs(info[clusterECalib] - info[clus_ENG_CALIB_TOT]) < difference_range:
			linearized.append(info)
		if info[clus_ENG_CALIB_TOT] < 0.1:
			outliers.append(info)
	# print(linearized)
	# print()
	# print()
	# print(outliers)
	linearized = np.array(linearized)
	outliers = np.array(outliers)
	return linearized, outliers
	
def sift_files(path, filenames):
	for filename in filenames:
		df = pd.read_csv(path + filename, sep=" ")
		file = open(path + filename, "r")
		top = file.readline().strip().split(" ")
		length = len(top)
		file.close()

		linear	= 	True
		outlier	= 	True
		if linear:
			print("Sifting for initial linearizations")
			print(df.size / length)
			df = df[(df["cluster_ENG_CALIB_TOT"] - df["clusterECalib"]).abs() < 0.2]
			print(df.size / length)
			df.to_csv("low-timing-results/in_" + filename, sep=" ", index=False)
			df = pd.read_csv(path + filename, sep=" ")
		if outlier:
			print("Sifting outlying GeV")
			print(df.size / length)
			df = df[(df["cluster_ENG_CALIB_TOT"] < 0.2)]
			print(df.size / length)
			df.to_csv("low-timing-results/out_" + filename, sep=" ", index=False)

'''
TO RUN
First run by sifting the two files into separate csv's with the separate constrictions.
python input-histograms.py --sift --rangeE range
Make sure out run for both the linear and outlier ranges turnes on separately

The run without the sifting option in the correct range to generate the histograms.
python input-histograms.py --rangeE range
'''
def  main():
	parser = argparse.ArgumentParser(description='Prepare CSV files for MLClusterCalibration')
	parser.add_argument('--sift', dest='sift', action='store_const', const=True, default=False, help='Select to sift csv files')
	parser.add_argument('--normalize', dest='normalize', action='store_const', const=True, default=False, help='Select to sift csv files')
	parser.add_argument('--rangeE', dest='rangeE', type=str, default='', help='range in energy')

	args = parser.parse_args()
	rangeE = args.rangeE
	path = "/home/opitcl/calo-jad/MLClusterCalibration/histograms/low-timing-results/"
	if args.sift:
		files = ["plot_all.csv", "plot_lowE.csv", "plot_midE.csv", "plot_highE.csv"]
		filenames = []
		for f in files:
			if rangeE != '' and f == "plot_" + rangeE + ".csv":
				filenames.append(f)
		if not rangeE:
			filenames = files
					
		sift_files(path, filenames)
		print("Finished sifting files")
	else:
		files = ["out_plot_all.csv", "out_plot_lowE.csv", "out_plot_midE.csv", "out_plot_highE.csv",
	"in_plot_all.csv", "in_plot_lowE.csv", "in_plot_midE.csv", "in_plot_highE.csv"]
		filenames = []
		for f in files:
			if rangeE != '' and f == "in_plot_" + rangeE + ".csv" or f == "out_plot_" + rangeE + ".csv":
				filenames.append(f)
		if not rangeE:
			filenames = files
		info, out = makeList(path, filenames)
		# print(info)
		
		print("Creating histograms...")	
		for i in range(len(info)):
			for j in range(len(info[i])):
				plot(rangeE, info[i], out[i], filenames, args.normalize)
					
main()
