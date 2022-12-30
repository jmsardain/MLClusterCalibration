


def main():
	files = ["train_lowE.csv", "train_midE.csv", "train_highE.csv", "train_all.csv", 
		"test_lowE.csv", "test_midE.csv", "test_highE.csv", "test_all.csv",
		"./FinalPlots/plot_lowE.csv", "./FinalPlots/plot_highE.csv", "./FinalPlots/plot_midE.csv",
		"./FinalPlots/plot_all.csv"]
	for fname in files:
		f = open(fname, "r")
		print(fname, "length of file:", len(f.readlines()))
		f.close()

main()
