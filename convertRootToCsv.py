import uproot as ur
import pandas as pd
import numpy  as np
import argparse
import matplotlib.pyplot as plt

def splitDataframe(df, cutoff=0.8):
    cut = int(len(df.index)*.8)
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    return train, test


def transDataframe(df, column_names):
    a = -1
    b = 1
    res = pd.DataFrame(columns=column_names)
    for i in df.columns:
        arr = np.array(df[i].values)
        if i == "clusterE" or i == "cluster_FIRST_ENG_DENS":
            arr = np.log(arr)
        minValue = np.quantile(a, 0.01) # np.min(arr)
        maxValue = np.quantile(a, 0.99) # np.max(arr)
        # -- transformation found in https://www.dropbox.com/s/kqynnef5y2nelvm/ProjectNotes.pdf?dl=0
        if i != "r_e_calculated":
            newcol = a + (b-a) * (arr - minValue) / (maxValue - minValue)
        if i == "r_e_calculated": ## do not make any transformation
            newcol = arr
        res[i] = newcol

    return res

def plot(df_before, df_after, prefix=""):

    for i in df_before.columns:
        # -- Get arrays
        arr_before = np.array(df_before[i].values)
        arr_after  = np.array(df_after[i].values)

        # -- Define 100 bins between min and max
        bins_before = np.linspace(np.min(arr_before), np.max(arr_before), 101, endpoint=True)
        bins_after  = np.linspace(np.min(arr_after),  np.max(arr_after), 101, endpoint=True)

        fig, ax = plt.subplots(2, 1)
        # -- Up: distribution before transformation
        ax[0].hist(arr_before, bins=bins_before, color='b', alpha=0.75)
        ax[0].set_yscale('log')
        ax[0].set_xlabel(i)
        ax[0].set_ylabel('Count')

        # -- Down: distribution after transformation
        ax[1].hist(arr_after, bins=bins_after, color='r', alpha=0.75)
        ax[1].set_yscale('log')
        ax[1].set_xlabel(i)
        ax[1].set_ylabel('Count')

        # Show the graph
        fig.tight_layout()
        plt.savefig('plots/'+prefix+"_"+i+'.png')
        plt.clf()

    pass


def main():

    parser = argparse.ArgumentParser(description='Prepare CSV files for MLClusterCalibration')
    parser.add_argument('--plot', dest='plot', action='store_const', const=True, default=False, help='Save plots (default: False)')
    parser.add_argument('--cutoff', dest='cutoff', type=float, default=0.8, help='Train / Total dataset (default: 0.8)')
    args = parser.parse_args()

    # -- Start
    filename="data/JZ.topo-cluster.root"
    file = ur.open(filename)
    tree = file["ClusterTree"]
    df = tree.arrays(library="pd")
    # -- Select only Truth cluster energy > 0.3
    df = df[df["cluster_ENG_CALIB_TOT"] > 0.3]
    #df = df[df["cluster_FIRST_ENG_DENS"] !=0] ## remove variables that would give inf when logged

    # -- Add response
    resp = np.array( df.clusterE.values ) /  np.array( df.cluster_ENG_CALIB_TOT.values )
    df["r_e_calculated"] = resp

    # -- Define train and test (train is 0.8 of whole root file, test is the rest)
    train, test = splitDataframe(df, cutoff=args.cutoff)

    # -- Select the needed columns for training and testing
    column_names = ['clusterE', 'clusterEtaCalib', 'cluster_ENG_CALIB_TOT', 'cluster_CENTER_LAMBDA', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE', 'nPrimVtx', 'avgMu', 'r_e_calculated']
    df_train = train[column_names]
    df_test  = test[column_names]

    # -- Sanity cuts (if not done, this variable gives inf when logged)
    df_train = df_train[df_train["cluster_FIRST_ENG_DENS"] != 0]
    df_test  = df_test[df_test["cluster_FIRST_ENG_DENS"] != 0]

    # -- Get min and max and transform dataframes (not sure if we have to do it for each dataframe or for the original dataframe)
    #train = transDataframe(df_train, column_names)
    #test  = transDataframe(df_test,  column_names)
    train = df_train
    test = df_test
    # -- Make plots
    if args.plot:
        plot(df_train, train, prefix="train")
        plot(df_test, test, prefix="test")
    # -- Save dataframes
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

    return

if __name__ == "__main__":
    main()
