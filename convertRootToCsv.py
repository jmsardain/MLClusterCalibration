import uproot as ur
import pandas as pd
import numpy  as np
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import  QuantileTransformer


def splitDataframe(df, cutoff=0.8):
    cut = int(len(df.index)*.8)
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    return train, test


def transDataframe(df, column_names, prefix=""):
    a = -1
    b = 1
    res = pd.DataFrame(columns=column_names)
    if prefix=="test":
        print("Take these minima and maxima for test.csv, and implement them in the plotting code to get f^{-1}")
    for i in df.columns:
        arr = np.array(df[i].values)
        if i == "clusterE" or i == "cluster_FIRST_ENG_DENS":
            arr = np.log(arr)

        brr = [[i] for i in arr]
        # minValue = np.quantile(arr, 0.02) # np.min(arr)
        # maxValue = np.quantile(arr, 0.98) # np.max(arr)
        # minValue = np.min(arr)
        # maxValue = np.max(arr)
        #maskmin = arr > minValue
        #maskmax = arr < maxValue
        #arr = arr[maskmin & maskmax]
        # if prefix=="test":
        #     print("minmax{}=[{}, {}]".format(i, minValue, maxValue))
        # -- transformation found in https://www.dropbox.com/s/kqynnef5y2nelvm/ProjectNotes.pdf?dl=0
        # newcol = a + (b-a) * (arr - minValue) / (maxValue - minValue)
        quantile = QuantileTransformer(output_distribution='normal')
        data_trans = quantile.fit_transform(brr)
        newcol = data_trans.flatten()
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
    parser.add_argument('--norm', dest='norm', action='store_const', const=True, default=False, help='Transform input features')
    parser.add_argument('--nentries', dest='nentries', type=int, default=0, help='random selection of events from df')
    args = parser.parse_args()

    # -- Start
    filename="data/JZ.topo-cluster.root"
    file = ur.open(filename)
    tree = file["ClusterTree"]
    # -- Select only Truth cluster energy > 0.3
    for variation in range(0, 4):
        df = tree.arrays(library="pd")
        if args.nentries > 0:
            df = df.sample(n = args.nentries)
        # variation 0: low energy
        # variation 1: mid energy
        # variation 2: high energy
        if   variation == 0:
            #df = df[(df["cluster_ENG_CALIB_TOT"] > 0.3) & (df["cluster_ENG_CALIB_TOT"] < 1)]
            df = df[(df["clusterE"] > 0.3) & (df["clusterE"] < 1)]
            rangeE = "lowE"
        elif variation == 1:
            #df = df[(df["cluster_ENG_CALIB_TOT"] >= 1) & (df["cluster_ENG_CALIB_TOT"] < 5)]
            df = df[(df["clusterE"] >= 1) & (df["clusterE"] < 5)]
            rangeE = "midE"
        elif variation == 2:
            #df = df[df["cluster_ENG_CALIB_TOT"] >= 5]
            df = df[df["clusterE"] >= 5]
            rangeE = "highE"
        elif variation == 3:
            #df = df[df["cluster_ENG_CALIB_TOT"] >= 0.3]
            df = df[df["clusterE"] >= 0.3]
            rangeE = "all"
        else:
            print("Not defined")
            return
        print(variation)
        #df = df[df["cluster_FIRST_ENG_DENS"] !=0] ## remove variables that would give inf when logged

        # -- Add response
        resp = np.array( df.clusterE.values ) /  np.array( df.cluster_ENG_CALIB_TOT.values )
        df["r_e_calculated"] = resp

        # -- Define train and test (train is 0.8 of whole root file, test is the rest)
        train, test = splitDataframe(df, cutoff=args.cutoff)
        print("I am here")
        # -- Select the needed columns for training and testing
        # column_names = ['clusterE', 'clusterECalib', 'clusterEtaCalib', 'cluster_ENG_CALIB_TOT',
        #                 'cluster_CENTER_LAMBDA', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
        #                 'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD',
        #                 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE', 'nPrimVtx', 'avgMu', 'r_e_calculated'
        #                 ]

        column_names = ['r_e_calculated',
                        'nPrimVtx', 'avgMu',
                        'clusterE', 'clusterPt', 'clusterPhi', 'cluster_MASS', 'cluster_sumCellE',
                        'cluster_time', 'cluster_fracE', 'cluster_PTD', 'cluster_ISOLATION',
                        'cluster_FIRST_ETA', 'cluster_FIRST_PHI', 'cluster_FIRST_ENG_DENS',
                        'cluster_SECOND_TIME', 'cluster_SECOND_R', 'cluster_SECOND_LAMBDA', 'cluster_SECOND_ENG_DENS',
                        'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG', 'cluster_CENTER_X', 'cluster_CENTER_Y', 'cluster_CENTER_Z',
                        'cluster_ENG_BAD_CELLS', 'cluster_ENG_BAD_HV_CELLS', 'cluster_ENG_FRAC_EM', 'cluster_ENG_FRAC_MAX', 'cluster_ENG_FRAC_CORE', 'cluster_ENG_POS',
                        'cluster_DELTA_THETA', 'cluster_DELTA_PHI',
                        'cluster_CELL_SIGNIFICANCE', 'cluster_CELL_SIG_SAMPLING',
                        'cluster_N_BAD_CELLS', 'cluster_BAD_CELLS_CORR_E',
                        'cluster_LONGITUDINAL', 'cluster_LATERAL', 'cluster_SIGNIFICANCE',
                        'nCluster', 'cluster_N_BAD_HV_CELLS', 'cluster_nCells', 'cluster_nCells_tot',
                        'cluster_ENG_CALIB_TOT', 'clusterECalib',
                        ]

        # column_names = ['clusterE', 'clusterPt', 'clusterPhi', 'cluster_SECOND_R',
        #                 'clusterECalib', 'cluster_ENG_CALIB_TOT', 'r_e_calculated',
        #                 'cluster_nCells_tot', 'cluster_SIGNIFICANCE',
        #                 'cluster_time', 'cluster_sumCellE', 'cluster_nCells', 'cluster_ENG_POS',
        #                 'cluster_FIRST_ENG_DENS', 'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG',
        #                 'cluster_PTD', 'cluster_CELL_SIG_SAMPLING',
        #                 'cluster_SECOND_ENG_DENS', 'cluster_CELL_SIGNIFICANCE', 'cluster_MASS',
        #                 'cluster_ENG_FRAC_MAX', 'cluster_ENG_FRAC_EM',
        #                 'nPrimVtx', 'clusterEtaCalib',
        #                 ]

        # column_names = ['avgMu', 'nPrimVtx',
        #                 'nCluster', 'clusterIndex', 'cluster_nCells', 'cluster_nCells_tot', 'clusterE', 'clusterPt',
        #                 'clusterPhi', 'cluster_sumCellE', 'cluster_time', 'cluster_fracE', 'clusterEtaCalib',
        #                 'cluster_EM_PROBABILITY', 'cluster_HAD_WEIGHT', 'cluster_OOC_WEIGHT', 'cluster_DM_WEIGHT',
        #                 'cluster_ENG_CALIB_TOT', 'clusterECalib',
        #                 'cluster_CENTER_MAG', 'cluster_FIRST_ENG_DENS', 'cluster_FIRST_PHI',
        #                 'cluster_FIRST_ETA', 'cluster_SECOND_R', 'cluster_SECOND_LAMBDA',
        #                 'cluster_DELTA_PHI', 'cluster_DELTA_THETA', 'cluster_DELTA_ALPHA',
        #                 'cluster_CENTER_X', 'cluster_CENTER_Y', 'cluster_CENTER_Z',
        #                 'cluster_CENTER_LAMBDA', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
        #                 'cluster_ENG_FRAC_EM', 'cluster_ENG_FRAC_MAX',
        #                 'cluster_ENG_FRAC_CORE', 'cluster_SECOND_ENG_DENS',
        #                 'cluster_ISOLATION', 'cluster_ENG_BAD_CELLS', 'cluster_N_BAD_CELLS',
        #                 'cluster_N_BAD_CELLS_CORR', 'cluster_BAD_CELLS_CORR_E',
        #                 'cluster_BADLARQ_FRAC', 'cluster_ENG_POS', 'cluster_SIGNIFICANCE',
        #                 'cluster_CELL_SIGNIFICANCE', 'cluster_CELL_SIG_SAMPLING',
        #                 'cluster_AVG_LAR_Q', 'cluster_AVG_TILE_Q',
        #                 'cluster_ENG_BAD_HV_CELLS', 'cluster_N_BAD_HV_CELLS', 'cluster_PTD',
        #                 'cluster_MASS', 'cluster_SECOND_TIME', 'r_e_calculated']
        df_train = train[column_names]
        df_test  = test[column_names]

        # -- Sanity cuts (if not done, this variable gives inf when logged)
        df_train = df_train[(df_train["cluster_FIRST_ENG_DENS"] != 0) & (df_train["cluster_ENG_CALIB_TOT"] >= 0.3)]
        df_test  = df_test[(df_test["cluster_FIRST_ENG_DENS"] != 0)   & (df_test["cluster_ENG_CALIB_TOT"]  >= 0.3) ]

        print("I am here again")
        # -- Get min and max and transform dataframes (not sure if we have to do it for each dataframe or for the original dataframe)
        if args.norm:
            train = transDataframe(df_train, column_names)
            test  = transDataframe(df_test,  column_names, "test")
        else:
            train = df_train
            test = df_test

        # -- Make plots
        if args.plot:
            plot(df_train, train, prefix="train")
            plot(df_test,  test,  prefix="test")
        print("I am here again and again")
        # -- Save dataframes
        train.to_csv("train_{}.csv".format(rangeE), index=False)
        test.to_csv("test_{}.csv".format(rangeE),   index=False)

    return

if __name__ == "__main__":
    main()
