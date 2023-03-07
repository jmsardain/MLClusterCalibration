import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

def plotRealPredict(values, trans,  min, max, label):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    bins = np.linspace(min, max, 21, endpoint=True)
    ax1.hist(values, bins=bins, color = 'r', label='original', alpha=0.4)
    ax2.hist(trans, bins=bins, color = 'r', label='transformed', alpha=0.4)
    ax1.set_xlabel(label)
    ax2.set_xlabel(label)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(1, 1e6)
    plt.savefig('fig/'+label+'.png')
    plt.clf()


def main():


    df = pd.read_csv("all_info_df.csv", sep=" ")
    df["clusterE"] = np.log(df.clusterE.values)
    df["cluster_FIRST_ENG_DENS"] = np.log(df.cluster_FIRST_ENG_DENS.values)
    df["cluster_CENTER_LAMBDA"] = np.log(df.cluster_CENTER_LAMBDA.values)
    # df["cluster_PTD"] = np.log(df.cluster_PTD.values)

    trans = QuantileTransformer(n_quantiles=100, output_distribution='normal')
    for i in df.columns:
        if i=='r_e_calculated':
            plotRealPredict(df[i], df[i], np.min(df[i]), np.max(df[i]), i)
        brr = [[i] for i in df[i]]
        datatrans = trans.fit_transform(brr).flatten()
        plotRealPredict(df[i], datatrans, np.min(df[i]), np.max(df[i]), i)


    #
    #
    ## divide dataset into :
    ## 60% train
    ## 20% validation
    ## 20% test
    train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
    train.to_csv("train.csv", sep=" ", index=False)
    validate.to_csv("vaidate.csv", sep=" ", index=False)
    test.to_csv("test.csv", sep=" ", index=False)

    train_np    = np.genfromtxt("train.csv", delimiter=" ", skip_header=True)
    validate_np = np.genfromtxt("vaidate.csv", delimiter=" ", skip_header=True)
    test_np     = np.genfromtxt("test.csv", delimiter=" ", skip_header=True)

    np.save('data_train.npy', train_np)
    np.save('data_validate.npy', validate_np)
    np.save('data_test.npy', test_np)
    return


if __name__ == "__main__":
    main()
