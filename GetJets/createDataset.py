import pandas as pd
import numpy as np
import uproot as ur
import matplotlib.pyplot as plt
def main():
    file = ur.open("/home/loch/Summer2022/MLTopoCluster/data/Akt4LCTopo.inclusive_topo_cluster.root")
    tree = file["ClusterTree"]
    df = tree.arrays(library="pd")
    # df = df.sample(n = 1000000)

    print("Finished reading and sampling")
    ## group clusters by jet raw E (since jet count is not working)
    # df2 = df.groupby('jetRawE').agg(clusterE=('clusterE',list),
    #                         clusterPt=('clusterPt',list),
    #                         clusterEtaCalib=('clusterEtaCalib',list),
    #                         clusterPhi=('clusterPhi',list),
    #                        ).reset_index()

    df2 = df[df.duplicated('jetRawE', keep=False)].groupby('jetRawE')['clusterE'].apply(list).reset_index()
    print("Done with reco")
    ## group LCW clusters by jet LCW E (since jet count is not working)
    df3 = df[df.duplicated('jetCalE', keep=False)].groupby('jetCalE')['clusterECalib'].apply(list).reset_index()
    print("Done with LCW")
    df4 = df[df.duplicated('jetRawE', keep=False)].groupby('jetRawE')['cluster_ENG_CALIB_TOT'].apply(list).reset_index()
    print("Done with LCW")


    CalcEnergyFromClusters = []
    for listClusE in df2.clusterE.values:
        CalcEnergyFromClusters.append(np.sum(listClusE))
    print("Done with computing sum of reco clusters")

    CalcEnergyFromClusters_LCW = []
    for listClusE in df3.clusterECalib.values:
        CalcEnergyFromClusters_LCW.append(np.sum(listClusE))
    print("Done with computing sum of LCW clusters")

    CalcEnergyFromClusters_Truth = []
    for listClusE in df4.cluster_ENG_CALIB_TOT.values:
        CalcEnergyFromClusters_Truth.append(np.sum(listClusE))
    print("Done with computing sum of truth clusters")

    ## add column that contains the sum of clusters
    df2["CalcEnergy_clus"] = CalcEnergyFromClusters
    df3["CalcEnergy_clus"] = CalcEnergyFromClusters_LCW
    df4["CalcEnergy_clus"] = CalcEnergyFromClusters_Truth

    print("Start saving")

    df2.to_csv("./energy_nocalib.csv", sep=" ")
    df3.to_csv("./energy_LCWcalib.csv", sep=" ")
    df4.to_csv("./energy_truthcalib.csv", sep=" ")


    return



if __name__ == "__main__":
    main()
