import ROOT
import numpy as np
import pandas as pd
import math
from rootplotting import ap
from rootplotting.tools import *
from root_numpy import fill_hist

def BinLogX(h):
    axis = h.GetXaxis()
    bins = axis.GetNbins()

    a = axis.GetXmin()
    b = axis.GetXmax()
    width = (b-a) / bins
    newbins = np.zeros([bins + 1])
    for i in range(bins+1):
        newbins[i] = pow(10, a + i * width)

    axis.Set(bins, newbins)
    del newbins
    pass

def BinLogY(h):
    axis = h.GetYaxis()
    bins = h.GetNbinsY()

    a = axis.GetXmin()
    b = axis.GetXmax()
    width = (b-a) / bins
    newbins1 = np.zeros([bins + 1])
    for i in range(bins+1):
        newbins1[i] = pow(10, a + i * width)

    axis.Set(bins, newbins1)
    del newbins1
    pass

def LogLogTH2D(namecola, namecolb, lowbin, highbin, linearr, xaxisname, yaxisname):
    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    c.pads()[0]._bare().SetLogz()

    xaxis = np.linspace(lowbin, highbin,  100 + 1, endpoint=True)
    yaxis = np.linspace(lowbin, highbin,  100 + 1, endpoint=True)

    h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], yaxis[-1] + 0.55 * (yaxis[-1] - yaxis[0])]))
    h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

    BinLogX(h1_backdrop)
    BinLogY(h1_backdrop)
    BinLogX(h1a)
    BinLogY(h1a)


    cola = np.array(namecola)
    colb = np.array(namecolb)

    mesh1a = np.vstack((cola, colb)).T

    fill_hist(h1a, mesh1a)

    c.hist2d(h1_backdrop, option='AXIS')
    c.hist2d(h1a,         option='COLZ')
    c.hist2d(h1_backdrop, option='AXIS')
    c.ylim(1e-1, 1e4)
    if len(linearr) == 4:
        line = c.line(linearr[0], linearr[1], linearr[2], linearr[3], linecolor=ROOT.kRed, linewidth=2)
    else:
        print("Please define the line as an array [xmin, ymin, xmax, ymax]")
    c.logx()
    c.log()
    c.xlabel(xaxisname)
    c.ylabel(yaxisname)
    c.text(["#sqrt{s} = 13 TeV" ], qualifier='Simulation Internal')
    return c



def Histo1D(jetE, clusE, labeljet='', labelclus='', xaxisname=''):
    c = ap.canvas(num_pads=2, batch=True)
    p0, p1 = c.pads()


    xaxis = np.linspace(-1, 4, 100 + 1, endpoint=True)

    hjet  = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    hclus  = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

    BinLogX(hjet)
    BinLogX(hclus)

    fill_hist(hjet,  jetE)
    fill_hist(hclus, clusE)

    c.hist(hjet, option='HIST', label=labeljet, linecolor=2)
    c.hist(hclus,  option='HIST', label=labelclus,  linecolor=4)
    c.log()
    c.logx()
    p1.ylim(0., 2.)
    p1.logx()
    c.ratio_plot((hjet,  hjet),  option='E2',      linecolor=2) #, oob=True)
    c.ratio_plot((hclus,   hjet),  option='HIST',    linecolor=4) #, oob=True)
    c.xlabel(xaxisname)
    c.ylabel('Events')
    p1.ylabel('{} / {}'.format(labelclus, labeljet))
    c.legend()
    c.text(["#sqrt{s} = 13 TeV"], qualifier='Simulation Internal')

    return c



def main():

    ROOT.gStyle.SetPalette(ROOT.kBird)
    df_reco  = pd.read_csv("energy_nocalib.csv", sep = " ")
    df_lcw   = pd.read_csv("energy_LCWcalib.csv", sep = " ")
    df_truth = pd.read_csv("energy_truthcalib.csv", sep = " ")

    cReco  = LogLogTH2D(df_reco["jetRawE"], df_reco["CalcEnergy_clus"], -1, 4, [1e-1, 1e-1, 1e4, 1e4], "E^{true}_{jet} [GeV]", "#Sigma E^{clus} [GeV]")
    cLCW   = LogLogTH2D(df_lcw["jetCalE"], df_lcw["CalcEnergy_clus"], -1, 4, [1e-1, 1e-1, 1e4, 1e4], "E^{LCW}_{jet} [GeV]", "#Sigma E^{clus} [GeV]")
    cTruth = LogLogTH2D(df_truth["jetRawE"], df_truth["CalcEnergy_clus"], -1, 4, [1e-1, 1e-1, 1e4, 1e4], "E^{true}_{jet} [GeV]", "#Sigma E^{clus} [GeV]")
    cReco.save("./Energy2Dplot_Jet_Clus_Reco.png")
    cLCW.save("./Energy2Dplot_Jet_Clus_LCW.png")
    cTruth.save("./Energy2Dplot_Jet_Clus_Trtuh.png")



    c1Reco  = Histo1D(df_reco["jetRawE"], df_reco["CalcEnergy_clus"],   labeljet='E^{true}_{jet}', labelclus='#Sigma E^{clus}', xaxisname='Energy [GeV]')
    c1LCW   = Histo1D(df_lcw["jetCalE"], df_lcw["CalcEnergy_clus"],     labeljet='E^{LCW}_{jet}', labelclus='#Sigma E^{clus}', xaxisname='Energy [GeV]')
    c1Truth = Histo1D(df_truth["jetRawE"], df_truth["CalcEnergy_clus"], labeljet='E^{true}_{jet}', labelclus='#Sigma E^{clus}', xaxisname='Energy [GeV]')
    c1Reco.save("./Energy_Jet_Clus_Reco.png")
    c1LCW.save("./Energy_Jet_Clus_LCW.png")
    c1Truth.save("./Energy_Jet_Clus_Truth.png")

    return



if __name__ == "__main__":
    main()
