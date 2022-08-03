import ROOT
import numpy as np
import pandas as pd

def GetMinMax(vec):
    min = np.min(vec)
    max = np.max(vec)
    return min, max

def BinLogX(h):
    axis = h.GetXaxis()
    bins = axis.GetNbins()

    a = axis.GetXmin()
    b = axis.GetXmax()
    width = (b-a) / bins
    newbins = np.zeros([bins + 1])
    for i in range(bins+1):
        newbins[i] = np.power(10, a + i * width)

    axis.Set(bins, newbins)
    del newbins
    pass


def main():
    ROOT.gStyle.SetOptStat(0)
    df = pd.read_csv("/home/jmsardain/JetCalib/FinalPlots/plot.csv", sep=' ')
    #print(df.columns)

    for i in df.columns:
        if i!="cluster_ENG_CALIB_TOT":
            continue
        if i=="r_e_calculated" or i=="r_e_predec":
            continue
        arr = df[i]
        min, max = GetMinMax(arr)
        if i == "cluster_ENG_CALIB_TOT":
            min, max = np.log(min), np.log(max)
        h2Calc       = ROOT.TH2D("", "", 100, -1, 3, 100, 0, 2)
        BinLogX(h2Calc)
        hProfileCalc = ROOT.TProfile("", "", 100, -1, 3, 0, 2)
        BinLogX(hProfileCalc)
        h2Pred       = ROOT.TH2D("", "", 100,-1, 3, 100, 0, 2)
        BinLogX(h2Pred)
        hProfilePred = ROOT.TProfile("", "", 100, -1, 3, 0, 2)
        BinLogX(hProfilePred)

        # -- Get re_calculated
        reCalc = df["r_e_calculated"]
        # -- Get re_pred
        rePred = df["r_e_predec"]

        # -- Start filling histograms
        for iter in range(len(arr)):
            h2Calc.Fill(arr[iter], reCalc[iter])
            hProfileCalc.Fill(arr[iter], reCalc[iter])
            h2Pred.Fill(arr[iter], rePred[iter])
            hProfilePred.Fill(arr[iter], rePred[iter])
            pass

        # -- Start by drawing 2D histo with re_calc
        c = ROOT.TCanvas("", "", 500, 500)
        c2 = ROOT.TCanvas("", "", 500, 500)
        c.cd()
        c.SetLogz()
        if i == "clusterE" or i == "cluster_ENG_CALIB_TOT":
            c.SetLogx()
        h2Calc.Draw("colz")
        hProfileCalc.SetLineColor(2)
        hProfileCalc.Draw("same")
        c.SaveAs("./plots/2DPlots/Calc_"+i+".png")
        # -- Draw 2D histos wih re_pred
        h2Pred.Draw("colz")
        hProfilePred.SetLineColor(2)
        hProfilePred.Draw("same")
        c.SaveAs("./plots/2DPlots/Pred_"+i+".png")
        # -- Draw both calc and prediction and make ratio plot
        c2.cd()
        upperPad = ROOT.TPad("pad1", "pad1",0.05,0.35,0.95,0.95)
        upperPad.SetTopMargin(0.05)
        upperPad.SetBottomMargin(0.01)
        upperPad.SetLeftMargin(0.10)
        upperPad.SetRightMargin(0.05)
        upperPad.Draw()
        upperPad.cd()
        ROOT.SetOwnership(upperPad, False)

        c2.cd()
        lowerPad = ROOT.TPad("pad2", "pad2", 0.05,0.05,0.95,0.35)
        lowerPad.SetBottomMargin(0.4)
        lowerPad.SetTopMargin(-0.05)
        lowerPad.SetLeftMargin(0.10)
        lowerPad.SetRightMargin(0.05)
        lowerPad.Draw()
        ROOT.SetOwnership(lowerPad, False)

        upperPad.cd()
        hProfileCalc.SetLineColor(4)
        hProfilePred.SetLineColor(2)
        hProfileCalc.Draw()
        hProfilePred.Draw("same")
        lowerPad.cd()
        h = hProfileCalc.Clone()
        h.Divide(hProfilePred)
        h.GetYaxis().SetTitle("Calc / Pred")
        h.Draw()
        c2.SaveAs("./plots/CalcvsPred/CalcvsPred_"+i+".png")



    # -- Now let's compare truthClusTotalE and predClusTotalE
    c1 = ROOT.TCanvas("", "", 500, 500)
    # -- Get the arrays
    truthClusTotalE = np.array(df["cluster_ENG_CALIB_TOT"])
    predClusTotalE  = np.array(df["cluster_ENG_TOT_frompred"])
    # -- Get min and max for histograms
    mintruthClusTotalE, maxtruthClusTotalE = GetMinMax(truthClusTotalE)
    minpredClusTotalE, maxpredClusTotalE = GetMinMax(predClusTotalE)
    # -- Define histograms
    htruthClusTotalE = ROOT.TH1D("", "", 100, -1, 3)
    hpredClusTotalE  = ROOT.TH1D("", "", 100, -1,  3);
    BinLogX(htruthClusTotalE)
    BinLogX(hpredClusTotalE)

    # -- Fill histograms
    for iter in range(len(truthClusTotalE)):
        htruthClusTotalE.Fill(truthClusTotalE[iter])
        hpredClusTotalE.Fill(predClusTotalE[iter])
        pass
    # -- Define legend
    leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9);
    leg.AddEntry(htruthClusTotalE, "TruthClusterTotalE", "l");
    leg.AddEntry(hpredClusTotalE,  "PredClusterTotalE",  "l");
    c1.cd()
    upperPad1 = ROOT.TPad("pad1", "pad1",0.05,0.35,0.95,0.95)
    upperPad1.SetTopMargin(0.05)
    upperPad1.SetBottomMargin(0.01)
    upperPad1.SetLeftMargin(0.10)
    upperPad1.SetRightMargin(0.05)
    upperPad1.Draw()
    upperPad1.cd()
    ROOT.SetOwnership(upperPad1, False)

    c1.cd()
    lowerPad1 = ROOT.TPad("pad2", "pad2", 0.05,0.05,0.95,0.35)
    lowerPad1.SetBottomMargin(0.4)
    lowerPad1.SetTopMargin(-0.05)
    lowerPad1.SetLeftMargin(0.10)
    lowerPad1.SetRightMargin(0.05)
    lowerPad1.Draw()
    ROOT.SetOwnership(lowerPad1, False)

    upperPad1.cd()
    upperPad1.SetLogy()
    upperPad1.SetLogx()
    leg.Draw()
    htruthClusTotalE.SetLineColor(4)
    hpredClusTotalE.SetLineColor(2)
    htruthClusTotalE.Draw()
    hpredClusTotalE.Draw("same")

    lowerPad1.cd()
    lowerPad1.SetLogx()
    h = htruthClusTotalE.Clone()
    h.Divide(hpredClusTotalE)
    h.GetYaxis().SetTitle("Truth / Pred")
    h.GetYaxis().SetRangeUser(0., 2.)
    h.Draw("P")
    c1.SaveAs("./plots/Comparison_Truth_Pred.png")


    # -- Now let's compare truthClusTotalE and predClusTotalE
    c3 = ROOT.TCanvas("", "", 500, 500)
    # -- Get the arrays
    reCalc = np.array(df["r_e_calculated"])
    rePred  = np.array(df["r_e_predec"])
    # -- Get min and max for histograms
    minreCalc, maxreCalc = GetMinMax(reCalc)
    minrePred, maxrePred = GetMinMax(rePred)
    # -- Define histograms
    hreCalc = ROOT.TH1D("", "",  100,  0, 2)
    hrePred  = ROOT.TH1D("", "", 100, 0, 2)
    h2CalcPred = ROOT.TH2D("", "",  100,  0, 2, 100, 0, 2)
    # -- Fill histograms
    for iter in range(len(reCalc)):
        hreCalc.Fill(reCalc[iter])
        hrePred.Fill(rePred[iter])
        h2CalcPred.Fill(reCalc[iter], rePred[iter])
        pass
    # -- Define legend
    leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9);
    leg.AddEntry(hreCalc, "Calculated resp", "l");
    leg.AddEntry(hrePred,  "Predicted resp",  "l");
    c3.cd()
    hreCalc.SetLineColor(4)
    hrePred.SetLineColor(2)
    hrePred.Draw()
    hreCalc.Draw("same")
    leg.Draw()
    c3.SaveAs("./plots/Comparison_Truth_Pred_Resp.png")
    c3.SetLogz()
    h2CalcPred.Draw("colz")
    c3.SaveAs("./plots/Comparison_Truth_Pred_Resp2DPlot.png")

    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
