import ROOT
import numpy as np
import pandas as pd

def GetMinMax(vec):
    min = np.min(vec)
    max = np.max(vec)
    return min, max

def main():
    df = pd.read_csv("/home/jmsardain/JetCalib/FinalPlots/plot.csv", sep=' ')
    #print(df.columns)

    for i in df.columns:
        if i=="r_e_calculated" or i=="r_e_predec":
            continue
        arr = df[i]
        min, max = GetMinMax(arr)
        h2Calc       = ROOT.TH2D("", "", 1000, min, max, 100, 0, 2)
        hProfileCalc = ROOT.TProfile("", "", 1000, min, max, 0, 2)
        h2Pred       = ROOT.TH2D("", "", 1000, min, max, 100, 0, 2)
        hProfilePred = ROOT.TProfile("", "", 1000, min, max, 0, 2)
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
    htruthClusTotalE = ROOT.TH1D("", "", 1000, mintruthClusTotalE, maxtruthClusTotalE)
    hpredClusTotalE  = ROOT.TH1D("", "", 1000, minpredClusTotalE,  maxpredClusTotalE);
    # -- Fill histograms
    for iter in range(len(truthClusTotalE)):
        htruthClusTotalE.Fill(truthClusTotalE[iter])
        hpredClusTotalE.Fill(predClusTotalE[iter])
        pass
    # -- Define legend
    leg = ROOT.TLegend(0.1, 0.1, 0.5, 0.3);
    leg.AddEntry(htruthClusTotalE, "TruthClusterTotalE", "l");
    leg.AddEntry(hpredClusTotalE,  "PredClusterTotalE",  "l");
    c1.cd()
    htruthClusTotalE.SetLineColor(4)
    hpredClusTotalE.SetLineColor(2)
    c1.SetLogy()
    c1.SetLogx()
    htruthClusTotalE.Draw()
    hpredClusTotalE.Draw("same")
    leg.Draw()
    c1.SaveAs("./plots/Comparison_Truth_Pred.png")

    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
