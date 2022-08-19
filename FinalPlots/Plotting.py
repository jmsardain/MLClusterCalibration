import ROOT
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mode
import math
import argparse
from rootplotting import ap
from rootplotting.tools import *

parser = argparse.ArgumentParser(description='Final plotting code')
parser.add_argument('--rangeE', dest='rangeE', required=True, type=str, default='', help='range in energy')
args = parser.parse_args()


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

def main():

    if args.rangeE == 'all':
        df = pd.read_csv("/home/jmsardain/JetCalib/FinalPlots/plot.csv", sep=' ')
    else:
        df = pd.read_csv("/home/jmsardain/JetCalib/FinalPlots/plot_{}.csv".format(args.rangeE), sep=' ')

    #print(df.columns)

    if args.rangeE == 'lowE':
        rangeEnergy = '0.3 #leq E < 1 GeV'
    elif args.rangeE =='midE':
        rangeEnergy = '1 #leq E < 5 GeV'
    elif args.rangeE == 'highE':
        rangeEnergy = 'E #geq 5 GeV'
    elif args.rangeE == 'all':
        rangeEnergy = 'E #geq 0.3 GeV'
    else:
        print("Precise range in code and rerun")
        return

    EnergyNoLogE         = False
    EnergyLogE           = False
    Reponse2DPlot        = False
    Reponse1DPlot        = False
    RatioVsInputFeatures = True

    # -- Plot cluster energy (truth, predicted, ) with no log scale for x axis
    if EnergyNoLogE:

        corr, _ = pearsonr(df["cluster_ENG_TOT_frompred"], df["cluster_ENG_CALIB_TOT"])

        c = ap.canvas(num_pads=2, batch=True)
        p0, p1 = c.pads()

        xaxis = np.linspace(0, 2, 100 + 1, endpoint=True)

        if args.rangeE == 'lowE':
            xaxis = np.linspace(0., 1.2, 100 + 1, endpoint=True)
        elif args.rangeE =='midE':
            xaxis = np.linspace(0.5, 10.5, 100 + 1, endpoint=True)
        elif args.rangeE == 'highE':
            xaxis = np.linspace(0, 1000, 100 + 1, endpoint=True)
        elif args.rangeE == 'all':
            xaxis = np.linspace(0, 1000, 100 + 1, endpoint=True)
        else:
            print("Precise range in code and rerun")

        # -- Now let's compare truthClusTotalE and predClusTotalE
        # -- Get the arrays
        truthClusTotalE = np.array(df["cluster_ENG_CALIB_TOT"])
        predClusTotalE  = np.array(df["cluster_ENG_TOT_frompred"])
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            calibClusTotalE = np.array(df["clusterECalib"])

        # -- Define histograms
        htruthClusTotalE = c.hist(truthClusTotalE, bins=xaxis, option='HIST', label='Truth total E',      linecolor=2)
        hpredClusTotalE  = c.hist(predClusTotalE,  bins=xaxis, option='HIST', label='Predicted total E',  linecolor=4)
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            hcalibClusTotalE = c.hist(calibClusTotalE, bins=xaxis, option='HIST', label='Calibrated total E', linecolor=8)

        p1.ylim(0., 2.)
        c.ratio_plot((htruthClusTotalE,  htruthClusTotalE),  option='E2',      linecolor=2) #, oob=True)
        c.ratio_plot((hpredClusTotalE,   htruthClusTotalE),  option='HIST',    linecolor=4) #, oob=True)
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            c.ratio_plot((hcalibClusTotalE,  htruthClusTotalE),  option='HIST',    linecolor=8) #, oob=True)


        p1.yline(1.0)
        # if args.rangeE == 'highE' or args.rangeE == 'all':
        #     p1.logx()
        #     c.logx()
        c.xlabel('Cluster energy [GeV]')
        c.ylabel('Events')
        p1.ylabel('Variation / Truth')

        c.legend()
        c.log()
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) ), "Correlation: %.4f " % (corr)], qualifier='Simulation Internal')
        c.save("./plots/{}/Comparison_Truth_Pred.png".format(args.rangeE))

    if EnergyLogE:
        corr, _ = pearsonr(df["cluster_ENG_TOT_frompred"], df["cluster_ENG_CALIB_TOT"])

        c = ap.canvas(num_pads=2, batch=True)
        p0, p1 = c.pads()
        #corr0p8_1, _ = pearsonr(df_0p8_1["r_e_calculated"], df_0p8_1["r_e_predec"])

        if args.rangeE == 'lowE':
            xaxis = np.linspace(-1, 1, 100 + 1, endpoint=True)
        elif args.rangeE =='midE':
            xaxis = np.linspace(-1, 2, 100 + 1, endpoint=True)
        elif args.rangeE == 'highE':
            xaxis = np.linspace(-1, 3, 100 + 1, endpoint=True)
        elif args.rangeE == 'all':
            xaxis = np.linspace(-1, 3, 100 + 1, endpoint=True)
        else:
            print("Precise range in code and rerun")

        h1_backdrop      = ROOT.TH1F('', '', 1, np.array([xaxis[0], xaxis[-1]]))
        htruthClusTotalE = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        hpredClusTotalE  = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        hcalibClusTotalE = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        htruthpred       = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        htruthcalib      = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

        BinLogX(h1_backdrop)
        BinLogX(htruthClusTotalE)
        BinLogX(hpredClusTotalE)
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            BinLogX(hcalibClusTotalE)
        BinLogX(htruthpred)
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            BinLogX(htruthcalib)

        # -- Now let's compare truthClusTotalE and predClusTotalE
        # -- Get the arrays
        truthClusTotalE = np.array(df["cluster_ENG_CALIB_TOT"])
        predClusTotalE  = np.array(df["cluster_ENG_TOT_frompred"])
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            calibClusTotalE = np.array(df["clusterECalib"])

        for iter in range(len(truthClusTotalE)):
            htruthClusTotalE.Fill(truthClusTotalE[iter])
            hpredClusTotalE.Fill(predClusTotalE[iter])
            if args.rangeE == 'lowE' or args.rangeE == 'highE':
                hcalibClusTotalE.Fill(calibClusTotalE[iter])
            htruthpred.Fill(truthClusTotalE[iter]  / predClusTotalE[iter])
            if args.rangeE == 'lowE' or args.rangeE == 'highE':
                htruthcalib.Fill(truthClusTotalE[iter] / calibClusTotalE[iter])
        # -- Define histograms
        c.hist(h1_backdrop, option='AXIS')
        c.hist(htruthClusTotalE,  option='HIST', label='Truth total E',      linecolor=2)
        c.hist(hpredClusTotalE,   option='HIST',  label='Predicted total E',   linecolor=4)
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            c.hist(hcalibClusTotalE,  option='HIST', label='Calibrated total E', linecolor=8)
        c.hist(h1_backdrop, option='AXIS')



        p1.ylim(0., 2.)
        p1.logx()
        c.ratio_plot((htruthClusTotalE,  htruthClusTotalE),  option='E2',      linecolor=2) #, oob=True)
        c.ratio_plot((hpredClusTotalE,   htruthClusTotalE),  option='HIST',    linecolor=4) #, oob=True)
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            c.ratio_plot((hcalibClusTotalE,  htruthClusTotalE),  option='HIST',    linecolor=8 )#, oob=True)

        c.xlabel('Cluster energy [GeV]')
        c.ylabel('Events')
        p1.ylabel('Variation / Truth')

        c.legend()
        c.log()
        c.logx()
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) ), "Correlation: %.4f " % (corr)], qualifier='Simulation Internal')
        c.save("./plots/{}/Comparison_Truth_Pred_Log.png".format(args.rangeE))


    if Reponse2DPlot:
        c = ap.canvas(batch=True, size=(600,600))
        c.pads()[0]._bare().SetRightMargin(0.2)
        c.pads()[0]._bare().SetLogz()
        xaxis = np.linspace(0, 2,  100 + 1, endpoint=True)
        yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)

        h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
        h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

        reCalc   = df['r_e_calculated']
        rePred   = df['r_e_predec']
        for iter in range(len(reCalc)):
            h1a.Fill(reCalc[iter] , rePred[iter])

        c.hist2d(h1_backdrop, option='AXIS')
        c.hist2d(h1a,         option='COLZ')
        c.hist2d(h1_backdrop, option='AXIS')
        line = c.line(0, 0, 2, 2, linecolor=ROOT.kRed, linewidth=2)

        c.xlabel('Calculated response')
        c.ylabel('Predicted response')
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c.save("./plots/{}/Response2Dplot.png".format(args.rangeE))

    if Reponse1DPlot:
        corr, _ = pearsonr(df["r_e_calculated"], df["r_e_predec"])
        c = ap.canvas(num_pads=2, batch=True)
        p0, p1 = c.pads()
        xaxis = np.linspace(0, 2.5, 100 + 1, endpoint=True)

        truthResponse  = np.array(df["r_e_calculated"])
        predResponse   = np.array(df["r_e_predec"])
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            calibResponse  = np.array(df["clusterE"] / df["clusterECalib"])

        # -- Define histograms
        htruthResponse  = c.hist(truthResponse,  bins=xaxis, option='HIST', label='Calculated', linecolor=2)
        hpredResponse   = c.hist(predResponse,   bins=xaxis, option='HIST', label='Predicted',  linecolor=4)
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            hcalibResponse  = c.hist(calibResponse,  bins=xaxis, option='HIST', label='Calibrated',  linecolor=8)

        p1.ylim(0., 2.)
        c.ratio_plot((htruthResponse,  htruthResponse),  option='E2',      linecolor=2) #, oob=True)
        c.ratio_plot((hpredResponse,   htruthResponse),  option='HIST',    linecolor=4) #, oob=True)
        if args.rangeE == 'lowE' or args.rangeE == 'highE':
            c.ratio_plot((hcalibResponse,   htruthResponse),  option='HIST',    linecolor=8) #, oob=True)

        p1.yline(1.0)
        c.xlabel('Energy response')
        c.ylabel('Events')
        p1.ylabel('Variation / Truth')

        c.legend()
        c.log()
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) ), "Correlation: %.4f " % (corr)], qualifier='Simulation Internal')
        c.save("./plots/{}/Comparison_Truth_Pred_Response.png".format(args.rangeE))

    if RatioVsInputFeatures:

        features = ['clusterE', 'cluster_ENG_CALIB_TOT', 'clusterEtaCalib', 'cluster_CENTER_LAMBDA', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
                    'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
                    'nPrimVtx', 'avgMu']

        for idx, ifeature in enumerate(features):
            #if ifeature!='clusterE': continue
            c = ap.canvas(batch=True, size=(600,600))
            c.pads()[0]._bare().SetRightMargin(0.2)
            c.pads()[0]._bare().SetLogz()
            yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)
            if ifeature=='clusterE':               xaxis = np.linspace(-1,  3, 100 + 1, endpoint=True)
            if ifeature=='cluster_ENG_CALIB_TOT':  xaxis = np.linspace(-1,  3, 100 + 1, endpoint=True)
            if ifeature=='clusterEtaCalib':        xaxis = np.linspace(-1.5,1.5, 100 + 1, endpoint=True)
            if ifeature=='cluster_CENTER_LAMBDA':  xaxis = np.linspace(0,   3,  100 + 1, endpoint=True)
            if ifeature=='cluster_ENG_FRAC_EM':    xaxis = np.linspace(0,   1, 100 + 1, endpoint=True)
            if ifeature=='cluster_FIRST_ENG_DENS': xaxis = np.linspace(-9, -4, 100 + 1, endpoint=True)
            if ifeature=='cluster_LATERAL':        xaxis = np.linspace(0,   1, 100 + 1, endpoint=True)
            if ifeature=='cluster_LONGITUDINAL':   xaxis = np.linspace(0,   1, 100 + 1, endpoint=True)
            if ifeature=='cluster_PTD':            xaxis = np.linspace(0,   1, 100 + 1, endpoint=True)
            if ifeature=='cluster_SECOND_TIME':    xaxis = np.linspace(0, 175, 100 + 1, endpoint=True)
            if ifeature=='cluster_SIGNIFICANCE':   xaxis = np.linspace(0, 100, 100 + 1, endpoint=True)
            if ifeature=='nPrimVtx':               xaxis = np.linspace(0,  60, 100 + 1, endpoint=True)
            if ifeature=='avgMu':                  xaxis = np.linspace(10, 72, 100 + 1, endpoint=True)

            h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
            h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
            h1prof = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

            if ifeature=='clusterE' or ifeature=='cluster_ENG_CALIB_TOT' or ifeature=='cluster_CENTER_LAMBDA' or ifeature=='cluster_FIRST_ENG_DENS':
                BinLogX(h1_backdrop)
                BinLogX(h1a)
                BinLogX(h1prof)


            clusterE = df[ifeature]
            reCalc   = df["r_e_calculated"]
            rePred   = df["r_e_predec"]

            for iter in range(len(reCalc)):
                h1a.Fill(clusterE[iter], reCalc[iter] / rePred[iter] )


            for ibinx in range(1, h1a.GetNbinsX()+1):
                mode_binX = []
                for ibiny in range(1, h1a.GetNbinsY()+1):
                    n = int(h1a.GetBinContent(ibinx, ibiny))
                    for _ in range(n):
                        mode_binX.append(h1a.GetYaxis().GetBinCenter(ibiny))
                        pass
                if not mode_binX:
                    continue
                #print(mode(mode_binX)[0].flatten())
                h1prof.SetBinContent(ibinx, mode(mode_binX)[0].flatten())



            if ifeature=='clusterE' or ifeature=='cluster_ENG_CALIB_TOT' or ifeature=='cluster_CENTER_LAMBDA' or ifeature=='cluster_FIRST_ENG_DENS':
                c.logx()
            c.hist2d(h1_backdrop, option='AXIS')
            c.hist2d(h1a,         option='COLZ')
            c.hist(h1prof,        option='P', markercolor=ROOT.kViolet + 7)
            c.hist2d(h1_backdrop, option='AXIS')

            if ifeature=='clusterE':               xlabelname = 'Calculated cluster energy ' + r'E^{dep}_{clus}' + ' [GeV]'
            if ifeature=='cluster_ENG_CALIB_TOT':  xlabelname = 'Truth cluster energy ' + r'E^{dep}_{clus}' + ' [GeV]'
            if ifeature=='clusterEtaCalib':        xlabelname = r'\mathcal{y}_{clus}'
            if ifeature=='cluster_CENTER_LAMBDA':  xlabelname = r'\lambda_{clus}'
            if ifeature=='cluster_ENG_FRAC_EM':    xlabelname = r'f_{emc}'
            if ifeature=='cluster_FIRST_ENG_DENS': xlabelname = r'\rho_{clus}'
            if ifeature=='cluster_LATERAL':        xlabelname = r'\mathcal{m}^{2}_{lat}'
            if ifeature=='cluster_LONGITUDINAL':   xlabelname = r'\mathcal{m}^{2}_{long}'
            if ifeature=='cluster_PTD':            xlabelname = r'p_{T}D'
            if ifeature=='cluster_SECOND_TIME':    xlabelname = r'\sigma^{2}_{t}'
            if ifeature=='cluster_SIGNIFICANCE':   xlabelname = r'\zeta^{EM}_{clus}'
            if ifeature=='nPrimVtx':               xlabelname = r'N_{PV}'
            if ifeature=='avgMu':                  xlabelname = r'\bigl\langle\mu\bigr\rangle'

            ylablename = r'\mathcal{R}^{pred} / \mathcal{R}^{calc}'
            c.xlabel(xlabelname)
            c.ylabel(ylablename)
            c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
            c.save("./plots/{}/Final_{}.png".format(args.rangeE, ifeature))

    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
