import ROOT
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mode, iqr
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

def main():
    ROOT.gStyle.SetPalette(ROOT.kBird)

    if args.rangeE == 'all':
        df = pd.read_csv("/home/opitcl/calo-jad/MLClusterCalibration/FinalPlots/plot_all.csv", sep=' ')
        # df1 = pd.read_csv("/home/opitcl/calo-jad/MLClusterCalibration/FinalPlots/plot_lowE.csv", sep=' ')
        # df2 = pd.read_csv("/home/opitcl/calo-jad/MLClusterCalibration/FinalPlots/plot_midE.csv", sep=' ')
        # df3 = pd.read_csv("/home/opitcl/calo-jad/MLClusterCalibration/FinalPlots/plot_highE.csv", sep=' ')
        # df = pd.concat([df1, df2, df3], ignore_index=True)
    else:
        df = pd.read_csv("/home/opitcl/calo-jad/MLClusterCalibration/FinalPlots/plot_{}.csv".format(args.rangeE), sep=' ')

    #print(df.columns)

    if args.rangeE == 'lowE':
        rangeEnergy = '0.3 #leq trueE < 1 GeV'
    elif args.rangeE =='midE':
        rangeEnergy = '1 #leq trueE < 5 GeV'
    elif args.rangeE == 'highE':
        rangeEnergy = 'trueE #geq 5 GeV'
    elif args.rangeE == 'all':
        rangeEnergy = 'trueE #geq 0.3 GeV'
    else:
        print("Precise range in code and rerun")
        return

    EnergyNoLogE         = True
    EnergyLogE           = True
    Energy2DPlot         = True
    ClusE2DPlotE         = False
    ClusE2DPlotR         = True
    Reponse2DPlot        = True
    Reponse1DPlot        = True
    RatioVsInputFeatures = False
    MedianIQR            = True ## Peter email Aug 23
    Linearity            = True ## Peter email Aug 23, includes median, ave, and rms

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
        calibClusTotalE = np.array(df["clusterECalib_new"])

        # -- Define histograms
        htruthClusTotalE = c.hist(truthClusTotalE, bins=xaxis, option='HIST', label='Truth total E',      linecolor=2)
        hpredClusTotalE  = c.hist(predClusTotalE,  bins=xaxis, option='HIST', label='Predicted total E',  linecolor=4)
        hcalibClusTotalE = c.hist(calibClusTotalE, bins=xaxis, option='HIST', label='Calibrated total E', linecolor=8)

        p1.ylim(0., 2.)
        c.ratio_plot((htruthClusTotalE,  htruthClusTotalE),  option='E2',      linecolor=2) #, oob=True)
        c.ratio_plot((hpredClusTotalE,   htruthClusTotalE),  option='HIST',    linecolor=4) #, oob=True)
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
	BinLogX(hcalibClusTotalE)
	BinLogX(htruthpred)
	BinLogX(htruthcalib)

	# -- Now let's compare truthClusTotalE and predClusTotalE
	# -- Get the arrays
	truthClusTotalE = np.array(df["cluster_ENG_CALIB_TOT"])
	predClusTotalE  = np.array(df["cluster_ENG_TOT_frompred"])
	calibClusTotalE = np.array(df["clusterECalib_new"])
	
	print(len(truthClusTotalE) == len(predClusTotalE))
	print("Maximums (pred, true):", np.max(predClusTotalE), np.max(truthClusTotalE))
	print("Minimums (pred, true):", np.min(predClusTotalE), np.min(truthClusTotalE))
	

	for iter in range(len(truthClusTotalE)):
	    htruthClusTotalE.Fill(truthClusTotalE[iter])
	    hpredClusTotalE.Fill(predClusTotalE[iter])
	    hcalibClusTotalE.Fill(calibClusTotalE[iter])
	    htruthpred.Fill(truthClusTotalE[iter]  / predClusTotalE[iter])
	    htruthcalib.Fill(truthClusTotalE[iter] / calibClusTotalE[iter])
	# -- Define histograms
	c.hist(h1_backdrop, option='AXIS')
	c.hist(htruthClusTotalE,  option='HIST', label='Truth E',      linecolor=2)
	c.hist(hpredClusTotalE,   option='HIST',  label='Predicted E',   linecolor=4)
	c.hist(hcalibClusTotalE,  option='HIST', label='LCW Calibrated E', linecolor=8)
	c.hist(h1_backdrop, option='AXIS')



	p1.ylim(0., 2.)
	p1.logx()
	c.ratio_plot((htruthClusTotalE,  htruthClusTotalE),  option='E2',      linecolor=2) #, oob=True)
	c.ratio_plot((hpredClusTotalE,   htruthClusTotalE),  option='HIST',    linecolor=4) #, oob=True)
	c.ratio_plot((hcalibClusTotalE,  htruthClusTotalE),  option='HIST',    linecolor=8 )#, oob=True)

	c.xlabel('Cluster energy [GeV]')
	c.ylabel('Events')
	p1.ylabel('Variation / Truth')

	c.legend()
	c.log()
	c.logx()
	c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) ), "Correlation: %.4f " % (corr)], qualifier='Simulation Internal')
	c.save("./plots/{}/Comparison_Truth_Pred_Log.png".format(args.rangeE))
 
    if Energy2DPlot:
        c = ap.canvas(batch=True, size=(600,600))
        c1 = ap.canvas(batch=True, size=(600,600))
        c2 = ap.canvas(batch=True, size=(600,600))
        c3 = ap.canvas(batch=True, size=(600,600))

        c.pads()[0]._bare().SetRightMargin(0.2)
        c.pads()[0]._bare().SetLogz()
        c1.pads()[0]._bare().SetRightMargin(0.2)
        c1.pads()[0]._bare().SetLogz()
        c2.pads()[0]._bare().SetRightMargin(0.2)
        c2.pads()[0]._bare().SetLogz()
        c3.pads()[0]._bare().SetRightMargin(0.2)
        c3.pads()[0]._bare().SetLogz()

        xaxis = np.linspace(-1, 3,  100 + 1, endpoint=True)
        yaxis = np.linspace(-1, 3,  100 + 1, endpoint=True)

        h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], yaxis[-1] + 0.55 * (yaxis[-1] - yaxis[0])]))
        h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1b = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1c = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1d = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

        BinLogX(h1_backdrop)
        BinLogY(h1_backdrop)
        BinLogX(h1a)
        BinLogY(h1a)
        BinLogX(h1b)
        BinLogY(h1b)
        BinLogX(h1c)
        BinLogY(h1c)
        BinLogX(h1d)
        BinLogY(h1d)

        # feed = df[(df["cluster_ENG_TOT_frompred"] > 0)]
        truthClusTotalE = np.array(df["cluster_ENG_CALIB_TOT"])
        predClusTotalE  = np.array(df["cluster_ENG_TOT_frompred"])
        calibClusTotalE = np.array(df["clusterECalib_new"])
        clusterE 	= np.array(df["clusterE"])
        calibClus_old	= np.array(df["clusterECalib_old"])
	
        for iter in range(len(truthClusTotalE)):
            # Truth on y-axis
            # h1a.Fill(predClusTotalE[iter], truthClusTotalE[iter])
            # h1b.Fill(calibClusTotalE[iter], truthClusTotalE[iter])
            # h1c.Fill(clusterE[iter], truthClusTotalE[iter])
            
            # Truth on x-axis
            h1a.Fill(truthClusTotalE[iter], predClusTotalE[iter])
            h1b.Fill(truthClusTotalE[iter], calibClusTotalE[iter])
            h1c.Fill(truthClusTotalE[iter], clusterE[iter])
            h1d.Fill(calibClus_old[iter],   calibClusTotalE[iter])
	
        c.hist2d(h1_backdrop, option='AXIS')
        c.hist2d(h1a,         option='COLZ')
        c.hist2d(h1_backdrop, option='AXIS')
        line = c.line(1e-1, 1e-1, 1e3, 1e3, linecolor=ROOT.kRed, linewidth=2)
	c.ylim(np.min(truthClusTotalE), np.max(truthClusTotalE))
        c.logx()
	# c.logy()
        c.log()

        c.xlabel('Truth cluster energy [GeV]')
        c.ylabel('Predicted cluster energy [GeV]')
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c.save("./plots/{}/Energy2Dplot_Pred_Truth.png".format(args.rangeE))



        c1.hist2d(h1_backdrop, option='AXIS')
        c1.hist2d(h1b,         option='COLZ')
        c1.hist2d(h1_backdrop, option='AXIS')
        line = c1.line(1e-1, 1e-1, 1e3, 1e3, linecolor=ROOT.kRed, linewidth=2)
	c1.ylim(np.min(truthClusTotalE), np.max(truthClusTotalE))
        c1.logx()
	# c1.logy()
        c1.log()
        c1.xlabel('Truth cluster energy [GeV]')
        c1.ylabel('LCW Calibrated cluster energy [GeV]')
        c1.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c1.save("./plots/{}/Energy2Dplot_Calib_Truth.png".format(args.rangeE))
	
        c2.hist2d(h1_backdrop, option='AXIS')
        c2.hist2d(h1c,         option='COLZ')
        c2.hist2d(h1_backdrop, option='AXIS')
        line = c2.line(1e-1, 1e-1, 1e3, 1e3, linecolor=ROOT.kRed, linewidth=2)
        c2.logx()
	c2.ylim(np.min(truthClusTotalE), np.max(truthClusTotalE))
	# c2.logy()
        c2.log()
        c2.xlabel('Truth cluster energy [GeV]')
        c2.ylabel('ClusterE - EM scale [GeV]')
        c2.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c2.save("./plots/{}/Energy2Dplot_ClusterE_Truth.png".format(args.rangeE))


        c3.hist2d(h1_backdrop, option='AXIS')
        c3.hist2d(h1d,         option='COLZ')
        c3.hist2d(h1_backdrop, option='AXIS')
        line = c3.line(1e-1, 1e-1, 1e3, 1e3, linecolor=ROOT.kRed, linewidth=2)
        c3.logx()
	# c2.logy()
        c3.log()
        c3.xlabel('Old calibrated cluster E [GeV]')
        c3.ylabel('LCW Calibrated cluster E [GeV]')
        c3.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c3.save("./plots/{}/Energy2Dplot_ClusterE_oldNew.png".format(args.rangeE))

    if ClusE2DPlotE:
        c = ap.canvas(batch=True, size=(600,600))
        c.pads()[0]._bare().SetRightMargin(0.2)
        c.pads()[0]._bare().SetLogz()
        xaxis = np.linspace(0, 5,  100 + 1, endpoint=True)
        yaxis = np.linspace(0, 15,  100 + 1, endpoint=True)

        h1_backdrop = ROOT.TH2F('', '', 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], yaxis[-1] + 0.55 * (yaxis[-1] - yaxis[0])]))
        h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

        # BinLogX(h1_backdrop)
        # BinLogX(h1a)
        # BinLogY(h1_backdrop)
        # BinLogY(h1a)


        ClustE          = np.array(df["clusterE"])
        truthClusTotalE = np.array(df["cluster_ENG_CALIB_TOT"])

        for iter in range(len(ClustE)):
             h1a.Fill(ClustE[iter], truthClusTotalE[iter])
             #print("{} {}".format(ClustE[iter], truthClusTotalE[iter]))

        #c.logx()
        c.log()
        c.hist2d(h1_backdrop, option='AXIS')
        c.hist2d(h1a,         option='COLZ')
        c.hist2d(h1_backdrop, option='AXIS')

        c.xlabel('Cluster E [GeV]')
        c.ylabel('Truth cluster energy [GeV]')
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c.save('./plots/{}/ClusterE_truth.png'.format(args.rangeE))

    if ClusE2DPlotR:
        c = ap.canvas(batch=True, size=(600,600))
        c.pads()[0]._bare().SetRightMargin(0.2)
        c.pads()[0]._bare().SetLogz()
        xaxis = np.linspace(0, 10,  100 + 1, endpoint=True)
        yaxis = np.linspace(0,  2, 100 + 1, endpoint=True)

        h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], yaxis[-1] + 0.55 * (yaxis[-1] - yaxis[0])]))
        h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

        # BinLogX(h1_backdrop)
        # BinLogX(h1a)

        ClustE          = np.array(df["clusterE"])
        RespCalc        = np.array(df["r_e_calculated"])

        for iter in range(len(ClustE)):
             h1a.Fill(ClustE[iter], RespCalc[iter] )


        #c.logx()
        c.hist2d(h1_backdrop, option='AXIS')
        c.hist2d(h1a,         option='COLZ')
        c.hist2d(h1_backdrop, option='AXIS')

        c.xlabel('Cluster E [GeV]')
        c.ylabel('Response')
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c.save("./plots/{}/ClusterE_response.png".format(args.rangeE))


    if Reponse2DPlot:
        c = ap.canvas(batch=True, size=(600,600))
        c.pads()[0]._bare().SetRightMargin(0.2)
        c.pads()[0]._bare().SetLogz()
        c1 = ap.canvas(batch=True, size=(600,600))
        c1.pads()[0]._bare().SetRightMargin(0.2)
        c1.pads()[0]._bare().SetLogz()
        xaxis = np.linspace(0, 2,  100 + 1, endpoint=True)
        yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)

        h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
        h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1b = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

        reCalc   = df['r_e_calculated']
        rePred   = df['r_e_predec']
        for iter in range(len(reCalc)):
            h1a.Fill(reCalc[iter] , rePred[iter])
            h1b.Fill(reCalc[iter] , rePred[iter] / reCalc[iter])

        c.hist2d(h1_backdrop, option='AXIS')
        c.hist2d(h1a,         option='COLZ')
        c.hist2d(h1_backdrop, option='AXIS')
        line = c.line(0, 0, 2, 2, linecolor=ROOT.kRed, linewidth=2)

        c1.hist2d(h1_backdrop, option='AXIS')
        c1.hist2d(h1b,         option='COLZ')
        c1.hist2d(h1_backdrop, option='AXIS')

        c.xlabel('EM scale response')
        c.ylabel('Predicted response')
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c.save("./plots/{}/Response2Dplot.png".format(args.rangeE))

        c1.xlabel('EM scale response')
        c1.ylabel('Predicted response / EM scale response')
        c1.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c1.save("./plots/{}/Response2Dplot_Ratio.png".format(args.rangeE))

    if Reponse1DPlot:
        corr, _ = pearsonr(df["r_e_calculated"], df["r_e_predec"])
        c = ap.canvas(num_pads=2, batch=True)
        p0, p1 = c.pads()
        xaxis = np.linspace(0, 2.5, 100 + 1, endpoint=True)

        truthResponse  = np.array(df["r_e_calculated"])
        predResponse   = np.array(df["r_e_predec"])
        calibResponse  = np.array(df["clusterECalib_new"] / df["cluster_ENG_CALIB_TOT"])

        # -- Define histograms
        htruthResponse  = c.hist(truthResponse,  bins=xaxis, option='HIST', label='EM scale', linecolor=2)
        hpredResponse   = c.hist(predResponse,   bins=xaxis, option='HIST', label='Predicted',  linecolor=4)
        hcalibResponse  = c.hist(calibResponse,  bins=xaxis, option='HIST', label='Calibrated',  linecolor=8)

        p1.ylim(0., 2.)
        c.ratio_plot((htruthResponse,  htruthResponse),  option='E2',      linecolor=2) #, oob=True)
        c.ratio_plot((hpredResponse,   htruthResponse),  option='HIST',    linecolor=4) #, oob=True)
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

        features = ['clusterE', 'cluster_ENG_CALIB_TOT', 'cluster_CENTER_LAMBDA', 'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS',
                     'cluster_LATERAL', 'cluster_LONGITUDINAL', 'cluster_PTD', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
                     'nPrimVtx', 'avgMu']
        #features = ['clusterE', 'cluster_ENG_CALIB_TOT']
        # features = ['clusterPt',  'nPrimVtx']

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
            if ifeature=='clusterPt':              xaxis = np.linspace(-1,  3, 100 + 1, endpoint=True)

            h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
            h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
            h1prof = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

            if ifeature=='clusterE' or ifeature=='clusterPt' or ifeature=='cluster_ENG_CALIB_TOT' or ifeature=='cluster_CENTER_LAMBDA' or ifeature=='cluster_FIRST_ENG_DENS':
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
                h1prof.SetBinError(ibinx, 0)



            if ifeature=='clusterE' or ifeature=='clusterPt' or ifeature=='cluster_ENG_CALIB_TOT' or ifeature=='cluster_CENTER_LAMBDA' or ifeature=='cluster_FIRST_ENG_DENS':
                c.logx()
            c.hist2d(h1_backdrop, option='AXIS')
            c.hist2d(h1a,         option='COLZ')
            c.hist(h1prof,        option='P', markercolor=ROOT.kRed)
            c.hist2d(h1_backdrop, option='AXIS')

            if ifeature=='clusterE':               xlabelname = 'EM scale cluster energy ' + r'E^{dep}_{clus}' + ' [GeV]'
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
            if ifeature=='clusterPt':              xlabelname = r'p_{T, clus}'

            ylablename = r'\mathcal{R}^{pred} / \mathcal{R}^{calc}'
            c.xlabel(xlabelname)
            c.ylabel(ylablename)
            c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
            c.save("./plots/{}/Final_{}.png".format(args.rangeE, ifeature))

    if MedianIQR:

        reCalc     = df["r_e_calculated"]
        rePred     = df["r_e_predec"]
        reCalib     = df["clusterECalib_new"] / df["cluster_ENG_CALIB_TOT"]
        trueEnergy = df["cluster_ENG_CALIB_TOT"]

        c = ap.canvas(batch=True, size=(600,600))
        c.pads()[0]._bare().SetRightMargin(0.2)
        c.pads()[0]._bare().SetLogz()

        c1 = ap.canvas(batch=True, size=(600,600))
        c1.pads()[0]._bare().SetRightMargin(0.2)
        c1.pads()[0]._bare().SetLogz()
        xaxis = np.linspace(-1,  3, 100 + 1, endpoint=True)
        yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)

        h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1b = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1c = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

        h1CalcResponse = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        h1PredResponse = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        h1CalibResponse = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

        BinLogX(h1a)
        BinLogX(h1b)
        BinLogX(h1c)
        BinLogX(h1CalcResponse)
        BinLogX(h1PredResponse)
        BinLogX(h1CalibResponse)

        h2a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h2b = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h2c = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

        h1CalcResolution = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        h1PredResolution = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        h1CalibResolution = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

        BinLogX(h2a)
        BinLogX(h2b)
        BinLogX(h2c)
        BinLogX(h1CalcResolution)
        BinLogX(h1PredResolution)
        BinLogX(h1CalibResolution)




        for iter in range(len(reCalc)):
            h1a.Fill(trueEnergy[iter], reCalc[iter])
            h1b.Fill(trueEnergy[iter], rePred[iter])
            h1c.Fill(trueEnergy[iter], reCalib[iter])
            h2a.Fill(trueEnergy[iter], reCalc[iter])
            h2b.Fill(trueEnergy[iter], rePred[iter])
            h2c.Fill(trueEnergy[iter], reCalib[iter])


        for ibinx in range(1, h1a.GetNbinsX()+1):
            median_binX = []
            for ibiny in range(1, h1a.GetNbinsY()+1):
                n = int(h1a.GetBinContent(ibinx, ibiny))
                for _ in range(n):
                    median_binX.append(h1a.GetYaxis().GetBinCenter(ibiny))
                    pass
            if not median_binX:
                continue
            #print(mode(mode_binX)[0].flatten())
            calcMedian =  np.median(median_binX)
            calcIQR =  iqr(median_binX, rng=(16, 84)) / (2 * np.median(median_binX))
            h1CalcResponse.SetBinContent(ibinx, calcMedian)
            h1CalcResolution.SetBinContent(ibinx, calcIQR)
            h1CalcResponse.SetBinError(ibinx, 0)
            h1CalcResolution.SetBinError(ibinx, 0)


        for ibinx in range(1, h1b.GetNbinsX()+1):
            median_binX = []
            for ibiny in range(1, h1b.GetNbinsY()+1):
                n = int(h1b.GetBinContent(ibinx, ibiny))
                for _ in range(n):
                    median_binX.append(h1b.GetYaxis().GetBinCenter(ibiny))
                    pass
            if not median_binX:
                continue
            predMedian =  np.median(median_binX)
            predIQR =  iqr(median_binX, rng=(16, 84)) / (2 * np.median(median_binX))
            h1PredResponse.SetBinContent(ibinx, predMedian)
            h1PredResolution.SetBinContent(ibinx, predIQR)
            h1PredResponse.SetBinError(ibinx, 0)
            h1PredResolution.SetBinError(ibinx, 0)


        for ibinx in range(1, h1c.GetNbinsX()+1):
            median_binX = []
            for ibiny in range(1, h1c.GetNbinsY()+1):
                n = int(h1c.GetBinContent(ibinx, ibiny))
                for _ in range(n):
                    median_binX.append(h1c.GetYaxis().GetBinCenter(ibiny))
                    pass
            if not median_binX:
                continue
            calibMedian =  np.median(median_binX)
            calibIQR =  iqr(median_binX, rng=(16, 84)) / (2 * np.median(median_binX))
            h1CalibResponse.SetBinContent(ibinx, calibMedian)
            h1CalibResolution.SetBinContent(ibinx, calibIQR)
            h1CalibResponse.SetBinError(ibinx, 0)
            h1CalibResolution.SetBinError(ibinx, 0)



        c.hist(h1CalcResponse, markercolor=ROOT.kViolet + 7, linecolor=ROOT.kViolet + 7, label="EM scale")
        c.hist(h1PredResponse, markercolor=ROOT.kAzure + 7, linecolor=ROOT.kAzure + 7, label="Predicted")
        c.hist(h1CalibResponse, markercolor=ROOT.kOrange-3, linecolor=ROOT.kOrange-3, label="Calibrated")
        c.logx()
        c.legend()
        c.xlabel("Truth Particle Energy [GeV]")
        c.ylabel("Response Median")
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c.save("./plots/{}/Median_Reponse.png".format(args.rangeE))

        c1.hist(h1PredResolution, markercolor=ROOT.kViolet + 7, linecolor=ROOT.kViolet + 7, label="Predicted")
        c1.hist(h1CalibResolution, markercolor=ROOT.kAzure + 7, linecolor=ROOT.kAzure + 7, label="Calibrated")
        c1.hist(h1CalcResolution, markercolor=ROOT.kOrange - 3, linecolor=ROOT.kOrange - 3, label="EM scale")
        c1.logx()
        c1.legend()
        c1.xlabel("Truth Particle Energy [GeV]")
        c1.ylabel("Response IQR / (2 #times Median)")
        c1.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c1.save("./plots/{}/IQR_Reponse.png".format(args.rangeE))


    if Linearity:

        colors = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink]

        c = ap.canvas(batch=True, size=(600,600))
        c.pads()[0]._bare().SetRightMargin(0.2)
        c.pads()[0]._bare().SetLogz()
        xaxis = np.linspace(-1,  3, 100 + 1, endpoint=True)
        yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)

        h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1b = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        PredOverTrueMedian = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        LCWOverTrueMedian  = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

        PredOverTrueAvg    = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        LCWOverTrueAvg     = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

        BinLogX(h1a)
        BinLogX(h1b)
        BinLogX(PredOverTrueMedian)
        BinLogX(LCWOverTrueMedian)
        BinLogX(PredOverTrueAvg)
        BinLogX(LCWOverTrueAvg)

        LCWEnergy  = df["clusterECalib_new"]
        PredEnergy = df["cluster_ENG_TOT_frompred"]
        TrueEnergy = df["cluster_ENG_CALIB_TOT"]

        for iter in range(len(TrueEnergy)):
            h1a.Fill(TrueEnergy[iter], PredEnergy[iter] / TrueEnergy[iter])
            h1b.Fill(TrueEnergy[iter], LCWEnergy[iter] / TrueEnergy[iter])


        for ibinx in range(1, h1a.GetNbinsX()+1):
            median_binX = []
            for ibiny in range(1, h1a.GetNbinsY()+1):
                n = int(h1a.GetBinContent(ibinx, ibiny))
                for _ in range(n):
                    median_binX.append(h1a.GetYaxis().GetBinCenter(ibiny))
                    pass
            if not median_binX:
                continue
            #print(mode(mode_binX)[0].flatten())
            predMedian =  np.median(median_binX)
            predAvg    =  np.mean(median_binX)

            PredOverTrueMedian.SetBinContent(ibinx, predMedian)
            PredOverTrueAvg.SetBinContent(ibinx, predAvg)
            PredOverTrueMedian.SetBinError(ibinx, 0)
            PredOverTrueAvg.SetBinError(ibinx, 0)


        for ibinx in range(1, h1b.GetNbinsX()+1):
            median_binX = []
            for ibiny in range(1, h1b.GetNbinsY()+1):
                n = int(h1b.GetBinContent(ibinx, ibiny))
                for _ in range(n):
                    median_binX.append(h1b.GetYaxis().GetBinCenter(ibiny))
                    pass
            if not median_binX:
                continue
            calcMedian =  np.median(median_binX)
            calcAvg    =  np.mean(median_binX)
            LCWOverTrueMedian.SetBinContent(ibinx, calcMedian)
            LCWOverTrueAvg.SetBinContent(ibinx, calcAvg)
            LCWOverTrueMedian.SetBinError(ibinx, 0)
            LCWOverTrueAvg.SetBinError(ibinx, 0)

        c.hist(PredOverTrueMedian, markercolor=colors[0], linecolor=colors[0], label="Median E^{pred} / E^{true}")
        c.hist(PredOverTrueAvg, markercolor=colors[1], linecolor=colors[1], label="Avg. E^{pred} / E^{true}")
        c.hist(LCWOverTrueMedian, markercolor=colors[2], linecolor=colors[2], label="Median E^{LCW} / E^{true}")
        c.hist(LCWOverTrueAvg, markercolor=colors[3], linecolor=colors[3], label="Avg. E^{LCW} / E^{true}")
        c.logx()
        c.legend()
        c.xlabel("Truth Particle Energy [GeV]")
        #c.ylabel("Response Median")
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c.save("./plots/{}/Linearity.png".format(args.rangeE))
    
    # Plot standard deviations of E_calib and E_predicted
    if Linearity:

        colors = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink]

        c = ap.canvas(batch=True, size=(600,600))
        c.pads()[0]._bare().SetRightMargin(0.2)
        c.pads()[0]._bare().SetLogz()
        xaxis = np.linspace(-1,  3, 100 + 1, endpoint=True)
        yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)

        h1a = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1b = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        h1c = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
        PredOverTrueRMS    = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        LCWOverTrueRMS  = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
        CalcOverTrueRMS  = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

        BinLogX(h1a)
        BinLogX(h1b)
        BinLogX(h1c)
        BinLogX(PredOverTrueRMS)
        BinLogX(LCWOverTrueRMS)
        BinLogX(CalcOverTrueRMS)
        # BinLogX(PredOverTrueAvg)
        # BinLogX(LCWOverTrueAvg)

        LCWEnergy  = df["clusterECalib_new"]
        PredEnergy = df["cluster_ENG_TOT_frompred"]
        clusterEnergy = df["clusterE"]
        TrueEnergy = df["cluster_ENG_CALIB_TOT"]

        for iter in range(len(TrueEnergy)):
            h1a.Fill(TrueEnergy[iter], PredEnergy[iter] / TrueEnergy[iter])
            h1b.Fill(TrueEnergy[iter], LCWEnergy[iter] / TrueEnergy[iter])
            h1c.Fill(TrueEnergy[iter], clusterEnergy[iter] / TrueEnergy[iter])


        for ibinx in range(1, h1a.GetNbinsX()+1):
            rms_binX = []
            for ibiny in range(1, h1a.GetNbinsY()+1):
                n = int(h1a.GetBinContent(ibinx, ibiny))
                for _ in range(n):
                    rms_binX.append(h1a.GetYaxis().GetBinCenter(ibiny))
                    pass
            if not rms_binX:
                continue
            #print(mode(mode_binX)[0].flatten())
            rms_binX = np.array(rms_binX)
            predRMS  = np.std(rms_binX)
            predAvg  = np.mean(median_binX)

            PredOverTrueRMS.SetBinContent(ibinx, predRMS)
            # PredOverTrueAvg.SetBinContent(ibinx, predAvg)
            PredOverTrueRMS.SetBinError(ibinx, 0)
            # PredOverTrueAvg.SetBinError(ibinx, 0)


        for ibinx in range(1, h1b.GetNbinsX()+1):
            rms_binX = []
            for ibiny in range(1, h1b.GetNbinsY()+1):
                n = int(h1b.GetBinContent(ibinx, ibiny))
                for _ in range(n):
                    rms_binX.append(h1b.GetYaxis().GetBinCenter(ibiny))
                    pass
            if not rms_binX:
                continue
            rms_binX = np.array(rms_binX)
            calibRMS  = np.std(rms_binX)
            calibAvg  = np.mean(median_binX)
            LCWOverTrueRMS.SetBinContent(ibinx, calibRMS)
            # LCWOverTrueAvg.SetBinContent(ibinx, calcAvg)
            LCWOverTrueRMS.SetBinError(ibinx, 0)
            # LCWOverTrueAvg.SetBinError(ibinx, 0)

        for ibinx in range(1, h1c.GetNbinsX()+1):
            rms_binX = []
            for ibiny in range(1, h1c.GetNbinsY()+1):
                n = int(h1c.GetBinContent(ibinx, ibiny))
                for _ in range(n):
                    rms_binX.append(h1c.GetYaxis().GetBinCenter(ibiny))
                    pass
            if not rms_binX:
                continue
            rms_binX = np.array(rms_binX)
            calcRMS  = np.std(rms_binX)
            calcAvg  = np.mean(median_binX)
            CalcOverTrueRMS.SetBinContent(ibinx, calcRMS)
            # LCWOverTrueAvg.SetBinContent(ibinx, calcAvg)
            CalcOverTrueRMS.SetBinError(ibinx, 0)
            # LCWOverTrueAvg.SetBinError(ibinx, 0)

        c.hist(PredOverTrueRMS, markercolor=colors[0], linecolor=colors[0], label="RMS E^{pred} / E^{true}")
        # c.hist(PredOverTrueAvg, markercolor=colors[1], linecolor=colors[1], label="Avg. E^{pred} / E^{true}")
        c.hist(LCWOverTrueRMS, markercolor=colors[1], linecolor=colors[1], label="RMS E^{LCW} / E^{true}")
        # c.hist(LCWOverTrueAvg, markercolor=colors[3], linecolor=colors[3], label="Avg. E^{LCW} / E^{true}")
        c.hist(CalcOverTrueRMS, markercolor=colors[4], linecolor=colors[4], label="RMS E^{EM scale} / E^{true}")
        c.logx()
        c.legend()
        c.xlabel("Truth Particle Energy [GeV]")
        c.ylabel("Response Ratio STD")
        #c.ylabel("Response Median")
        c.text(["#sqrt{s} = 13 TeV", ("%s" % (rangeEnergy) )], qualifier='Simulation Internal')
        c.save("./plots/{}/Linearity-STD.png".format(args.rangeE))
    

    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
