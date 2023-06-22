# MLClusterCalibration

## Running the code

If you are running on the Chicago server, you do not really need to do much (all dependencies are already installed). If you are not running on the Chicago servers, you can find all the packages needed in the file called environment.yml


The first code you need reads the ROOT file that contains the clusters info, transforms the input features and split the dataset into train, val, test.
The whole dataset is also stored, unchanged, in order to be able to plot the prediction at the end w.r.t. variables we do not train on.
```
python read_csv_and_make_plots.py
```


For the ML part, there is one code, depending on the argument, trains, retrains, tests.

```
python train.py --train   --outdir out
python train.py --retrain --outdir out
python train.py --test    --outdir out
```


## Running the plotting script

The output of the training/testing code is a bunch of numpy arrays. You can either create your own code to plot, or rely on this.

In a clean terminal
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_88/x86_64-slc6-gcc49-opt/setup.sh
export PYTHONPATH=$PWD:$PYTHONPATH
python plotting.py
```
