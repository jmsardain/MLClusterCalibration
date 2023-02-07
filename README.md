# MLClusterCalibration

Olivia's version: 

```
source setup.sh
python training.py ## --nentries NENTRIES to sample the ROOT file
```
The training code prompts the user to set some of the hyperparamters (activation function, cut on truth clusterE, etc.)


It produces the plotting dataframe in the FinalPlots repository. To run the plotting code: 
```
cd FinalPlots/
source setup.sh 
python Plotting.py
```


Old version: 
```
source setup.sh
python convertRootToCsv.py # add --nentries 1000000 for fewer inputs and add --plot if you also want before and after plots
python traintest.py --train --rangeE all 
python traintest.py --test --path PathToModelDirectory --rangeE all 
python traintest.py --plot --rangeE all 

cd FinalPlots
python -b Plotting.py

```
