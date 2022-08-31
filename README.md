# MLClusterCalibration

```
source setup.sh
python convertRootToCsv.py # add --plot if you also want before and after plots
python traintest.py --train --rangeE all 
python traintest.py --test --path PathToModelDirectory --rangeE all 
python traintest.py --plot --rangeE all 

cd FinalPlots
python -b Plotting.py

```
