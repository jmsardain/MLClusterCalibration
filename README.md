# MLClusterCalibration

```
source setup.sh
python convertRootToCsv.py # add --plot if you also want before and after plots
python traintest.py --train
python traintest.py --test --path PathToModelDirectory
python traintest.py --plot

cd FinalPlots
python -b final.py

```
