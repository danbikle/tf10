#!/bin/bash

# rpt_model.bash

# This script should report accuracy and effectiveness for a model.
# Demo:
# . ~/tf10/envtf.bash
# ${TF}/rpt_model.bash

cd ${TFTMP}/csv/

MODEL=tf11
echo prediction count:
wc -l all_predictions_${MODEL}.csv
head -1   predictions_${MODEL}_2016.csv            > all_predictions_${MODEL}.csv
cat predictions_${MODEL}_????.csv | grep -v cdate >> all_predictions_${MODEL}.csv
python ${TF}/rpt_model.py $MODEL
echo ooooooooooooooooooooooooooooooooooooooo
MODEL=sk_lr
echo prediction count:
wc -l all_predictions_${MODEL}.csv
head -1   predictions_${MODEL}_2016.csv            > all_predictions_${MODEL}.csv
cat predictions_${MODEL}_????.csv | grep -v cdate >> all_predictions_${MODEL}.csv
python ${TF}/rpt_model.py $MODEL
echo ooooooooooooooooooooooooooooooooooooooo
exit
