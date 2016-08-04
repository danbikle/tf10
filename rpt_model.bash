#!/bin/bash

# rpt_model.bash

# This script should report accuracy and effectiveness for a model.
# Demo:
# . ~/tf10/envtf.bash
# ${TF}/rpt_model.bash

cd ${TFTMP}/csv/

models='tf11 tf12 sk_lr'

for MODEL in $models
do
  echo ooooooooooooooooooooooooooooooooooooooo
  head -1   predictions_${MODEL}_2016.csv            > all_predictions_${MODEL}.csv
  cat predictions_${MODEL}_????.csv | grep -v cdate >> all_predictions_${MODEL}.csv
  echo prediction count:
  wc -l all_predictions_${MODEL}.csv
  python ${TF}/rpt_model.py $MODEL
  echo ooooooooooooooooooooooooooooooooooooooo
done

exit
