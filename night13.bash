#!/bin/bash

# night13.bash

# This script should calculate predictions of the S&P500.

# I should assume that I am in the same folder as envtf.bash:
. envtf.bash

# I should get prices
${TF}/wgetGSPCnight.bash

# I should generate features from prices:
cd ${TFTMP}/csv
${HOME}/anaconda3/bin/python ${TF}/genf.py GSPC2.csv
${HOME}/anaconda3/bin/python ${TF}/genofd.py GSPC2.csv

# Next I generate training data CSV files:

STARTYR=1981
ENDYR=2016
TRAINING_AMOUNT=30 #years
${HOME}/anaconda3/bin/python ${TF}/gentrain_test.py ftrGSPC2.csv $TRAINING_AMOUNT $STARTYR $ENDYR
${HOME}/anaconda3/bin/python ${TF}/gentrain_test.py ftr_ofdGSPC2.csv $TRAINING_AMOUNT $STARTYR $ENDYR

# Then, I should train models.
# With the models, predict the test data.
# And, collect Accuracy and Effectiveness:

models=sk_lr_1hf

for MODEL in $models
do
  ${HOME}/anaconda3/bin/python ${TF}/train_test_${MODEL}.py $STARTYR $ENDYR
done

# I should report Accuracy and Effectiveness:

for MODEL in $models
do
  echo ooooooooooooooooooooooooooooooooooooooo
  head -1 predictions_${MODEL}_2016.csv              > all_predictions_${MODEL}.csv
  cat predictions_${MODEL}_????.csv | grep -v cdate >> all_predictions_${MODEL}.csv
  echo prediction count:
  wc -l all_predictions_${MODEL}.csv
  python ${TF}/rpt_model.py $MODEL
  echo ooooooooooooooooooooooooooooooooooooooo
done

exit
