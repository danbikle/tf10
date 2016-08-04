#!/bin/bash

# night10.bash

# This script should calculate predictions of the S&P500.

# I should assume that I am in the same folder as envtf.bash:
. envtf.bash

# I should get prices
# debug ${TF}/wgetGSPCnight.bash

# I should generate features from prices:
cd ${TFTMP}/csv
${HOME}/anaconda3/bin/python ${TF}/genf.py GSPC2.csv

# Next I generate training data CSV files:

STARTYR=1981
ENDYR=2016
TRAINING_AMOUNT=20 #years
${HOME}/anaconda3/bin/python ${TF}/gentrain_test.py ftrGSPC2.csv $TRAINING_AMOUNT $STARTYR $ENDYR

# I should learn then test using sklearn logistic regression:
${HOME}/anaconda3/bin/python ${TF}/train_test_sk_lr.py $STARTYR $ENDYR

# I should use Tensorflow too:
#${HOME}/anaconda3/bin/python ${TF}/train_test_tf11.py $STARTYR $ENDYR

# I should report Accuracy and Effectiveness:
${TF}/rpt_model.bash

exit
