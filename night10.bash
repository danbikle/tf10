#!/bin/bash

# night10.bash

# This script should calculate predictions of the S&P500.

. envtf.bash

# I should get prices
${TF}/wgetGSPCnight.bash

# I should generate features from prices:
cd ${TFTMP}/csv
${HOME}/anaconda3/bin/python ${TF}/genf.py GSPC2.csv

# I should generate training and test data from features.

# Next I generate training data CSV files:

STARTYR=1981
ENDYR=2016
TRAINING_AMOUNT=20 #years
${HOME}/anaconda3/bin/python ${TF}/gentrain_test.py ftrGSPC2.csv $TRAINING_AMOUNT $STARTYR $ENDYR

exit
