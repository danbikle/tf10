#!/bin/bash

# night10.bash

# This script should calculate predictions of the S&P500.

. envtf.bash

# I should get prices
${TF}/wgetGSPCnight.bash

# I should generate features from prices:
cd ${TFTMP}/csv
${HOME}/anaconda3/bin/python ${TF}/genf.py GSPC2.csv

exit
