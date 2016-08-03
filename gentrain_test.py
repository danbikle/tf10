# gentrain_test.py

# This script should give me two sets of CSV files.
# In the first set, the training set,
# Each file should contain at least 20 years of data.
# Actually that is configurable, but 30 years is good.

# ref:
# https://github.com/danbikle/sklearnem/blob/master/multiyr/gentrain_test.py

# Demo:
# cd $TFTMP/csv/
# STARTYR=2013
# ENDYR=2016
# TRAINING_AMOUNT=20 #years
# ${HOME}/anaconda3/bin/python ${TF}/gentrain_test.py ftrGSPC2.csv $TRAINING_AMOUNT $STARTYR $ENDYR

# Above demo gives me:
# ${HOME}/tftmp/csv/train2013.csv
# ${HOME}/tftmp/csv/train2013.csv
# ${HOME}/tftmp/csv/train2014.csv
# ${HOME}/tftmp/csv/train2015.csv
# ${HOME}/tftmp/csv/train2016.csv
#
# ${HOME}/tftmp/csv/test2013.csv
# ${HOME}/tftmp/csv/test2014.csv
# ${HOME}/tftmp/csv/test2015.csv
# ${HOME}/tftmp/csv/test2016.csv
# For each year, the train data should not overlap the test data.

import pandas as pd
import numpy  as np
import pdb

# I should check cmd line arg
import sys
if (len(sys.argv) < 5):
  print('Demo:')
  print('cd ${TFTMP}/csv')
  print('python ${TF}/gentrain_test.py ftrGSPC2.csv 30 $STARTYR $ENDYR')
  sys.exit()

infile = sys.argv[1]
print('I am building CSV files from this file:')
print(infile)
print('Busy...')
numyr   = int(sys.argv[2])
startyr = int(sys.argv[3])
finalyr = int(sys.argv[4])

# I should create a loop which builds the files I want:

for yr in range(startyr,1+finalyr):
  boundry_right = str(yr)
  boundry_left  = str(yr-numyr)
  # I should create a DF from infile
  in_df = pd.read_csv(infile)
  # I should get RHS of df:
  rhs_pred = (in_df['cdate'] > boundry_left)
  rhs_df   =  in_df[rhs_pred]
  # I should get LHS of df:
  lhs_pred = (rhs_df['cdate'] < boundry_right)
  lhs_df   =  rhs_df[lhs_pred]
  # I should write the df to csv file:
  lhs_df.to_csv('train'+str(yr)+'.csv', float_format='%4.3f', index=False)
  # I should get the test data now.
  boundry_right_test = str(yr+1)
  boundry_left_test  = str(yr)
  # I should create a DF from infile
  in_df = pd.read_csv(infile)
  # I should get RHS of df:
  rhs_test_pred = (in_df['cdate'] > boundry_left_test)
  rhs_test_df   =  in_df[rhs_test_pred]
  # I should get LHS of df:
  lhs_test_pred = (rhs_test_df['cdate'] < boundry_right_test)
  lhs_test_df   =  rhs_test_df[lhs_test_pred]
  # I should write the df to csv file:
  lhs_test_df.to_csv('test'+str(yr)+'.csv', float_format='%4.3f', index=False)
# I should now have some CSV files.
print('Done.')
