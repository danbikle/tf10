# train_test_tf10.py

# This script should use train and test CSV data in ${TFTMP}/csv
# to train and test.
# The results should get written to CSV files in ${TFTMP}/csv

# Demo:
# cd ${TFTMP}/csv
# ${HOME}/anaconda3/bin/python ${TF}/train_test_tf10.py $STARTYR $ENDYR

import numpy  as np
import pandas as pd
import pdb
import tensorflow as tf

# I should check cmd line arg
import sys
if (len(sys.argv) < 3):
  print('Demo:')
  print('cd ${TFTMP}/csv')
  print('${HOME}/anaconda3/bin/python ${TF}/train_test_tf10.py $STARTYR $ENDYR')
  sys.exit()

startyr = int(sys.argv[1])
finalyr = int(sys.argv[2])

sess = tf.InteractiveSession()

# I should create a loop which does train and test for each yr.
for yr in range(startyr,1+finalyr):
  trainf = 'train'+str(yr)+'.csv'
  train_df = pd.read_csv(trainf)
  train_a  = np.array(train_df)
  # I should declare some integers to help me navigate the Arrays.
  cdate_i    = 0
  cp_i       = 1
  pctlead_i  = 2
  pctlag1_i  = 3
  pctlag2_i  = 4
  pctlag4_i  = 5
  pctlag8_i  = 6
  pctlag16_i = 7
  end_i      = 8
  x_train_a  = train_a[:,pctlag1_i:end_i]
  y_train_a  = train_a[:,pctlead_i]
  # TF wants labels to be 1-hot-encoded:
  ytrain1h_a = np.array([[0,1] if tf else [1,0] for tf in (y_train_a > 0.03)])
  # [0,1] means up-observation
  # [1,0] means down-observation

  # I declare 2d Tensors.
  # I should use 0th row of x_train_a to help shape x:
  fnum_i  = len(x_train_a[0, :])
  label_i = len(ytrain1h_a[0,:])

  # I should learn from x_train_a,label_train_a.

  xvals = tf.placeholder(tf.float32, shape=[None, fnum_i], name='x-input')

'bye'