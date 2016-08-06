# train_test_sk_lr_1hf.py

# This script should use train and test CSV data in ${TFTMP}/csv
# to train and test.
# The results should get written to CSV files in ${TFTMP}/csv

# Demo:
# cd ${TFTMP}/csv
# ${HOME}/anaconda3/bin/python ${TF}/train_test_sk_lr_1hf.py $STARTYR $ENDYR
# tail ${TFTMP}/csv/predictions_sk_lr_1hf_2016.csv

# I use this model to test sklearn logistic regression on ordinal features built from dates:
model_name = 'sk_lr_1hf'
# The ordinal features are:
# week of day 1,2,3,4,5
# day of month 1 through 31
# month of year 1 through 12
# week of year 0 through 53

# reusable syntax:

import numpy  as np
import pandas as pd
import pdb
import tensorflow as tf
from sklearn import linear_model
# from sklearn.preprocessing import OneHotEncoder
import sklearn

# I should check cmd line arg
import sys
if (len(sys.argv) < 3):
  print('Demo:')
  print('cd ${TFTMP}/csv')
  print('${HOME}/anaconda3/bin/python ${TF}/train_test_'+model_name+'.py $STARTYR $ENDYR')
  sys.exit()

startyr = int(sys.argv[1])
finalyr = int(sys.argv[2])
sess    = tf.InteractiveSession()
class_boundry_f = 0.03 # days above this are in 'up' class.
learning_rate   = 0.001
# I should create a loop which does train and test for each yr.
for yr in range(startyr,1+finalyr):
  trainf   = 'train'+str(yr)+'.csv' # Data should be in this file.
  train_df = pd.read_csv(trainf)
  train_a  = np.array(train_df)     # Data should be in this Array.
  # I should declare some integers to help me navigate the Arrays.
  cdate_i    = 0
  cp_i       = 1
  pctlead_i  = 2
  # unused:
  #  pctlag1_i  = 3
  #  pctlag2_i  = 4
  #  pctlag4_i  = 5
  #  pctlag8_i  = 6
  #  pctlag16_i = 7
  #  end_i      = 8
  wday_i = 3
  dom_i  = 4
  moy_i  = 5
  woy_i  = 6
  end_i  = 7
  #x_train_a  = train_a[:,pctlag1_i:end_i] # Machine should learn from this.
  x_train_o_a  = train_a[:,wday_i:end_i] # Ordinal features.
  # I should 1hot-encode the ordinal features:
  enc = sklearn.preprocessing.OneHotEncoder()
  enc.fit(x_train_o_a)
  x_train_a = enc.transform(x_train_o_a).toarray() # My encoded features.
  
  # sklearn can use label_train_a:
  label_train_a = (train_a[:,pctlead_i] > class_boundry_f) # And this too.
  # But, TF wants labels to be 1-hot-encoded:
  ytrain1h_a = np.array([[0,1] if tf else [1,0] for tf in label_train_a])
  # [0,1] means up-observation
  # [1,0] means down-observation

  # I declare 2d Tensors.
  # I should use 0th row of x_train_a to help shape xvals:
  fnum_i  = len(x_train_a[0, :])
  label_i = len(ytrain1h_a[0,:]) # Should usually be 2.
  # The test data should help me gauge Accuracy and Effectiveness:
  testf     = 'test'+str(yr)+'.csv' # Data should be in this file.
  test_df   = pd.read_csv(testf)
  test_a    = np.array(test_df)
  #  x_test_a  = test_a[:,pctlag1_i:end_i]
  x_test_o_a  = test_a[:,wday_i:end_i] # Ordinal features
  # I should test using 1hot-features:
  x_test_a = enc.transform(x_test_o_a).toarray() # My encoded features.
  y_test_a  = test_a[:,pctlead_i]
  label_test_a = (test_a[:,pctlead_i] > class_boundry_f)
  ytest1h_a = np.array([[0,1] if tf else [1,0] for tf in label_test_a])

  #####################
  # model specific syntax:
  clf = linear_model.LogisticRegression()
  clf.fit(x_train_a, label_train_a)
  pdb.set_trace()
  prob_a = clf.predict_proba(x_test_a)
  #####################

  # reusable syntax:
  # I should write Accuracy and Effectiveness to CSV file.
  # I only want the probability of the 'up' class:
  prob_l        = [prob[1] for prob in prob_a]
  prob_a        = np.array(prob_l)
  predictions_l = [1 if tf else -1 for tf in (prob_a >= 0.5)]
  eff1d_a       = np.array(predictions_l) * y_test_a
  acc_a         = np.sign(eff1d_a)
  # I should save predictions, eff, acc, so I can report later.
  test_df['actual_dir'] = np.sign(y_test_a)
  test_df['prob']       = prob_l
  test_df['pdir']       = predictions_l
  test_df['eff1d']      = eff1d_a
  test_df['accuracy']   = acc_a
  test_df.to_csv('predictions_'+model_name+'_'+str(yr)+'.csv', float_format='%4.3f', index=False)  

  'bye'

  
