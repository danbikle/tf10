# train_test_sk_lr.py

# This script should use train and test CSV data in ${TFTMP}/csv
# to train and test.
# The results should get written to CSV files in ${TFTMP}/csv

# Demo:
# cd ${TFTMP}/csv
# ${HOME}/anaconda3/bin/python ${TF}/train_test_sk_lr.py $STARTYR $ENDYR

import numpy  as np
import pandas as pd
import pdb

# I should check cmd line arg
import sys
if (len(sys.argv) < 3):
  print('Demo:')
  print('cd ${TFTMP}/csv')
  print('${HOME}/anaconda3/bin/python ${TF}/train_test_sk_lr.py $STARTYR $ENDYR')
  sys.exit()

startyr = int(sys.argv[1])
finalyr = int(sys.argv[2])

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
  train_median  = np.median(y_train_a)
  label_train_a = y_train_a > train_median
  # I should learn from x_train_a,label_train_a:

  from sklearn import linear_model
  from sklearn.naive_bayes import GaussianNB

  clf_lr = linear_model.LogisticRegression()
  clf_nb = GaussianNB()
  clf_lr.fit(x_train_a, label_train_a)
  clf_nb.fit(x_train_a, label_train_a)

  # Now that I have learned, I should predict:
  testf    = 'test'+str(yr)+'.csv'
  test_df  = pd.read_csv(testf)
  test_a   = np.array(test_df)
  x_test_a = test_a[:,pctlag1_i:end_i]
  y_test_a = test_a[:,pctlead_i]
  label_test_a  = y_test_a > train_median
  prob_lr_l        = []
  predictions_lr_l = []
  eff1d_lr_l       = [0.0]
  recent_eff_lr_l  = [0.0]
  acc_lr_l         = []
  predictions_nb_l = []
  eff1d_nb_l       = [0.0]
  recent_eff_nb_l  = [0.0]
  acc_nb_l         = []
  xcount           = -1
  for xoos_a in x_test_a:
    xcount        += 1 # should == 0 1st time through
    xf_a           = xoos_a.astype(float)
    xr_a           = xf_a.reshape(1, -1)
    aprediction_lr = clf_lr.predict_proba(xr_a)[0,1]
    aprediction_nb = clf_nb.predict(xr_a)[0]
    prob_lr_l.append(aprediction_lr)
    if (aprediction_lr > 0.5):
      predictions_lr_l.append(1)  # up   prediction
    else:
      predictions_lr_l.append(-1) # down prediction
    if (aprediction_nb == True):
      predictions_nb_l.append(1)  # up   prediction
    else:
      predictions_nb_l.append(-1) # down prediction
    # I should save effectiveness of each prediction:
    pctlead = y_test_a[xcount]
    eff1d_lr_l.append(predictions_lr_l[xcount]*pctlead)
    eff1d_nb_l.append(predictions_nb_l[xcount]*pctlead)
    # I should save recent effectiveness of each prediction:
    if (xcount < 5):
      recent_eff_lr_l.append(0.0)
      recent_eff_nb_l.append(0.0)
    else:
      recent_eff_lr_l.append(np.mean(eff1d_lr_l[-5:]))
      recent_eff_nb_l.append(np.mean(eff1d_nb_l[-5:]))
    # I should save accuracy of each prediction
    #
    if ((pctlead > 0) and (aprediction_lr > 0.5)):
      acc_lr_l.append('True Positive')
    elif ((pctlead > 0) and (aprediction_lr < 0.5)):
      acc_lr_l.append('False Negative')
    elif ((pctlead < 0) and (aprediction_lr > 0.5)):
      acc_lr_l.append('False Positive')
    else:
      acc_lr_l.append('True Negative')
    #
    if ((pctlead > 0) and (aprediction_nb == True)):
      acc_nb_l.append('True Positive')
    elif ((pctlead > 0) and (aprediction_nb == False)):
      acc_nb_l.append('False Negative')
    elif ((pctlead < 0) and (aprediction_nb == True)):
      acc_nb_l.append('False Positive')
    else:
      acc_nb_l.append('True Negative')
    #
    'end for'
  # I should save predictions, eff, acc, so I can report later.
  test_df['actual_dir']    = np.sign(test_df['pctlead'])
  #
  test_df['prob_lr']       = prob_lr_l
  test_df['pdir_lr']       = predictions_lr_l
  test_df['eff1d_lr']      = eff1d_lr_l[1:]
  test_df['recent_eff_lr'] = recent_eff_lr_l[1:]
  if (len(test_df) - len(acc_lr_l) == 1):
    # I should deal with most recent observation:
    acc_lr_l.append('unknown')
  test_df['accuracy_lr'] = acc_lr_l
  #
  # I should write to CSV:
  test_df.to_csv('predictions_sk_lr'+str(yr)+'.csv', float_format='%4.3f', index=False)
'bye'
