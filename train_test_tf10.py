# train_test_tf10.py

# This script should use train and test CSV data in ${TFTMP}/csv
# to train and test.
# The results should get written to CSV files in ${TFTMP}/csv

# Demo:
# cd ${TFTMP}/csv
# ${HOME}/anaconda3/bin/python ${TF}/train_test_tf10.py $STARTYR $ENDYR
# tail ${TFTMP}/csv/predictions_tf10_2016.csv

model_name = 'tf10'

# reusable syntax:

import numpy  as np
import pandas as pd
import pdb
import tensorflow as tf

# I should check cmd line arg
import sys
if (len(sys.argv) < 3):
  print('Demo:')
  print('cd ${TFTMP}/csv')
  print('${HOME}/anaconda3/bin/python ${TF}/train_test_'+model_name+'.py $STARTYR $ENDYR')
  sys.exit()

startyr = int(sys.argv[1])
finalyr = int(sys.argv[2])
class_boundry_f = 0.03 # days above this are in 'up' class.
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
  ytrain1h_a = np.array([[0,1] if tf else [1,0] for tf in (y_train_a > class_boundry_f)])

  # [0,1] means up-observation
  # [1,0] means down-observation

  # I declare 2d Tensors.
  # I should use 0th row of x_train_a to help shape x:
  fnum_i  = len(x_train_a[0, :])
  label_i = len(ytrain1h_a[0,:])

  # I should learn from x_train_a,label_train_a.

  xvals = tf.placeholder(tf.float32, shape=[None, fnum_i], name='x-input')
  
  #####################
  # model specific syntax:
  weight1 = tf.Variable(tf.zeros([fnum_i, label_i]))
  weight0 = tf.Variable(tf.zeros([label_i]))
  yhat    = tf.nn.softmax(tf.matmul(xvals, weight1) + weight0)
  # Define loss and optimizer
  yactual       = tf.placeholder(tf.float32, [None, label_i])
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(yactual * tf.log(yhat), reduction_indices=[1]))
  train_step    = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  # Train
  tf.initialize_all_variables().run()
  for i in range(100):
    train_step.run({xvals: x_train_a, yactual: ytrain1h_a})
  #####################

  # reusable syntax:
  # Now that I have learned, I should predict:
  testf     = 'test'+str(yr)+'.csv'
  test_df   = pd.read_csv(testf)
  test_a    = np.array(test_df)
  x_test_a  = test_a[:,pctlag1_i:end_i]
  # I should compute predictions from x_test_a:
  prob_a    = sess.run(yhat, feed_dict={xvals: x_test_a})
  # Here is the actual data.
  y_test_a  = test_a[:,pctlead_i]
  ytest1h_a = np.array([[0,1] if tf else [1,0] for tf in (y_test_a > class_boundry_f)])
  # I should collect predictions along with actual data.
  # softmax() gives me probabilities of all classes.
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
  
  
