# train_test_tf12.py

# This script should use train and test CSV data in ${TFTMP}/csv
# to train and test.
# The results should get written to CSV files in ${TFTMP}/csv

# Demo:
# cd ${TFTMP}/csv
# ${HOME}/anaconda3/bin/python ${TF}/train_test_tf12.py $STARTYR $ENDYR
# tail ${TFTMP}/csv/predictions_tf12_2016.csv

model_name = 'tf12'

# reusable syntax:

import numpy  as np
import pandas as pd
import pdb
import tensorflow as tf
from sklearn import linear_model
  
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
  pctlag1_i  = 3
  pctlag2_i  = 4
  pctlag4_i  = 5
  pctlag8_i  = 6
  pctlag16_i = 7
  end_i      = 8
  x_train_a  = train_a[:,pctlag1_i:end_i] # Machine should learn from this.
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
  x_test_a  = test_a[:,pctlag1_i:end_i]
  y_test_a  = test_a[:,pctlead_i]
  label_test_a = (test_a[:,pctlead_i] > class_boundry_f)
  ytest1h_a = np.array([[0,1] if tf else [1,0] for tf in label_test_a])

  #####################
  # model specific syntax:
  learning_rate = 0.001
  max_steps_i   = 9
  print(str(yr)+' VERY Busy...')
  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    And adds a number of summary ops.
    """
    # layer_name should be used in later version.
    # I should hold the state of the weights for the layer.
    weights     = weight_variable([input_dim, output_dim])
    biases      = bias_variable([output_dim])    
    preactivate = tf.matmul(input_tensor, weights) + biases
    # I should use act() passed in via arg.
    # Default: act=tf.nn.relu
    activations = act(preactivate, 'activation')
    return activations
  
  xvals     = tf.placeholder(tf.float32, shape=[None, fnum_i] , name='x-input')
  yactual   = tf.placeholder(tf.float32, shape=[None, label_i], name='y-input')
  keep_prob = tf.placeholder(tf.float32, name='probability2keep-not-drop')
  yhat_train_l = [[0.0, 1.0]]*len(ytrain1h_a)
  yhat_test_l  = [[0.0, 1.0]]*len(ytest1h_a)
  yhat_train   = tf.Variable(yhat_train_l)
  yhat_test    = tf.Variable(yhat_test_l)
  #  yhat = nn_layer(dropped, fnum_i, label_i, 'layer1', act=tf.nn.softmax)
  yhat = nn_layer(xvals, fnum_i, label_i, 'layer1', act=tf.nn.softmax)
  yhat         = yhat_train
  
  cross_entropy = -tf.reduce_mean(yactual * tf.log(yhat))
  train_step    = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
  tf.initialize_all_variables().run()
  
  for i in range(max_steps_i):
    sess.run(train_step, feed_dict={xvals: x_train_a, yactual: ytrain1h_a, keep_prob: 1.0})

  prob_a = sess.run(yhat_test, feed_dict={xvals: x_test_a, yactual: ytest1h_a,  keep_prob: 1.0})
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
