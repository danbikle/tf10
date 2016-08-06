# rpt_model.py

# This script should report accuracy and effectiveness of a model

import pandas as pd
import numpy  as np
import pdb

# I should check cmd line arg
import sys

if len(sys.argv) == 1:
  print('Demo:')
  print('cd ${TFTMP}/csv')
  print('${HOME}/anaconda3/bin/python ${TF}/rpt_model.py tf11')
  sys.exit()

model = sys.argv[1]

my_df = pd.read_csv('all_predictions_'+model+'.csv')
print('For model: '+model)
print('Accuracy sum is:')
print(np.sum(my_df['accuracy']))
print('Accuracy is:')
pred = (my_df['accuracy'] == 1)
print(str(np.round(100.0 * len(my_df[pred]) / len(my_df))) + '%')

print('Effectiveness sum is:')
print(np.round(np.sum(my_df['eff1d'])))
print('Competition: Long Only Results:')
print('Long Only Accuracy sum is:')
print(np.round(np.sum(np.sign(my_df['pctlead']))))

print('Long Only Accuracy is:')
pred = (my_df['pctlead'] >= 0)
print(str(np.round(100.0 * len(my_df[pred]) / len(my_df))) + '%')

print('Long Only Effectiveness sum is:')
print(np.round(np.sum(my_df['pctlead'])))
