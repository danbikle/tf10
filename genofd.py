# genofd.py

# This script should generate ordinal features from dates.

# Demo:
# cd ${TFTMP}/csv
# ${HOME}/anaconda3/bin/python ${TF}/genofd.py GSPC2.csv

import pandas as pd
import numpy  as np
import pdb

# I should check cmd line arg
import sys

if len(sys.argv) == 1:
  print('Demo:')
  print('cd ${TFTMP}/csv')
  print('${HOME}/anaconda3/bin/python ${TF}/genofd.py GSPC2.csv')
  sys.exit()

infile = sys.argv[1]
print('I am building features from this file:')
print(infile)
print('Busy...')

df1  = pd.read_csv(infile)
df1.columns = ['cdate','cp']
