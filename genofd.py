# genofd.py

# This script should generate ordinal features from dates.

# Demo:
# cd ${TFTMP}/csv
# ${HOME}/anaconda3/bin/python ${TF}/genofd.py GSPC2.csv

# ref:
# http://strftime.org

import pandas as pd
import numpy  as np
import pdb
from datetime import datetime as dt

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

df1 = pd.read_csv(infile)
df1.columns = ['cdate','cp']



for day_s in df1['cdate']:
    my_dt     = dt.strptime(day_s, "%Y-%m-%d")
    weekday_i = my_dt.weekday()         # Monday is 0
    wday_i    = dt.strftime(my_dt,'%w') # Monday is 1
    dom_i     = dt.strftime(my_dt,'%-d') # day of month
date_l      = [dt.strptime(day_s, "%Y-%m-%d") for day_s in df1['cdate']]
weekday_i_l = [day_dt.weekday()               for day_dt in date_l]
wday_i_l    = [dt.strftime(day_dt,'%w')       for day_dt in date_l]
dom_i_l     = [dt.strftime(day_dt,'%-d')      for day_dt in date_l]
print(df1.head())
print(weekday_i_l[:11])
print(wday_i_l[:11])
print(dom_i_l[:11])

