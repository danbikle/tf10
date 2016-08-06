# gen1hf.py

# This script should generate 1hot-encoded features from dates.

# Demo:
# cd ${TFTMP}/csv
# ${HOME}/anaconda3/bin/python ${TF}/gen1hf.py GSPC2.csv

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
  print('${HOME}/anaconda3/bin/python ${TF}/gen1hf.py GSPC2.csv')
  sys.exit()

infile = sys.argv[1]
print('I am building features from this file:')
print(infile)
print('Busy...')

df1 = pd.read_csv(infile)
df1.columns = ['cdate','cp']

# I should order cdate_l,cp by date ascending:
cdate_l = list(reversed(df1['cdate'].values))
cp_l    = list(reversed(df1['cp'].values   ))

# I should work towards pctlead_a:
cplead_l  = cp_l + [cp_l[-1]]
cp_a      = np.array(cp_l)
cplead_a  = np.array(cplead_l[1:])
pctlead_a = 100.0 * (cplead_a - cp_a)/cp_a

# syntax to study via pdb:
for day_s in cdate_l:
    # pdb.set_trace()
    my_dt     = dt.strptime(day_s, "%Y-%m-%d")
    weekday_i = my_dt.weekday()         # Monday is 0
    wday_i    = dt.strftime(my_dt,'%w') # Monday is 1
    dom_i     = dt.strftime(my_dt,'%-d') # day of month
    moy_i     = dt.strftime(my_dt,'%-m') # month of year 1 through 12
    woy_i     = dt.strftime(my_dt,'%W')  # week of year
# syntax I actually use:
date_l      = [dt.strptime(day_s, "%Y-%m-%d") for day_s in cdate_l]
weekday_i_l = [day_dt.weekday()               for day_dt in date_l]
wday_i_l    = [dt.strftime(day_dt,'%w')       for day_dt in date_l]
dom_i_l     = [dt.strftime(day_dt,'%-d')      for day_dt in date_l]
moy_i_l     = [dt.strftime(day_dt,'%-m')      for day_dt in date_l]
woy_i_l     = [dt.strftime(day_dt,'%W')       for day_dt in date_l]


df2 = pd.DataFrame(cdate_l)
df2.columns = ['cdate']
df2['cp']   = cp_l
df2['pctlead'] = pctlead_a
df2['wday']    = wday_i_l
df2['dom']     = dom_i_l
df2['moy']     = moy_i_l
df2['woy']     = woy_i_l

ofd_a = np.array(df2[['wday','dom','moy','woy']])
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(ofd_a)
pdb.set_trace()
enc.n_values_

# I should save my work into a CSV file.
# My input file should look something like this:
# GSPC2.csv
# I should save my work as something like this:
# ftr_1hfGSPC2.csv
df2.to_csv('ftr_1hf'+infile, float_format='%4.3f', index=False)
print('Done...')

# done
