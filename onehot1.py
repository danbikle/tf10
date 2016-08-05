# onehot1.py

# ref:
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

import pandas as pd
import numpy  as np
import pdb
import sklearn
import datetime

days_l = []
days_l.append('2016-08-01')
days_l.append('2016-08-02')
days_l.append('2016-08-03')
days_l.append('2016-08-04')
days_l.append('2016-08-05')

for day_s in days_l:
    my_dt = datetime.datetime.strptime(day_s, "%Y-%m-%d")
    day_i = my_dt.weekday()
    print(day_s+' gives:')
    print(day_i)
'bye'
