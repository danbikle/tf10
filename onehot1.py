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

days_dt_l = [datetime.datetime.strptime(day_s, "%Y-%m-%d") for day_s in days_l]
days_i_l  = [day_dt.weekday() for day_dt in days_dt_l]
print(days_i_l)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
pdb.set_trace()
data_a = np.array([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
enc.fit(data_a)
# This should give ...
# n_values_ : array of shape (n_features,)
#    Maximum number of values per feature.
enc.n_values_
# array([2, 3, 4])
# The above array has same number of elems as columns in data.
# The 2 says that the 0th col has 2 possible values.
# The 3 says that the data_a[:,1] has 3 possible values.
# The 3 says that the data_a[:,2] has 4 possible values: 0,1,2,3

'bye'
