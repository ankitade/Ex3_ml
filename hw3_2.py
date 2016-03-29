
# coding: utf-8

# In[4]:

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
import scipy
import random
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# In[5]:

Y_train = df.ix[:,0]
X_train = df.ix[:,[1,2,3,5,6,7,8,9]]
X_test = df_test.ix[:,[1,2,3,5,6,7,8,9]]



# In[9]:

model = xgb.XGBClassifier( learning_rate =0.1, n_estimators=400, max_depth=10,reg_alpha = 0.1,
 min_child_weight=1, gamma=0.3, subsample=0.9, colsample_bytree=0.6,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=33)
model.fit(X_train,Y_train)
predictions = model.predict_proba(X_test)[:,1]
submission = pd.DataFrame({'Id': df_test.ix[:,0], 'Action' : predictions})
submission.to_csv("submission_xgb.csv", index=False)


# In[ ]:



