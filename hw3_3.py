
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
import scipy
import random
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# In[2]:

Y_train = df.ix[:,0]
X_train = df.ix[:,[1,2,3,5,6,7,8,9]]
X_test = df_test.ix[:,[1,2,3,5,6,7,8,9]]


# In[3]:

n = 10
for i in range(n):
    s = random.randint(0,100)
    model = xgb.XGBClassifier( learning_rate =0.1, n_estimators=400, max_depth=10,reg_alpha = 0.1,
 min_child_weight=1, gamma=0.3, subsample=0.9, colsample_bytree=0.6,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=s)
    model.fit(X_train,Y_train)
    predictions = model.predict_proba(X_test)[:,1]
    submission = pd.DataFrame({'Id': df_test.ix[:,0], 'Action' : predictions})
    submission.to_csv("submission_average"+str(i)+".csv", index=False)


# In[4]:

predictions = predictions / float(n)
submission = pd.DataFrame({'Id': df_test.ix[:,0], 'Action' : predictions})
submission.to_csv("submission_average.csv", index = False)


# In[5]:

predictions = np.zeros(len(X_test))
df = pd.read_csv("submission_average.csv")
predictions = predictions + 1*np.array(df['Action'])
df = pd.read_csv("lrfeature3s333.csv")
predictions = predictions +18*np.array(df['ACTION'])
df = pd.read_csv("lrfeature3s1001.csv")
predictions = predictions +9*np.array(df['ACTION'])
df = pd.read_csv("log75.csv")
predictions = predictions +27*np.array(df['ACTION'])
predictions = predictions /55.0
submission = pd.DataFrame({'Id': df_test.ix[:,0], 'Action' : predictions})
submission.to_csv("submission_log_average.csv", index = False)


# In[ ]:



