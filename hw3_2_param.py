
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



# In[ ]:

param1 =  { 'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(X_train,Y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[ ]:

param1 =  { 'max_depth':[8,9,10],'min_child_weight':[1,2,3]}
gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(X_train,Y_train)


# In[ ]:

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=10,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train,Y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[ ]:

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=10,
 min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_train,Y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[ ]:

param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=10,
 min_child_weight=1, gamma=0.1, subsample=0.9, colsample_bytree=0.6,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(X_train,Y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[ ]:

model = xgb.XGBClassifier( learning_rate =0.1, n_estimators=400, max_depth=10,reg_alpha = 0.1,
 min_child_weight=1, gamma=0.3, subsample=0.9, colsample_bytree=0.6,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=33)
model.fit(X_train,Y_train)
predictions = model.predict_proba(X_test)[:,1]
submission = pd.DataFrame({'Id': df_test.ix[:,0], 'Action' : predictions})
submission.to_csv("submission_xgb.csv", index=False)

