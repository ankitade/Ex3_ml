from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations

import numpy as np
import pandas as pd
import xgboost as xgb
SEED = 333

def group_data(data, degree, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'


def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.roc_auc_score(y_cv, preds)
        mean_auc += auc
    return mean_auc/N
    
def main(train='train.csv', test='test.csv', submit='logistic_pred.csv'):    
    print "Reading dataset..."
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))
    
    num_train = np.shape(train_data)[0]
    
    # Transform data
    dp = group_data(all_data, degree=2) 
    dt = group_data(all_data, degree=3)
    df = group_data(all_data, degree=4)
    y = array(train_data.ACTION)
    X = all_data[:num_train]
    X_2 = dp[:num_train]
    X_3 = dt[:num_train]
#    X_4 = df[:num_train]
    X_test = all_data[num_train:]
    X_test_2 = dp[num_train:]
    X_test_3 = dt[num_train:]
#    X_test_4 = df[num_train:]
    X_train_all = np.hstack((X, X_2, X_3))
    X_test_all = np.hstack((X_test, X_test_2, X_test_3))
    num_features = X_train_all.shape[1]

    #choose a mdoel
    #logistic regression        
    model = linear_model.LogisticRegression()
    #xgboostclassifier
#    model = xgb.XGBClassifier()

    Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]
    score_hist = []
    N = 10
    good_features = set([])
    # Greedy feature selection loop
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        print score_hist        
        scores = []
        for f in range(len(Xts)):
            if f not in good_features:
                feats = list(good_features) + [f]
                Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                score = cv_loop(Xt, y, model, N)
                scores.append((score, f))
                print "Feature: %i Mean AUC: %f" % (f, score)
        good_features.add(sorted(scores)[-1][1])
        score_hist.append(sorted(scores)[-1])
        print "Current features: %s" % sorted(list(good_features))
    
    # Remove last added feature from good_features
    good_features.remove(score_hist[-1][1])
    good_features = sorted(list(good_features))
    print "Selected features %s" % good_features
    # Hyperparameter selection loop
    score_hist = []
    Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
    
#    train model parameters
#    train logistic regression model parameter    
    Cvals = np.logspace(-4, 4, 15, base=2)
    for C in Cvals:
        model.C = C
        score = cv_loop(Xt, y, model, N)
        score_hist.append((score,C))
        print "C: %f Mean AUC: %f" %(C, score)
    bestC = sorted(score_hist)[-1][1]
    print "Best C value: %f" % (bestC)
    
#    model = xgb.XGBClassifier( learning_rate =0.1, n_estimators=400, max_depth=10,reg_alpha = 0.1,
#                              min_child_weight=1, gamma=0.3, subsample=0.9, colsample_bytree=0.6,
#                              objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=20)
#    model = xgb.XGBClassifier()
    
    Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
    Xt, keymap = OneHotEncoder(Xt)
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]
    ##save transformed dataset
    np.savez('train', data = X_train.data ,indices=X_train.indices, indptr = X_train.indptr, shape = X_train.shape)
    np.savez('test',  data = X_test.data , indices=X_test.indices,  indptr = X_test.indptr,  shape = X_test.shape)
#    loader = np.load('train.npz')
#    train = scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),shape = loader['shape'] )         
#    loader = np.load('train.npz')
#    test = scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),shape = loader['shape'] )
    model.fit(X_train, y)
    preds = model.predict_proba(X_test)[:,1]
    create_test_submission(submit, preds)
    
if __name__ == "__main__":
    args = { 'train':  'train.csv',
             'test':   'test.csv',
             'submit': 'logistic_regression_pred.csv' }
    main(**args)
    
