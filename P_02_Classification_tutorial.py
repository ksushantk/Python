# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 11:33:01 2017

@author: sushant.kulkarni
"""

print(__doc__)

#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

#%%

# Load dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

### Initial model with TF-IDF matrix as features
# Build TF-IDF	matrix for training dataset
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=10, max_df=0.90, ngram_range=(1,2), use_idf=True)
X_train = vectorizer.fit_transform(train['lemm'].str.lower())
print("TF-IDF data dimension is %s" % str(X_train.shape))

# Get feature names after TF-IDF vectorization
feature_names = vectorizer.get_feature_names()
print()

X_test = vectorizer.transform(test['lemm'])

# Train Naive bayes using TF -IDF matrix
clf = MultinomialNB().fit(X_train, target)
test_op = pd.DataFrame(clf.predict_proba(X_test), test['id']).reset_index()
test_op.columns = ['id', 'raw_p1', 'raw_p2']
test_op = test_op.merge(test, how='inner', on = 'id')

# Get probabilty scores for training data
train_op = pd.DataFrame(clf.predict_proba(X_train), train['id']).reset_index()
train_op.columns = ['id', 'raw_p1', 'raw_p2']

# Second Model with Bag of Words presence as features
train['cnt'] = nprnd.randint(10, size=25000)
test_op['cnt'] = nprnd.randint(10, size=50000)
test_op['mini']=test_op[['raw_p1', 'raw_p2']].min(axis=1)
test_op['max']=test_op[['raw_p1', 'raw_p2']].max(axis=1)
test_op['score'] = test_op['mini'].where(test_op['cnt']==0, test_op['max'])
test_op = test_op[['id', 'cnt', 'raw_p1', 'raw_p2', 'score']]

temp = pd.DataFrame(train[['cnt']])
temp['bin']=temp['cnt'].where(temp['cnt']==0,1)
clf=MultinomialNB().fit(temp, target)
# Get probabilty scores for training data
train_bow = pd.DataFrame(clf.predict_proba(temp), train['id']).reset_index()
train_bow.columns = ['id', 'bow_p1', 'bow_p2']

temp = pd.DataFrame(test_op[['cnt']])
temp['bin']=temp['cnt'].where(temp['cnt']==0,1)
test_bow = pd.DataFrame(clf.predict_proba(temp[['cnt', 'bin']]),test['id']).reset_index()
test_bow.columns = ['id', 'bow_p1', 'bow_p2'] # BOW prob

comb_train = train_op.merge(train_bow, how='inner', on ='id')
comb_test = test_op.merge(test_bow, how='inner', on='id')
clf = MultinomialNB().fit(comb_train[['raw_p1', 'raw_p2', 'bow_p1', 'bow_p2']], target)
fin_mod = pd.DataFrame(clf.predict_proba(comb_test[['raw_p1', 'raw_p2', 'BOW_P1', 'BOW_P2']]),test_op['id']).reset_index()
fin_mod = fin_mod.merge(test_op[['id', 'score']], how='inner')

from datetime import datetime
import calendar
t = pd.read_csv('datetest.csv')
t['new']=t['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%m/%d/%Y'))
t['month']=t['date'].apply(lambda x: calendar.month_name[int(str(x)[4:6])])
