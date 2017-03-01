# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:04:05 2017
@author: sushant.kulkarni

======================================================
Classification of text documents using sparse features
======================================================

This is a script involving scikit-learn to classify documents
by topics using a bag-of-words approach. It uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

"""
from __future__ import print_function

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from time import time
from optparse import OptionParser

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


train = pd.read_csv("output/training_2_28.csv", header=0)
test =  pd.read_csv("output/test_2_28.csv", header=0)
# retain only necessary rows
test = test[['Unnamed: 0', 'Activity Notes']]
test.columns = ['id', 'Activity Notes']

data_train = sklearn.datasets.base.Bunch(data= train['Activity Notes'], 
                                         target = train['class'],
                                         target_names = ['c1', 'c2', 'o'])

data_test = sklearn.datasets.base.Bunch(data= test['Activity Notes'])

print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6