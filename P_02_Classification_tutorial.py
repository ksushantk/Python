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

