# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:43:15 2017

@author: yashraj.arab
"""
import pandas as pd
import os
import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import spacy as sc
import gensim as gm
import nltk
import re
from collections import Counter
from itertools import chain, starmap
import string
from nltk import word_tokenize, tokenize, tokenize_text

from stop_words import get_stop_words
stop_words = get_stop_words('en')


os.chdir("C:/Users/naveen.dadala/Desktop/MARS_Notes")
df = pd.read_csv("mars_notes.csv")


#os.chdir("C:/Users/yashraj.arab/Desktop/")
#df = pd.read_csv("mars_notes.csv")

Columns = df['ACTVY_NOTE_TX']
#Columns.to_csv("comments.csv")



pd.DataFrame(Columns[1:10000]).to_csv('comm10000.csv',index=False)


f_read = open('comm10000.csv')
f_write = open('words10000.csv','a')

for line_no,line in enumerate(f_read):
    line = re.sub('[^A-Za-z0-9]+','\n',line.lower())
    f_write.write(line + '\n')
    print(line_no)

f_read.close()
f_write.close()

    

def tokens(text):
    return re.findall('[a-z]+', text.lower())
   
WORDS = tokens(open('comm10000.csv').read())
WORD_COUNTS = Counter(WORDS)

df = pd.DataFrame.from_dict(WORD_COUNTS,orient='index').reset_index()

print (df)


#f_stopwords = tokens(open('words10000.csv').read())


#def remove_stopwords(tokens):
#    stopword_list = stop_words
#    filtered_tokens = [token for token in tokens if token not in stopword_list]
#    return filtered_tokens
#
stoplist=[]
f_write = open('words10000.csv','r')
noise_removed=open('noise_removed.csv','w')       
stop_words = get_stop_words('en')

for line in stop_words:
    w = line.split()
    for word in w:
        stoplist.append(word)
print (stoplist)

import csv
a = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
with open("output1.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerow(a)

    

#for line in f_write:
#    w = line.split()
#    for word in w:
#        if word in stoplist: continue
#    else:
#        noise_removed.write(word)
#        
#f_write.close()
#noise_removed.close()
#stop.close()
f_write = open('words10000.csv','r')
stopword=open('output1.csv','r')
noise_removed=open('noise_removed.csv','w')     

first_words=[]
second_words=[]
for line in f_write:
    words=line.split()
    for w in words:
        first_words.append(a)

for line in stopword :
    w = line.split()
    for i in w:
        second_words.append(i)
        
for word1 in first_words:
    for word2 in second_words:
        if word1==word2:
            first_words.remove(word2)

for word in first_words:
    noise_removed.write(str(word))    

f_write.close()
stopword.close()
noise_removed.close()
    

#Working Code

f_read = open('comm10000.csv')
f_write = open('words10000_line.csv','a')

for line_no,line in enumerate(f_read):
    line = re.sub('[^A-Za-z0-9]+',' ',line.lower())
    text = [word for word in line.split() if word not in stoplist]
    #print(text)
    f_write.write(str(text) + '\n')
    print(line_no)
    #    if line_no == 10:
    #        break

f_read.close()
f_write.close()

my_corp = []

f =  open('words10000_line.csv','r')

for line in f:
    my_corp.append(re.sub('[\[|\]]','',line))
    
f.close()


texts = [word for word in str(my_corp).split()]











 
        
        