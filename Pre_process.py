# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

pd.set_option('display.max_colwidth', -1)
import operator
import string
import numpy as np
import spacy
nlp = spacy.laod('en')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#------------------------------------------------------------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english', min_df=1)
vec = vectorizer.fit_transform(convo)


"""
#------------------------------------------------------------------------------

from __future__ import division
import os
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import re
# import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

print(os.getcwd())
os.chdir("C:/Users/sushant.kulkarni/Desktop/P01")
print(os.getcwd())

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Sample 1% data
# data = pd.read_csv("ah170_2016_samp01pct.csv")
# Sample 10% data 
data = pd.read_csv("ah170_2016_samp10pct.csv")
data.shape
# (12304,9) 1% data
# (1203929, 9) 10% data

#------------------------------------------------------------------------------
# count number of tokens for each record
def filx(line):
    cnt = 0
    for token in word_tokenize(str(line)):
        cnt = cnt + 1
    return cnt
    
#------------------------------------------------------------------------------
### drop records with no tokens for Activity Notes
filt_data = data.dropna(subset=['Activity Notes'])
filt_data.shape
## (12303, 9) - 1% data
## (1203909, 9) - 10% data

### filter out all notes with just one or two tokens
filt_data = filt_data[filt_data.apply(lambda x: filx(x['Activity Notes'])>2, axis = 1)]
filt_data.shape
# (10806, 9) - 1497 records are dropped
# (1055094, 9) - 10% data
# filt_data.to_pickle('filtred_data.p')
# filt_data = pd.read_pickle('filtred_data.p')
###----------------------------------------------------------------------------

# retaining only text columns for further analyis
convo = pd.DataFrame(filt_data.iloc[:,4])

# Computing the number of characters
convo['Length'] = convo['Activity Notes'].apply(len)
plt.hist(convo['Length'])
# maximum number of characters are 1500

# Converting to lower case might result in some loss which could effect Further
# process such as POS tagging [difficult to differentiate Proper nouns]
# further str.split() is not efficient approach - as it fails to seperate out
# punctuations or other symbols at end of word
# convo['ActNotesSplit'] = convo['Activity Notes'].str.lower().str.split()

# tokenize the words using NLTK package
start_time = time.time()
convo['ActNotesSplit'] = convo['Activity Notes'].apply(lambda x: [item for item in word_tokenize(x)])
print("--- %s seconds ---" % (time.time() - start_time))

# Remove stop words and punctuation and other special characters
customStopWords = set(stopwords.words("english")+list(punctuation))
convo['ActNotes'] = convo['ActNotesSplit'].apply(lambda x: [item for item in x if item not in customStopWords])

# refining stopwords list manually based on visual inspection
StopWrds_list = stopwords.words("english") + list(punctuation)
StopWrds_list = pd.DataFrame(StopWrds_list)
StopWrds_list.columns = ['Stopwords']
StopWrds_list.to_csv("StopWords.csv")

StopWords_Mod = pd.read_csv('StopWords_Mod.csv')
StpWrds_Modlist = StopWords_Mod['Stopwords'].values.tolist()

# do stop word filters again and aplly less than 2 filter again. 
customStopWordsMod = set(StpWrds_Modlist)
convo['ActNotes'] = convo['ActNotesSplit'].apply(lambda x: [item for item in x if item not in customStopWordsMod])
convo.shape
## (10806, 4) - 1% data
## (1055094, 4) - 10% data
convo_filt = convo[convo.apply(lambda x: len(x['ActNotes'])>2, axis = 1)]
convo_filt.shape
## (10661, 4)
## (1039176, 4)

# convo_filt.to_pickle('filtred_moddata.p')
# convo_filt = pd.read_pickle('filtred_moddata.p')
###----------------------------------------------------------------------------


# find out n gramns
# Spelling correction - Levenshtein Ditances, Dictionary Correction
# slang Look up
# Grammar check
# Apostrophe Lookup
# split Attached words

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#### Most frequent word analysis
# Combine row list into single string
convo_list = pd.DataFrame(convo_filt['ActNotes'].str.join(' '))
temp_list = convo_list['ActNotes'].str.cat(sep=' ')
    
words = temp_list.lower().split()
uniques = []
for word in words:
  if word not in uniques:
    uniques.append(word)

# Make a list of (count, unique) tuples.
counts = []
for unique in uniques:
  count = 0              # Initialize the count to zero.
  for word in words:     # Iterate over the words.
    if word == unique:   # Is this word equal to the current unique?
      count += 1         # If so, increment the count
  counts.append((count, unique))

counts.sort()            # Sorting the list puts the lowest counts first.
counts.reverse()         # Reverse it, putting the highest counts first.
# Print the ten words with the highest counts.
for i in range(min(10, len(counts))):
  count, word = counts[i]
  print('%s %d' % (word, count))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Performing Lemmatization
convo_list = convo_filt['ActNotes'].str.join(' ')
def clean(doc):
     stop_free = " ".join([i for i in doc.lower().split() if i not in StpWrds_Modlist])
     normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
     return normalized

doc_clean = [clean(doc).split() for doc in convo_list]

convo_filt['Notes_Lemm'] = doc_clean
convo_notes = convo_filt['Notes_Lemm'].str.join(' ')

# TfidfVectorizer converts colection of raw documents to a matrix of TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words=StpWrds_Modlist, min_df=1)
vec = vectorizer.fit_transform(convo_notes)

# define set of keywords for each category
# Compute TF-IDF matrix for input keywords
my_q = vectorizer.transform(
['ira s1 rollver ro roth inherited contributions distribution',
'spc smp windhaven tpi sip'])
# First string is keywords for Brokerage
# Second string is keywords for Managed Solutions

my_q.toarray().shape
## (2, 15846798) - 10% data

# Compute Cosine Similarity between keywords and Mars notes document
cs = cosine_similarity(my_q, vec)

# get top 500 records with high CS score for Brokerage
rs = pd.Series(cs[0]).sort_values(ascending=0)
top500 = rs.iloc[0:500]
text_1 = convo_filt.iloc[top500.index]

# get top 500 records with high CS score for Managed solutions
rs = pd.Series(cs[1]).sort_values(ascending=0)
top500_2 = rs.iloc[0:500]
text_2 = convo_filt.iloc[top500_2.index]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# computes and returns all possible token for given input string
def tokens(text):
    return re.findall('[a-z]+', text.lower())

# get top frequent terms in Brokerage 
temp = text_1['Notes_Lemm'].str.join(', ')
temp = ' '.join(temp)
top1_wrds = tokens(temp)
top1_count = Counter(top1_wrds)
top1_count = pd.DataFrame.from_dict(top1_count, orient = 'index')

# get top frequent terms in Managed solutions  
temp = text_1['Notes_Lemm'].str.join(', ')
temp = ' '.join(temp)
top2_wrds = tokens(temp)
top2_count = Counter(top2_wrds)
top2_count = pd.DataFrame.from_dict(top2_count, orient = 'index')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
rst = rs
rst = rst[rst == 0]
text_2_cs0 = convo_filt.iloc[rst.index]
