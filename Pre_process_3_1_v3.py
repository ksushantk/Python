# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

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
# (12304,9) 1% data

# Sample 10% data 
data = pd.read_csv("ah170_2016_samp10pct.csv")
data.shape
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
# (10806, 9) - 1497 records are dropped -- 1% data
# (1055094, 9) - 148815 (12.36%) records are dropped -- 10% data

filt_data['Activity Type'] = filt_data['Activity Type'].replace(['LeadTaken', 'Lead Taken'], 'Lead_Taken')
filt_data['Activity Type'] = filt_data['Activity Type'].replace(['Event - Attended', 'Event - Client Cancelled', 
                                'Event - Enrolled', 'Event - Not Attended', 'Event - Schwab Cancelled', 'FA Delink - Review', 
                                'Interest List - Enrolled', 'Offer Cancelled', 'Offer Enrolled', 'Offer FollowUp', 
                                'Brochure Supplement', 'Core Offer Accepted', 'Core Offer Cancelled', 'Core Offer Denied', 
                                'Core Offer Enrolled', 'Data Capture', ], 'Offers_Events_Others')
filt_data['Activity Type'] = filt_data['Activity Type'].replace(['Call - Callback'], 'Outbound Call')
filt_data['Activity Type'] = filt_data['Activity Type'].replace(['MARS Mapped'], 'Left Message')

# Get average length and count for each activity type 
filt_data['Charlen'] = filt_data['Activity Notes'].apply(len)
temp = pd.DataFrame(filt_data['Activity Type'].unique())
avg_groups = pd.DataFrame(filt_data['Charlen'].
                          groupby(filt_data['Activity Type']).
                          agg(['mean', 'count']))
avg_groups.to_csv('output\Activity_Typ_charlen_2_27_v2.csv')
# filt_data.to_pickle('filtred_data_2_27.p')
# filt_data = pd.read_pickle('filtred_data_2_27.p')
###----------------------------------------------------------------------------

# Computing the number of characters
convo = pd.DataFrame(filt_data.iloc[:,(1,4)])
convo['Length'] = convo['Activity Notes'].apply(len)
plt.hist(convo['Length'])
# maximum number of characters are 1500

# Converting to lower case might result in some loss which could effect.
# Further process such as POS tagging [difficult to differentiate Proper nouns]
# further str.split() is not efficient approach - as it fails to seperate out
# punctuations or other symbols at end of word
# convo['ActNotesSplit'] = convo['Activity Notes'].str.lower().str.split()

# Tokenize the words using NLTK package
start_time = time.time()
convo['ActNotesSplit'] = convo['Activity Notes'].apply(lambda x: [item for item in word_tokenize(x)])
print("--- %s seconds ---" % (time.time() - start_time))

# Remove stop words and punctuation and other special characters
#customStopWords = set(stopwords.words("english")+list(punctuation))
# convo['ActNotes'] = convo['ActNotesSplit'].apply(lambda x: [item for item in x if item not in customStopWords])

# refining stopwords list manually based on visual inspection
#
StopWrds_list = pd.DataFrame(stopwords.words("english") + list(punctuation))
StopWrds_list.columns = ['Stopwords']
StopWrds_list.to_csv("output\StopWords.csv")

# Read stopwords after customizing it manually
StopWords_Mod = pd.read_csv('output\StopWords_Mod.csv')
StpWrds_Modlist = StopWords_Mod['Stopwords'].values.tolist()

# Do stop word filters and apply less than 2 tokens filter again. 
customStopWordsMod = set(StpWrds_Modlist)
convo['ActNotes'] = convo['ActNotesSplit'].apply(lambda x: [item for item in x if item not in customStopWordsMod])
convo.shape
## (10806, 4) - 1% data
## (1055094, 4) - 10% data
convo_filt = convo[convo.apply(lambda x: len(x['ActNotes'])>2, axis = 1)]
convo_filt.shape
## (10661, 4)
## (1039176, 4) Additional 15918 records are removed


###----------------------------------------------------------------------------
# find out n gramns
# Spelling correction - Levenshtein Distances, Dictionary Correction
# slang Look up
# Grammar check
# Apostrophe Lookup
# split Attached words
###----------------------------------------------------------------------------

# Performing Lemmatization
convo_list = convo_filt['ActNotes'].str.join(' ')
def clean(doc):
     stop_free = " ".join([i for i in doc.lower().split() if i not in StpWrds_Modlist])
     normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
     return normalized

doc_clean = [clean(doc).split() for doc in convo_list]

convo_filt['Notes_Lemm'] = doc_clean
convo_notes = convo_filt['Notes_Lemm'].str.join(' ')

# Store and read pickle files for faster execution
# convo_filt.to_pickle('filtred_moddata_3_1.p')
# convo_filt = pd.read_pickle('filtred_moddata_3_1.p')
convo_filt = convo_filt[['Activity Notes','Activity Type', 'Notes_Lemm']]

# TfidfVectorizer converts colection of raw documents to a matrix of TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words=StpWrds_Modlist, min_df=1)
vec = vectorizer.fit_transform(convo_notes)

# define set of keywords for each category
# Compute TF-IDF matrix for input keywords
my_q = vectorizer.transform(
['ira s1 rollver ro roth inherited contributions distribution rmd toa new open step move form online moved',
'spc smp windhaven tpi sip fee account acct strategy time allocation asset portfolio management review performance growth income managed manage manager'])
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
top500 = rs.iloc[0:500]
text_2 = convo_filt.iloc[top500.index]

#------------------------------------------------------------------------------
# Fetch CS score for different categories in vertical format
data = pd.DataFrame(cs).T
data.columns = ['catg1', 'catg2']

# For each record, assign category with highest CS score
data['class'] = data.idxmax(axis = 1)
# For records with ZERO CS score for all categories, assign to "others"
data.loc[(data['catg1']==0) & (data['catg2']==0), ['class']] = 'others'


# all_data = convo_filt.reset_index(drop=True)
all_data = pd.concat([convo_filt.reset_index(drop=True), data], axis = 1)
all_data = all_data[['Activity Notes','Activity Type', 'Notes_Lemm', 'class']]
mask = all_data['Activity Type'] == 'Offers_Events_Others'
all_data, others = all_data[~mask], all_data[mask]

# fetch 500 records from catg1 based on cosine scores
all_data = all_data.sort_values(by='catg1', ascending=0)
train_data =  all_data.iloc[:500,:]
all_data = all_data.iloc[500:]

# fetch 500 records from catg2 based on cosine scores
all_data = all_data.sort_values(by='catg2', ascending=0)
train_temp =  all_data.iloc[:500,:]
all_data = all_data.iloc[500:]
train_data = train_data.append(train_temp)

# fetch 500 records from others based on cosine scores
all_data = all_data.sort_values(by='class', ascending=0)
train_temp =  all_data.iloc[:500,:]
train_data = train_data.append(train_temp)
all_data = all_data.iloc[500:]

test_data = all_data
del(all_data, train_temp)

train_data.to_csv('output/training_2_28.csv')
test_data.to_csv('output/test_2_28.csv')

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# computes and returns all possible tokens for given input string
def tokens(text):
    return re.findall('[a-z]+', text.lower())

# Get top frequent terms in Brokerage 
temp = text_1['Notes_Lemm'].str.join(', ')
temp = ' '.join(temp)
top1_wrds = tokens(temp)
top1_count = Counter(top1_wrds)
top1_count = pd.DataFrame.from_dict(top1_count, orient = 'index')
top1_count.columns = ['count1']
top1_count['Words'] = top1_count.index

# Get top frequent terms in Managed solutions  
temp = text_2['Notes_Lemm'].str.join(', ')
temp = ' '.join(temp)
top2_wrds = tokens(temp)
top2_count = Counter(top2_wrds)
top2_count = pd.DataFrame.from_dict(top2_count, orient = 'index')
top2_count.columns = ['count2']
top2_count['Words'] = top2_count.index

# Compare frequency of common words between managed solutions and brokerage
test = pd.merge(top1_count, top2_count, how='inner', on = 'Words')
# Write files for reference
test.to_csv('output\common_wrds_freq_2_23.csv')
top1_count.to_csv('output\brokerage_top_wrds_freq_2_23.csv')
top2_count.to_csv('output\managedsoln_top_wrds_freq_2_23.csv')

del(convo, convo_list, convo_notes, filt_data)
#------------------------------------------------------------------------------
