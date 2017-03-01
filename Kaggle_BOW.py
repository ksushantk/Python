# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:48:27 2017

@author: sushant.kulkarni
"""

import os

print(os.getcwd())
os.chdir("C:/Users/sushant.kulkarni/Desktop/P01")
print(os.getcwd())


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd

train = pd.read_csv("output/training_2_28.csv", header=0)
test =  pd.read_csv("output/test_2_28.csv", header=0)
# retain only necessary rows
test = test[['Unnamed: 0', 'Activity Notes']]
test.columns = ['id', 'Activity Notes']

# Initialize an empty list to hold the clean notes
clean_train_reviews = []

# Loop over each note; create an index i that goes from 0 to the length
# of the MARS notes list

print ("Cleaning and parsing the training set Mars Notes...\n")
for i in range( 0, len(train["Activity Notes"])):
    clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["Activity Notes"][i], True)))

#  ****** Create a bag of words from the training set
# 
print ("Creating the bag of words...\n")


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 500) 


# fit_transform() does two functions: First, it fits the model 
# and learns the vocabulary; second, it transforms our training data 
# into feature vectors. The input to fit_transform should be a list of 
# strings. 

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# ******* Train a random forest using the bag of words
#
print ("Training the random forest (this may take a while)...")

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the category labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["class"] )



# Create an empty list and append the clean notes one by one
clean_test_reviews = []

print ("Cleaning and parsing the test set Mars Notes...\n")
for i in range(0,len(test["Activity Notes"])):
    clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["Activity Notes"][i], False)))

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
print ("Predicting test labels...\n")
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "Activity Notes":test["Activity Notes"], "class":result} )

# Use pandas to write the comma-separated output file
output.to_csv('output/Bag_of_Words_model.csv')
print ("Wrote results to Bag_of_Words_model.csv")

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------