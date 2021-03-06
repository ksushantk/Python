# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:27:45 2017

@author: sushant.kulkarni
"""

import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        if remove_stopwords:
            review_text = BeautifulSoup(review, "lxml").get_text()
        else:
            review_text = review
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a note into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
    
    def get_bigrams(myString):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(myString)
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)
    result = [' '.join([(w).lower() for w in x.split()]) for x in tokens]
    return result

    doc = 'my name is sushant kulkarni. i work at tiger analytics. am interested in mutual funds'
    doc = re.sub(r'[#@%[\]\(\)\!`~$^&.*]', '', doc)
    find = ['sushant', 'sushant kulkarni', 'tiger analytics', 'mutual funds', 'interested']
    bi_grams = get_bigrams(doc)
    trigram = ngrams(doc.split(), 3)
    trigrams = [ ' '.join(grams) for grams in trigram]
    n_gram = bi_grams + trigrams
    cnt = ".".join([i for i in find if i in n_gram])
    test = cnt.split(".")
