# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:31:10 2023

@author: coope
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin


# NLTK Packages
import nltk
#nltk.download( 'vader_lexicon' )
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Spacy Packages
import spacy
#from spacy.lang.en.stop_words import STOP_WORDS


# Models used for classification
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

# Feature Extractor for text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#import skipthoughts


input_path_nominal = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\Raw Data\\Yelp Data\\nominal.csv'
input_path_binary = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\Raw Data\\Yelp Data\\binary.csv'
path = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\yelp_dataset\\total_data.csv'

df = pd.read_csv(path, index_col = 0)
#df = pd.read_csv(input_path_binary)
#df = pd.read_csv(input_path_nominal)

"""
pos = df[df["nominal"] == "Positive"].sample(543108)
neg = df[df["nominal"] == "Negative"].sample(543108)
neut = df[df["nominal"] == "Neutral"]
df = pd.concat([pos, neg, neut])
"""

df_sample = df.sample(20000, random_state = 12345)

#pd.crosstab(index = df["nominal"], columns = "prop")

X_train, X_test, y_train, y_test = train_test_split( df_sample["text"], df_sample["stars"], test_size = 0.30, random_state = 12345)
#X_train, X_test, y_train, y_test = train_test_split( df_sample["text"], df_sample["nominal"], test_size = 0.30, random_state = 12345)

plt.hist(df["nominal"])
plt.show() 

plt.hist(df_sample["nominal"])
plt.show() 

plt.hist(y_train)
plt.show() 

plt.hist(y_test)
plt.show() 

## Word Representation

## N-grams (Phrase)

## Part of Speech tagging - enrich words by categorizing as nouns, verbs, etc

## Tokenization using spaCy




## Stopwords - NLTK





## Stopwords - spacy
nlp = spacy.load("en_core_web_sm")

text = "This is a sample sentence with some stop words"
doc = nlp(text)

filtered_tokens = [token.text for token in doc if not token.is_stop]

print(filtered_tokens)



"""
stopwords = list(STOP_WORDS)

for word in stopwords:
    if word.is_stop == False and not word.is_punct:
"""


## StopWords
"a about above after again against all am an and any are as at be because been before being below between both but by can did do does doing don down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own s same she should so some such t than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with you your yours yourself yourselves"




## Lemmatizing



## Bag of words - count
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1, 2))
# strip_accents = "unicode"
# stop_words = "english"  # or could pass a list of stopwords
# binary = True


## Bag of words - TF-IDF
vectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1, 2))


## Word Embeddings







# TF-IDF bag of n-grams pipeline and logistic regression
pipeline_tfidf_LOG = Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1, 2))),
                        ('classifier', LogisticRegression())])


pipeline_tfidf_LOG.fit(X_train, y_train)
pipeline_tfidf_LOG.score(X_test, y_test) # 62.017 %


"""
# TF-IDF bag of n-grams pipeline and support vector machine
pipeline_tfidf_SVM = Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1, 2))),
                        ('classifier', svm.SVC())])


pipeline_tfidf_SVM.fit(X_train, y_train)
pipeline_tfidf_SVM.score(X_test, y_test) # 60.5 %
"""

# TF-IDF bag of n-grams pipeline and support vector machine (Linear)
pipeline_tfidf_SVM_L = Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1, 2))),
                        ('classifier', svm.LinearSVC())])


pipeline_tfidf_SVM_L.fit(X_train, y_train)
pipeline_tfidf_SVM_L.score(X_test, y_test) # 63.067%


# TF-IDF bag of n-grams pipeline and Multinomial Naive Bayes
pipeline_tfidf_MNB = Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1, 2))),
                        ('classifier', MultinomialNB())])


pipeline_tfidf_MNB.fit(X_train, y_train)
pipeline_tfidf_MNB.score(X_test, y_test) # 42.883%


# TF-IDF bag of n-grams pipeline and Complement Naive Bayes
pipeline_tfidf_CNB = Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1, 2))),
                        ('classifier', ComplementNB())])


pipeline_tfidf_CNB.fit(X_train, y_train)
pipeline_tfidf_CNB.score(X_test, y_test) # 45.467%













## Baseline

# Sentiment score via SpaCy

for i in range(len(X_test)):
    

# Sentiment Score via Vadar
sentiment = SentimentIntensityAnalyzer()
score = sentiment.polarity_scores( X_test[27361] )
print( score )








# SkipThoughts pipeline
pipeline_skipthought = Pipeline(steps = [('vectorizer', SkipThoughtsVectorizer()),
                        ('classifier', LogisticRegression())])

# Combination of SkipThoughts and TF-IDF pipeline
feature_union = ('feature_union', FeatureUnion([
    ('skipthought', SkipThoughtsVectorizer()),
    ('tfidf', TfidfVectorizer(ngram_range = (1, 2))),]))

pipeline_both = Pipeline(steps = [feature_union,
                        ('classifier', LogisticRegression())])



for train_size in (20, 50, 100, 200, 500, 1000, 2000, 3000, len(tweets_train)):
    print(train_size, '--------------------------------------')
    # skipthought
    pipeline_skipthought.fit(tweets_train[:train_size], classes_train[:train_size])
    print ('skipthought', pipeline_skipthought.score(tweets_test, classes_test))

    # tfidf
    pipeline_tfidf.fit(tweets_train[:train_size], classes_train[:train_size])
    print('tfidf', pipeline_tfidf.score(tweets_test, classes_test))

    # both
    pipeline_both.fit(tweets_train[:train_size], classes_train[:train_size])
    print('skipthought+tfidf', pipeline_both.score(tweets_test, classes_test))