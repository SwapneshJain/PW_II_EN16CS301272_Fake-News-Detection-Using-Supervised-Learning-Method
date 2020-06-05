# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:35:52 2020

@author: Swapnesh Jain
"""

#loading packages and libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
#import pickle

#creating instance of stemmer and stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
porter = PorterStemmer()

#function to preprocess data
#removing special characters
#using PorterStemmer from NLTK library to stemmatize each word
def dataprep(feature):
    col1 = list(feature)
    replace_schar = [re.sub(r'[^a-z0-9]',' ',line.lower()) for line in col1]
    new_col = []
    for line in replace_schar:
        temp = ""
        for j in line.split():
            if j not in stopwords:
                temp = temp + porter.stem(j) + " "
        new_col.append(temp)
    return new_col


#reading the data files
train_news = pd.read_csv('train.csv')
train_news['Statement'].replace(' ', np.nan, inplace=True)
train_news['Statement'].replace('  ', np.nan, inplace=True)
train_news = train_news.dropna()

train_label = [0 if line == False else 1 for line in train_news['Label']]

                                              
valid_news = pd.read_csv('valid.csv')
valid_news['Statement'].replace(' ', np.nan, inplace=True)
valid_news['Statement'].replace('  ', np.nan, inplace=True)
valid_news = valid_news.dropna()

valid_label = [0 if line == 'FALSE' else 1 for line in valid_news['Label']]

#calling dataprep function to our data files
train_statement = dataprep(train_news['Statement'])
valid_statement = dataprep(valid_news['Statement'])

#converting text data into token count
countvector = CountVectorizer()
train_news_count = countvector.fit_transform(train_statement)
valid_news_count = countvector.transform(valid_statement)


#converting text data to tf-idf matrix
tfidfvector = TfidfVectorizer()
train_tfidf = tfidfvector.fit_transform(train_statement)
valid_tfidf = tfidfvector.transform(valid_statement)


#first we will use bag of words to create models
#building classifier using naive bayes
classifier = MultinomialNB()
classifier.fit(train_news_count, train_label)

predictions = classifier.predict(valid_news_count)
accuracy_score(valid_label, predictions)
confusion_matrix(valid_label, predictions)

#building classifier using logistic regression
classifier = LogisticRegression()
classifier.fit(train_news_count, train_label)

predictions = classifier.predict(valid_news_count)
accuracy_score(valid_label, predictions)
confusion_matrix(valid_label, predictions)

#building classifier using random forest
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(train_news_count, train_label)

predictions = classifier.predict(valid_news_count)
accuracy_score(valid_label, predictions)
confusion_matrix(valid_label, predictions)

#building classifier using xgboost (extreme gradient boosting)
classifier = XGBClassifier()
classifier.fit(train_news_count, train_label)

predictions = classifier.predict(valid_news_count)
accuracy_score(valid_label, predictions)
confusion_matrix(valid_label, predictions)


#now we will use tf-idf vector to create models
#building classifier using naive bayes
classifier = MultinomialNB()
classifier.fit(train_tfidf, train_label)

predictions = classifier.predict(valid_tfidf)
accuracy_score(valid_label, predictions)
confusion_matrix(valid_label, predictions)

#building classifier using logistic regression
classifier = LogisticRegression()
classifier.fit(train_tfidf, train_label)

predictions = classifier.predict(valid_tfidf)
accuracy_score(valid_label, predictions)
confusion_matrix(valid_label, predictions)

#building classifier using random forest
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(train_tfidf, train_label)

predictions = classifier.predict(valid_tfidf)
accuracy_score(valid_label, predictions)
confusion_matrix(valid_label, predictions)

#building classifier using xgboost (extreme gradient boosting)
classifier = XGBClassifier()
classifier.fit(train_tfidf, train_label)

predictions = classifier.predict(valid_tfidf)
accuracy_score(valid_label, predictions)
confusion_matrix(valid_label, predictions)


'''#saving best model to the disk
model_name = 'finalmodel.sav'
pickle.dump(classifier, open(model_name, 'wb'))'''