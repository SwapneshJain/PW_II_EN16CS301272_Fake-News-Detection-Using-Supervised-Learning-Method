# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 01:43:25 2020

@author: Swapnesh Jain
"""

#loading packages and libraries
import pandas as pd
import numpy as np
import IncompleteNews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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
import pickle


#reading the data files
train_news = pd.read_csv('train.csv')
train_news = train_news.drop(['id', 'title', 'author'], axis = 1)
train_news['text'].replace(' ', np.nan, inplace=True)
train_news['text'].replace('  ', np.nan, inplace=True)
train_news = train_news.dropna()

#splitting dataset into train-test set with test size of 20%
X_train, X_test, y_train, y_test = train_test_split(train_news['text'], train_news['label'], 
                                                    test_size=0.20, shuffle = False, stratify = None)


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

#calling dataprep function to our data files
new_X_train = dataprep(X_train)
new_X_test = dataprep(X_test)

y_train = list(y_train)
y_test = list(y_test)

train_feature = pd.DataFrame(new_X_train, columns = ['text'])
train_target = pd.DataFrame(y_train, columns = ['label'])

train_feature = train_feature.drop(IncompleteNews.train_split)
train_target = train_target.drop(IncompleteNews.train_split)

test_feature = pd.DataFrame(new_X_test, columns = ['text'])
test_target = pd.DataFrame(y_test, columns = ['label'])

test_feature = test_feature.drop(IncompleteNews.test_split)
test_target = test_target.drop(IncompleteNews.test_split)

#converting text data into token count
countvector = CountVectorizer()
train_news_count = countvector.fit_transform(train_feature['text'])
test_news_count = countvector.transform(test_feature['text'])

#converting text data to tf-idf matrix
tfidfvector = TfidfVectorizer()
train_tfidf = tfidfvector.fit_transform(train_feature['text'])
test_tfidf = tfidfvector.transform(test_feature['text'])

#converting text data to tf-idf (with bi-gram) matrix
tfidfvector = TfidfVectorizer(ngram_range = (2,2))
train_bigram = tfidfvector.fit_transform(train_feature['text'])
test_bigram = tfidfvector.transform(test_feature['text'])

#converting text data to tf-idf (with tri-gram) matrix
tfidfvector = TfidfVectorizer(ngram_range = (3,3))
train_trigram = tfidfvector.fit_transform(train_feature['text'])
test_trigram = tfidfvector.transform(test_feature['text'])



#first we will use bag of words to create models
#building classifier using naive bayes
classifierNB = MultinomialNB()
classifierNB.fit(train_news_count, train_target['label'])

predictions = classifierNB.predict(test_news_count)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using logistic regression
classifierLR = LogisticRegression()
classifierLR.fit(train_news_count, train_target['label'])

predictions = classifierLR.predict(test_news_count)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using random forest
classifierRF = RandomForestClassifier(n_estimators = 100)
classifierRF.fit(train_news_count, train_target['label'])

predictions = classifierRF.predict(test_news_count)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using xgboost (extreme gradient boosting)
classifierXG = XGBClassifier()
classifierXG.fit(train_news_count, train_target['label'])

predictions = classifierXG.predict(test_news_count)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)



#now we will use tf-idf vector to create models
#building classifier using naive bayes
classifierNB = MultinomialNB()
classifierNB.fit(train_tfidf, train_target['label'])

predictions = classifierNB.predict(test_tfidf)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using logistic regression
classifierLR = LogisticRegression()
classifierLR.fit(train_tfidf, train_target['label'])

predictions = classifierLR.predict(test_tfidf)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using random forest
classifierRF = RandomForestClassifier(n_estimators = 100)
classifierRF.fit(train_tfidf, train_target['label'])

predictions = classifierRF.predict(test_tfidf)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using xgboost (extreme gradient boosting)
classifierXG = XGBClassifier()
classifierXG.fit(train_tfidf, train_target['label'])

predictions = classifierXG.predict(test_tfidf)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)



#now we will use bigram to create models
#building classifier using naive bayes
classifierNB = MultinomialNB()
classifierNB.fit(train_bigram, train_target['label'])

predictions = classifierNB.predict(test_bigram)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using logistic regression
classifierLR = LogisticRegression()
classifierLR.fit(train_bigram, train_target['label'])

predictions = classifierLR.predict(test_bigram)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using random forest
classifierRF = RandomForestClassifier(n_estimators = 100)
classifierRF.fit(train_bigram, train_target['label'])

predictions = classifierRF.predict(test_bigram)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using xgboost (extreme gradient boosting)
classifierXG = XGBClassifier()
classifierXG.fit(train_bigram, train_target['label'])

predictions = classifierXG.predict(test_bigram)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)



#now we will use trigram to create models
#building classifier using naive bayes
classifierNB = MultinomialNB()
classifierNB.fit(train_trigram, train_target['label'])

predictions = classifierNB.predict(test_trigram)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using logistic regression
classifierLR = LogisticRegression()
classifierLR.fit(train_trigram, train_target['label'])

predictions = classifierLR.predict(test_trigram)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using random forest
classifierRF = RandomForestClassifier(n_estimators = 100)
classifierRF.fit(train_trigram, train_target['label'])

predictions = classifierRF.predict(test_trigram)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)

#building classifier using xgboost (extreme gradient boosting)
classifierXG = XGBClassifier()
classifierXG.fit(train_trigram, train_target['label'])

predictions = classifierXG.predict(test_trigram)
accuracy_score(test_target['label'], predictions)
confusion_matrix(test_target['label'], predictions)


#saving best model to the disk
model_name = 'finalmodel.sav'
pickle.dump(classifierXG, open(model_name, 'wb'))