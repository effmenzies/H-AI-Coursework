import bs4 as bs, joblib
from bs4 import BeautifulSoup as bsoup
import nltk, string
from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np


'''
punctuation = string.punctuation.replace('.','')
with open('data/dialogs.txt','r',encoding='utf8') as f:
    text = ""
    for line in f:
        line1 = line.replace("\n"," ")
        line2 = "".join([char for char in line1 if char not in punctuation])
        text+=line2
    
    sents = nltk.sent_tokenize(text)
    tokenized_text=[]
    for s in sents:
        s=s.lower()
        sequence = word_tokenize(s)
        for word in sequence:
            if word==".":
                sequence.remove(word)
        tokenized_text.append(sequence)
'''

def create_df():
    with open('data/dialogs.txt','r',encoding='utf8') as f:
        data=[]
        for line in f:
            line = line.replace('\n','')
            split = line.split(' ')
            if split[-1]=='':
                line = ' '.join(split[:-1])
            else: line = ' '.join(split)
            data.append([line,'smalltalk'])

    df = pd.read_csv('data/qanda.csv')
    questions = df['Question'].values
    for q in questions:
        processed=" ".join([word.lower() for word in word_tokenize(q) if word not in string.punctuation])
        data.append([processed,'qanda'])

    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    df.columns=['Utterance','Intent']
    joblib.dump(df,'dfs/intent.joblib')
    return True

def create_classifier():
    df = joblib.load('dfs/intent.joblib')
    vect = TfidfVectorizer(sublinear_tf=True, use_idf=True, lowercase=True)
    data = df['Utterance'].values
    labels = df['Intent'].values
    X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.3, random_state=3)
    X_train_tf = vect.fit_transform(X_train)
    clf = SVC(C=0.3, kernel='rbf').fit(X_train_tf, y_train)
    joblib.dump(clf,'clf/clf.joblib')
    joblib.dump(vect,'vects/clf.joblib')
    return True

def clf_init():
    create_df()
    create_classifier()

def classify(input):
    clf= joblib.load('clf/clf.joblib')
    vect=joblib.load('vects/clf.joblib')
    return clf.predict(vect.transform([input]))[0]
