import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import nltk, re, pprint, string
from nltk import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from smalltalk import *

def stemmer_sw(doc):
    stem = PorterStemmer()
    analyzer = TfidfVectorizer(stop_words=stopwords.words('english')).build_analyzer()
    return [stem.stem(w) for w in analyzer(doc)]

def create_dt_matrix(filename):
    df = joblib.load(f'dfs/{filename}.joblib')
    vect = TfidfVectorizer(use_idf=True, sublinear_tf=True, analyzer=stemmer_sw, lowercase=True)
    matrix = vect.fit_transform(df['Answer'].values)
    joblib.dump(matrix,f'dtm/{filename}_a.joblib')
    joblib.dump(vect,f'vects/{filename}_a.joblib')
    matrix = vect.fit_transform(df['Question'].values)
    joblib.dump(matrix,f'dtm/{filename}_q.joblib')
    joblib.dump(vect,f'vects/{filename}_q.joblib')

def cosine_sim_answer(input, filename):
    vect = joblib.load(f'vects/{filename}_a.joblib')
    dtm = joblib.load(f'dtm/{filename}_a.joblib')
    df = joblib.load(f'dfs/{filename}.joblib')
    similarities = cosine_similarity(vect.transform([input]), dtm).flatten()
    match = similarities.argsort()[-1:]
    return df.iloc[match]['Answer'].values[0]

def cosine_sim_question(input,filename):
    vect = joblib.load(f'vects/{filename}_q.joblib')
    dtm = joblib.load(f'dtm/{filename}_q.joblib')
    df = joblib.load(f'dfs/{filename}.joblib')
    similarities = cosine_similarity(vect.transform([input]), dtm).flatten()
    match = similarities.argsort()[-1:]
    return df.iloc[match]['Question'].values[0]

def qanda_init():
    for filename in ['qanda']:
        create_df(filename)
        create_dt_matrix(filename)