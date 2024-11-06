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

stem = PorterStemmer()
analyzer = TfidfVectorizer(stop_words=stopwords.words('english')).build_analyzer()
punctuation = string.punctuation
dataset = pd.read_csv('Coursework/data/QAdataset.csv')
joblib.dump(dataset,'qanda_df.joblib')
questions = dataset['Question'].values

def stemmer(doc):
    return (stem.stem(w) for w in analyzer(doc))

stem_vectorizer = TfidfVectorizer(analyzer=stemmer)

def build_dt_matrix(documents):
    vectors = stem_vectorizer.fit_transform(documents)
    joblib.dump(vectors, 'qanda_vectors.joblib')
    return

def transform_input(input):
    return stem_vectorizer.transform(input)

def q_similarity(query):
    vectors = joblib.load('qanda_vectors.joblib')
    return cosine_similarity(query, vectors).flatten()

def qanda_search(query):
    q = transform_input([query]).toarray()
    similarities = q_similarity(q)
    index = np.argmax(similarities)
    qanda_df = joblib.load('qanda_df.joblib')
    return qanda_df['Answer'].iloc[index]