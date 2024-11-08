import pandas as pd
import joblib, nltk, re, pprint, string, numpy as np, statistics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


def create_smalltalk_dataset():
    smalltalk_df = pd.read_csv('data/smalltalk_intent.csv')
    smalltalk_df['Intent'] = smalltalk_df['Intent'].str[10:]
    smalltalk_df['Subject'] = smalltalk_df['Intent'].str.split('_').str[0]
    smalltalk_df['Intent'] = smalltalk_df['Intent'].str.split('_').str[1:].apply('_'.join)
    smalltalk_df = smalltalk_df.rename(columns= {'Utterances':'Input'})
    smalltalk_df = smalltalk_df[['Input','Subject','Intent']]
    smalltalk_df = smalltalk_df.drop_duplicates()
    joblib.dump(smalltalk_df, 'smalltalk/df.joblib')
    return True

def create_subject_classifier():
    smalltalk_df = joblib.load('smalltalk/df.joblib')
    inputs = smalltalk_df['Input'].values
    labels = smalltalk_df['Subject'].values
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, stratify=labels, test_size=0.25, random_state=42)
    vect = TfidfVectorizer(sublinear_tf=True, use_idf=True)
    X_train_tf = vect.fit_transform(X_train)
    clf = SVC(C=1.0, kernel='rbf').fit(X_train_tf, y_train)
    joblib.dump(clf, 'smalltalk/subject_clf.joblib')
    joblib.dump(vect, 'smalltalk/classifier_vectorizer.joblib')
    return True

def create_intent_dfs():
    smalltalk_df = joblib.load('smalltalk/df.joblib')
    labels = set(smalltalk_df['Subject'])
    df_dict = {}
    for label in labels:
        df_dict[label] = smalltalk_df[smalltalk_df['Subject']==label]
    joblib.dump(df_dict, 'smalltalk/intent_df_dict.joblib')
    return True

def stemmer(doc):
    stem = PorterStemmer()
    analyzer = TfidfVectorizer(stop_words=stopwords.words('english')).build_analyzer()
    return [stem.stem(w) for w in analyzer(doc)]

def create_vectors(df):
    vect = TfidfVectorizer(use_idf=True, sublinear_tf=True, analyzer=stemmer)
    docs = df['Input'].values
    return vect.fit_transform(docs), vect

def build_dt_matrices():
    df_dict = joblib.load('smalltalk/intent_df_dict.joblib')
    for df in df_dict.keys():
        vectors, vect = create_vectors(df_dict[df])
        joblib.dump(vectors, f"smalltalk/vectors/{df}_vectors.joblib")
        joblib.dump(vect, f"smalltalk/vectors/{df}_vectorizer.joblib")
    return True

def __init__():
    create_smalltalk_dataset()
    create_subject_classifier()
    create_intent_dfs()
    build_dt_matrices()


def cosine_sim(subject, input):
    vect = joblib.load(f'smalltalk/vectors/{subject}_vectorizer.joblib')
    vectors = joblib.load(f'smalltalk/vectors/{subject}_vectors.joblib')
    return cosine_similarity(vect.transform([input]), vectors).flatten()

def match_intent(subject, input):
    similarities = cosine_sim(subject, input)
    best_match = statistics.mode(similarities.argsort()[-3:][::-1])
    df_dict = joblib.load('smalltalk/intent_df_dict.joblib')
    intents = df_dict[subject]['Intent'].values
    return intents[best_match]

def match_subject(input):
    clf = joblib.load('smalltalk/subject_clf.joblib')
    return clf.predict(input)[0]

def print_subject_intent(input):
    vect = joblib.load('smalltalk/classifier_vectorizer.joblib')
    subject = match_subject(vect.transform([input]))
    intent = match_intent(subject, input)
    print(subject, intent)