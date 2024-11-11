import pandas as pd
import joblib, nltk, re, pprint, string, numpy as np, statistics, json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

def create_resopnse_dataset():
    df = pd.read_csv('data/intent_responses.csv')
    joblib.dump(df, 'smalltalk/responses_df.joblib')
    return True

def create_smalltalk_dataset():
    smalltalk_df = pd.read_csv('data/smalltalk_intent.csv')
    smalltalk_df['Output'] = smalltalk_df['Intent'].str.split('_').str[2:].apply('_'.join)
    smalltalk_df['Intent'] = smalltalk_df['Intent'].str.split('_').str[1]
    smalltalk_df = smalltalk_df.rename(columns= {'Utterances':'Input'})
    smalltalk_df = smalltalk_df[['Input','Intent','Output']]
    smalltalk_df = smalltalk_df.drop_duplicates()
    joblib.dump(smalltalk_df, 'smalltalk/df.joblib')
    return True

def create_classifier(dataset):
    df = joblib.load(f'{dataset}/df.joblib')
    inputs = df['Input'].values
    labels = df['Intent'].values
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, stratify=labels, test_size=0.25, random_state=42)
    vect = TfidfVectorizer(sublinear_tf=True, use_idf=True, lowercase=True)
    X_train_tf = vect.fit_transform(X_train)
    clf = SVC(C=1.0, kernel='rbf').fit(X_train_tf, y_train)
    joblib.dump(clf, f'{dataset}/clf.joblib')
    joblib.dump(vect, f'{dataset}/clf_vect.joblib')
    return True

def create_intent_dfs(dataset):
    df = joblib.load(f'{dataset}/df.joblib')
    labels = set(df['Intent'])
    df_dict = {}
    for label in labels:
        df_dict[label] = df[df['Intent']==label]
    joblib.dump(df_dict, f'{dataset}/intent_df_dict.joblib')
    return True

def stemmer(doc):
    stem = PorterStemmer()
    analyzer = TfidfVectorizer(stop_words=stopwords.words('english')).build_analyzer()
    return [stem.stem(w) for w in analyzer(doc)]

def create_vectors(df):
    vect = TfidfVectorizer(use_idf=True, sublinear_tf=True, analyzer=stemmer, lowercase=True)
    docs = df['Input'].values
    return vect.fit_transform(docs), vect

def build_dt_matrices(dataset):
    df_dict = joblib.load(f'{dataset}/intent_df_dict.joblib')
    for df in df_dict.keys():
        vectors, vect = create_vectors(df_dict[df])
        joblib.dump(vectors, f"{dataset}/vectors/{df}_vectors.joblib")
        joblib.dump(vect, f"{dataset}/vectors/{df}_vectorizer.joblib")
    return True

def cosine_sim(intent, input, dataset):
    vect = joblib.load(f'{dataset}/vectors/{intent}_vectorizer.joblib')
    vectors = joblib.load(f'{dataset}/vectors/{intent}_vectors.joblib')
    return cosine_similarity(vect.transform([input]), vectors).flatten()

def match_output(intent, input, dataset):
    similarities = cosine_sim(intent, input, dataset)
    best_match = statistics.mode(similarities.argsort()[-3:][::-1])
    df_dict = joblib.load(f'{dataset}/intent_df_dict.joblib')
    intents = df_dict[intent]['Output'].values
    return intents[best_match]

def match_intent(input, dataset):
    clf = joblib.load(f'{dataset}/clf.joblib')
    return clf.predict(input)[0]

def response(input, dataset):
    vect = joblib.load(f'{dataset}/clf_vect.joblib')
    intent = match_intent(vect.transform([input]), dataset)
    output = match_output(intent, input, dataset)
    return (intent, output)

def find_response(input):
    output = response(input, 'smalltalk')
    responses = joblib.load('smalltalk/responses_df.joblib')
    output = responses[(responses['Intent']==output[0]) & (responses['Input']==output[1])]['Output']
    if len(output)!=0:
        return output.values[0]
    else: return None

###################################################################################

def create_intent_response_dataset():
    with open('data/intents.json','r') as file:
        data = json.load(file)['intents']
    norm_data=[]
    for tag in data:
        patterns = tag['patterns']
        responses = tag['responses']
        for pattern, response in zip(patterns, responses):
            norm_data.append({'Input':pattern, 'Intent':tag['tag'], 'Output':response})
    intents_df = pd.DataFrame(norm_data)
    joblib.dump(intents_df,'intent_response/df.joblib')
    return True

def smalltalk_init(dataset):
    if dataset=='smalltalk':
        create_smalltalk_dataset()
    elif dataset=='intent_response':
        create_intent_response_dataset()
    create_classifier(dataset)
    create_intent_dfs(dataset)
    build_dt_matrices(dataset)
    create_resopnse_dataset()

