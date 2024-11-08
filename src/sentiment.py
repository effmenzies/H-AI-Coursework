import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


def create_classifier():
    cats = ['pos','neg']
    data = []
    labels = []
    for cat in cats:
        filepath = 'data/movie_reviews'
        filepath += f'/{cat}'
        for file in os.listdir(filepath):
            new_path = filepath + f'/{file}'
            with open(new_path, encoding='utf8', errors='ignore', mode='r') as review:
                data.append(review.read())
                labels.append(cat)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.25, random_state=42)
    vect = TfidfVectorizer(sublinear_tf=True, use_idf=True, lowercase=True)
    X_train_tf = vect.fit_transform(X_train)
    clf = SVC(C=1.0, kernel='rbf').fit(X_train_tf, y_train)
