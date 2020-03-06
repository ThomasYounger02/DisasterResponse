import sys
import numpy as np
import pandas as pd
import re
import sqlite3
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''load the data from a database

    Args:
        database_filepath(str): the path to the database.

    Return:
        X(Series): the message.
        X(dataframe): the categories information.
        category_names(list): the category names list.
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterMessage', engine)
    X = df['message']
    category_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    Y = df[category_names]
    return X, Y, category_names

def tokenize(text):
    '''tokenize text message, including deleting stop words, normalization, and lemmatizing.

    Args:
        text(str): the text to deal with.

    Return:
        clean_tokens(list): the cleaned tokens list
    '''
    stop_words = stopwords.words('english')
    #normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # token
    tokens = word_tokenize(text) 
    #lem
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return clean_tokens


def build_model():
    '''build a pipe line to organise the machine learning model training process.

    Args:
        None

    Return:
        cv(model): the model instance.
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''print the model performence information including: precison, recall, f1_score.

    Args:
        model(model): the trained model.
        X_test(Series): the X test part data.
        Y_test(dataframe): the Y test part data.
        category_names(list): the whole of category names.

    Return:
        None
    '''
    pred = model.predict(X_test)
    # classification report
    print(classification_report(Y_test, y_pred, target_names=category_names,digits=4))


def save_model(model, model_filepath):
    '''save the model information to a pkl file.

    Args:
        model(model): the trained model.
        model_filepath(str): the path to store the pkl file.

    Return:
        None
    '''
    with open(model_filepath, 'wb') as pkl_file:
        pickle.dump(model, pkl_file)
    pkl_file.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()