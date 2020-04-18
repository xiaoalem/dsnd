import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.externals import joblib

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

for i in ("stopwords", "punkt", "wordnet"):
    nltk.download(i)


def load_data(database_filepath):
    '''
    Load data from SQLite database file.
    
    Parameters:
        database_filepath: path of SQLite database file.
    
    Returns:
        X: Training data features
        Y: Training data labels
        category_names: Names of training data labels
    '''
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize input text.
    
    Parameters:
        text: input text
    
    Returns:
        tokens: tokens from the text
    '''
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t).lower().strip() for t in tokens]
    return tokens

    
def build_model():
    '''
    Build ML model to classify messages. Training not done yet.
    
    Returns:
        The result model (before training is done).
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])

    param_grid = dict(
        tfidf__smooth_idf=[True, False],
        clf__estimator__estimator__C=[1, 2],
    )
    cv = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates a trainned model.
    Prints evaluation report per category.
    
    Parameters:
        model: A trained model for classifying messages.
        X_test: features of test dataset
        Y_test: labels of test dataset
        category_names: names of labels
    '''
    YHat = model.predict(X_test)
    metrics = [
        [
            m(Y_test.iloc[:, i].values, YHat[:, i], average="micro")
            for m in (f1_score, precision_score, recall_score)
        ]
        for i in range(len(Y_test.columns))
    ]
    report = pd.DataFrame(metrics, columns=["F1Score", "Precision", "Recall"], index=Y_test.columns)
    pd.set_option('display.max_rows', None)
    print(report)


def save_model(model, model_filepath):
    '''
    Saves trained model to file.
    
    Parameters:
        model: A trained model for classifying messages.
        model_filepath: path to save
    '''
    
    joblib.dump(model, model_filepath)



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
        
        print('Load model from file again and re-evaluate')
        model = None
        model = joblib.load(model_filepath)
        evaluate_model(model, X_test, Y_test, category_names)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()