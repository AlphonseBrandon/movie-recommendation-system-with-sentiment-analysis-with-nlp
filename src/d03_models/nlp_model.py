import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import naive_bayes
import pickle

class modeling:
    def __init__(self):
        global path_to_data
        global save_the_mode_path
        save_the_mode_path = 'D:/github-repos/movie-recommendation-system-with-sentiment-analysis-with-nlp/notebooks/05_reports/nlp_model.pkl'
        path_to_data = 'D:/github-repos/movie-recommendation-system-with-sentiment-analysis-with-nlp/data/01_raw/reviews.txt'

    def download_stopwords(self):
        nltk.download('stopwords')

    def load_data(self):
        global reviews_data
        reviews_data = pd.read_csv(path_to_data, sep='\t', names=['Reviews', 'Comments'])

    def set_language(self):
        global stopset
        stopset = set(stopwords.words('english'))

    def make_vectorizer(self):
        global vectorizer
        vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

    def load_the_features(self):
        global X, y
        X = vectorizer.fit_transform(reviews_data.Comments)
        y = reviews_data.Reviews
        pickle.dump(vectorizer, open('transform', 'wb'))

    def split_the_data(self):
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

    def teach_the_model(self):
        global clf
        clf = naive_bayes.MultinomialNB()
        clf.fit(X_train, y_train)   

    def get_the_model_accuracy(self):
        pred = clf.predict(X_test)
        accuracy_score(y_test, pred)*100 

    def safe_the_model(self):
        global save_the_mode_path
        pickle.dump(clf, open(save_the_mode_path, 'wb'))   

    def model_pipeline(self):
        self.download_stopwords()
        self.load_data()
        self.set_language()
        self.make_vectorizer()
        self.load_the_features()
        self.split_the_data()
        self.teach_the_model()
        self.get_the_model_accuracy()
        self.safe_the_model()
        
nlp_model = modeling()
nlp_model.model_pipeline()