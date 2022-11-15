import pandas as pd
from tmdbv3api import TMDb
import json
import requests
tmdb = TMDb()
tmdb.api_key = '663ff3650c35de4553dbd25b8eb2de0d'
from tmdbv3api import Movie
tmdb_movie = Movie()
import numpy as np

global movie_year
global movies_of_
global save_data_path
movie_year = 2020
movies_of_ = 'movies_of_{movie_year}'
save_data_path = 'D:/github-repos/movie-recommendation-system-with-sentiment-analysis-with-nlp/data/02_intermediate'


class datapipeline:
    def __init__(self):
        pass

    def scrap_wikipedia(self):
        '''This function scrapes wikipedia for movie data'''
        
        global link
        link = "https://en.wikipedia.org/wiki/List_of_American_films_of_{}".format(movie_year)
        df1 = pd.read_html(link, header=0)[2]
        df2 = pd.read_html(link, header=0)[3]
        df3 = pd.read_html(link, header=0)[4]
        df4 = pd.read_html(link, header=0)[5]
        df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        global data
        data = df

    def get_genre(self, x):
        '''This function returns the genre of the movie from the imdb link
        :param x: movie title from scraped data'''

        genres = []
        result = tmdb_movie.search(x)
        movie_id = result[0].id
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        data_json = response.json()
        if data_json['genres']:
            genre_str = " " 
            for i in range(0,len(data_json['genres'])):
                genres.append(data_json['genres'][i]['name'])
            return genre_str.join(genres)
        else:
            np.NaN

    def make_genres_column(self):
        data['genres'] = data['Title'].map(lambda x: self.get_genre(str(x)))

    def extract_features(self):
        movies_of_ = data[['Title','Cast and crew','genres']]


    def get_director(self, x):
        if " (director)" in x:
            return x.split(" (director)")[0]
        elif " (directors)" in x:
            return x.split(" (directors)")[0]
        else:
            return x.split(" (director/screenplay)")[0]

    def make_directors_column(self):
        movies_of_['director_name'] = movies_of_['Cast and crew'].map(lambda x: self.get_director(x))
    
    def get_actor1(self, x):
        return ((x.split("screenplay); ")[-1]).split(", ")[0])

    def make_actor_1_column(self):
        movies_of_['actor_1_name'] = movies_of_['Cast and crew'].map(lambda x: self.get_actor1(x))

    def get_actor2(self, x):
        if len((x.split("screenplay); ")[-1]).split(", ")) < 2:
            return np.NaN
        else:
            return ((x.split("screenplay); ")[-1]).split(", ")[1])

    def make_actor_2_column(self):
        movies_of_['actor_2_name'] = movies_of_['Cast and crew'].map(lambda x: self.get_actor2(x))
        
    def get_actor3(self, x):
        if len((x.split("screenplay); ")[-1]).split(", ")) < 3:
            return np.NaN
        else:
            return ((x.split("screenplay); ")[-1]).split(", ")[2])

    def make_actor_3_column(self):
        movies_of_['actor_3_name'] = movies_of_['Cast and crew'].map(lambda x: self.get_actor3(x))       

    def rename_title(self):
        global movies_of_
        movies_of_ = movies_of_.rename(columns={'Title':'movie_title'})

    def fill_missing_values(self):
        movies_of_['actor_2_name'] = movies_of_['actor_2_name'].replace(np.nan, 'unknown')
        movies_of_['actor_3_name'] = movies_of_['actor_3_name'].replace(np.nan, 'unknown')

    def make_titles_lower_case(self):
        movies_of_['movie_title'] = movies_of_['movie_title'].str.lower() 

    def combine_features_in_one_column(self):
        movies_of_['comb'] = movies_of_['actor_1_name'] + ' ' + movies_of_['actor_2_name'] + ' '+ movies_of_['actor_3_name'] + ' '+ movies_of_['director_name'] +' ' + movies_of_['genres']

    def get_features(self):
        global movies_of_
        movies_of_ = movies_of_.loc[:,['director_name','actor_1_name','actor_2_name','actor_3_name','genres','movie_title']]

    def save_data_to_file(self):
        global save_data_path
        movies_of_.to_csv('{save_data_path}/{movies_of}.csv'.format(save_data_path, movies_of_), index=False)       

    def preprocess(self):
        self.scrap_wikipedia()
        self.make_genres_column()
        self.extract_features()
        self.make_directors_column()
        self.make_actor_1_column()
        self.make_actor_2_column()
        self.make_actor_3_column()
        self.rename_title()
        self.fill_missing_values()
        self.make_titles_lower_case()
        self.combine_features_in_one_column()
        self.get_features()
        self.save_data_to_file()

data = datapipeline()
data.preprocess()
