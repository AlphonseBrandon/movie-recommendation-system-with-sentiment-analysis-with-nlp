'''Class to preprocess metadata and extract 2017 movies'''
import pandas as pd
import numpy as np
import ast
class PreprocessMetadata:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.movies_up_to_2017 = pd.read_csv(self.metadata_path)
        self.movie = None
        self.movies_up_to_2016 = None

    def _evaluate_expression_nodes(self):
        '''evaluate expression nodes containing strings and python literals'''
        self.movies_up_to_2017['genres'] = self.movies_up_to_2017['genres'].map(lambda x: ast.literal_eval(x))
        self.movies_up_to_2017['cast'] = self.movies_up_to_2017['cast'].map(lambda x: ast.literal_eval(x))
        self.movies_up_to_2017['crew'] = self.movies_up_to_2017['crew'].map(lambda x: ast.literal_eval(x))

    def _make_genres_list(self):
        '''make a list of all the genres'''
        def make_genres_list(x):
            genres = []
            st = " "
            for i in x:
                if i.get('name') == 'Science Fiction':
                    scifi = 'Sci-Fi'
                    genres.append(scifi)
                else:
                    genres.append(i.get('name'))
            if genres == []:
                return np.nan
            else:
                return (st.join(genres))
        self.movies_up_to_2017['genres_list'] = self.movies_up_to_2017['genres'].map(lambda x: make_genres_list(x))

    def _get_principal_cast(self):
        '''get the list of principal cast'''
        def get_actor_1(cast):
            casts = []
            for actor in cast:
                casts.append(actor.get('name'))
            if casts == []:
                return np.nan
            else:
                return casts[0]
        self.movies_up_to_2017['actor_1_name'] = self.movies_up_to_2017['cast'].map(lambda x: get_actor_1(x))

        def actor_2(cast):
            casts = []
            for actor in cast:
                casts.append(actor.get('name'))
            if len(casts) <= 1:
                return np.nan
            else:
                return casts[1]
        self.movies_up_to_2017['actor_2_name'] = self.movies_up_to_2017['cast'].map(lambda x: actor_2(x))

        def actor_3(cast):
            casts = []
            for actor in cast:
                casts.append(actor.get('name'))
            if len(casts) <= 2:
                return np.nan
            else:
                return casts[2]
        self.movies_up_to_2017['actor_3_name'] = self.movies_up_to_2017['cast'].map(lambda x: actor_3(x))

    def _get_directors(self):
        '''get the list of directors'''
        def get_directors(crew):
            directors = []
            st = " "
            for i in crew:
                if i.get('job') == 'Director':
                    directors.append(i.get('name'))
            if directors == []:
                return np.nan
            else:
                return (st.join(directors))
        self.movies_up_to_2017['director_name'] = self.movies_up_to_2017['crew'].map(lambda x: get_directors(x))

    def _drop_unnecessary_columns(self):
        '''drop unnecessary columns'''
        self.movies_up_to_2017 = self.movies_up_to_2017.drop(columns=['homepage', 'overview', 'tagline', 'status', 'production_countries', 'spoken_languages', 'crew', 'cast', 'genres'])

    def _drop_duplicate_rows(self):
        '''drop duplicate rows'''
        self.movies_up_to_2017.drop_duplicates(subset ="movie_title", keep = 'last', inplace = True)

    def _make_new_column(self):
        '''make a new column'''
        self.movies_up_to_2017['comb'] = self.movies_up_to_2017['actor_1_name'] + ' ' + self.movies_up_to_2017['actor_2_name'] + ' '+ self.movies_up_to_2017['actor_3_name'] + ' '+ self.movies_up_to_2017['director_name'] +' ' + self.movies_up_to_2017['genres_list']

    def _save_to_csv(self):
        '''save to csv'''
        self.movies_up_to_2017.to_csv('D:/github-repos/movie-recommendation-system-with-sentiment-analysis-with-nlp/data/02_intermediate/movies_up_to_2017.csv', index=False)

    def preprocess(self):
        '''preprocess metadata'''
        self._evaluate_expression_nodes()
        self._make_genres_list()
        self._get_principal_cast()
        self._get_directors()
        self._drop_unnecessary_columns()
        self._drop_duplicate_rows()
        self._make_new_column()
        self._save_to_csv()

if __name__ == '__main__':
    metadata_path = 'D:/github-repos/movie-recommendation-system-with-sentiment-analysis-with-nlp/data/01_raw/movies_metadata.csv'
    preprocess_metadata = PreprocessMetadata(metadata_path)
    preprocess_metadata.preprocess()

'''The code above is a class that preprocesses the metadata. The class has a method preprocess() that calls all the other methods in the class. The class has a constructor that takes in the path to the metadata file. The constructor also reads the metadata file into a pandas dataframe and assigns it to the movies_up_to_2017 attribute. The movies_up_to_2017 attribute is a pandas dataframe that contains the metadata for all the movies up to 2017.'''