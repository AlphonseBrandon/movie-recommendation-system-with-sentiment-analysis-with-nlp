def clean_movie_meta_data(file_name):
    '''Function to clean meta data
    :param file_name: path to the file'''
    import pandas as pd
    import numpy as np
    data = pd.read_csv(file_name)
    data['actor_1_name'] = data['actor_1_name'].replace(np.nan, 'unknown')
    data['actor_2_name'] = data['actor_2_name'].replace(np.nan, 'unknown')
    data['actor_3_name'] = data['actor_3_name'].replace(np.nan, 'unknown')
    data['director_name'] = data['director_name'].replace(np.nan, 'unknown')
    data['genres'] = data['genres'].str.replace('|', ' ')
    data['movie_title'] = data['movie_title'].str.lower()
    data['movie_title'] = data['movie_title'].str[:-1]
    return data