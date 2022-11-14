
def number_of_movies_per_column_name (column_name):
    """"This function plots the number of movies per column name"""
    import matplotlib.pyplot as plt

    import sys
    sys.path.insert(1, 'D:/github-repos/movie-recommendation-system-with-sentiment-analysis-with-nlp/src/d01_utils')
    sys.path.insert(1, 'D:/github-repos/movie-recommendation-system-with-sentiment-analysis-with-nlp/data/01_raw')

    import load_raw_data

    file_name = 'D:/github-repos/movie-recommendation-system-with-sentiment-analysis-with-nlp/data/01_raw/movie_metadata.csv'
    data = load_raw_data.load_data(file_name)


    plt.figure(figsize=(20, 20))
    plt.barh(data[column_name].value_counts().index, data[column_name].value_counts())
    plt.title('Number of movies per ' + column_name)
    plt.ylabel(column_name)
    plt.xlabel('Number of movies')
    plt.show()