def load_data(file_path):
    '''Function to load file raw data into pandas dataframe 
    :param file_path: path to the file'''
    import pandas as pd  
    data = pd.read_csv(file_path)
    return data 