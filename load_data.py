import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
# import Latent_Matrix as LM
class load_dataset():
    def __init__(self):
        super(load_dataset, self).__init__()
        self.n_users = 0
        self.items = 0
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        user_occ_header = ['user_id', 'occ']
        user_age_attr =['user_id', 'age']
        movie_header = ['movie', 'gen']

        self.movie_genr = pd.read_csv('movie/movie_genre.dat', sep='\t', names=movie_header)
        self.movie = pd.read_csv('movie/user_movie.dat', sep='\t', names=header)
        self.user_age = pd.read_csv('movie/user_age.dat', sep='\t', names = user_age_attr)
        self.user_occu = pd.read_csv('movie/user_occupation.dat', sep='\t', names = user_occ_header)

    def user_attributes(self):
        user = self.user_age['user_id']
        age = self.user_age['age']
        occ = self.user_occu['occ']
        list_tuples = list(zip(user, age, occ))
        dframe = pd.DataFrame(list_tuples, columns=['user', 'age', 'occ'])
        print(dframe)


    def user_item_onehot(self):
        user_id = []
        item_id = []
        for row in movie.itertuples():
            user_id.append(row[1])
            item_id.append(row[2])
        user_id = pd.Series(user_id)
        item_id = pd.Series(item_id)
        user_one_hot = pd.get_dummies(user_id) # (100000, 943)
        item_one_hot = pd.get_dummies(item_id) # (100000, 1682)
        return user_one_hot, item_one_hot


    def movie_att_onehot(self):
        movie_attr = np.zeros((1682, 18), dtype=np.int)
        for row in self.movie_genr.itertuples():
            movie_attr[row[1] - 1, row[2] - 1] = 1

    def movie_info(self):
        return  self.movie

    def user_info(self):
        self.n_users = self.movie.user_id.unique().shape[0]
        self.n_items = self.movie.item_id.unique().shape[0]
        # print('Number of users = ' + str(self.n_users) + ' | Number of movies = ' + str(self.n_items))
        return  self.n_users, self.n_items

    def create_matrix(self,train_data,test_data):
        n_users, n_items = self.user_info()
        # Create two user-item matrices, one for training and another for testing
        train_data_matrix = np.zeros((n_users, n_items))
        # print(np.shape(train_data))
        for line in train_data.itertuples():
            train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
        test_data_matrix = np.zeros((n_users, n_items))
        for line in test_data.itertuples():
            test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

        return train_data_matrix, test_data_matrix






