import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import load_data # from load_data.py
from sklearn import model_selection as cv
import scipy.stats
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances
import numpy.ma as ma
import numpy as np
# from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer


def user_info():
    n_users = movie.user_id.unique().shape[0]
    n_items = movie.item_id.unique().shape[0]
    # print('Number of users = ' + str(self.n_users) + ' | Number of movies = ' + str(self.n_items))
    return n_users, n_items


header = ['user_id', 'item_id', 'rating', 'timestamp']
user_occ_header = ['user_id', 'occ']
user_age_attr = ['user_id', 'age']
# movie_genr_att = []
movie = pd.read_csv('movie/user_movie.dat', sep='\t', names=header)
user_age = pd.read_csv('movie/user_age.dat', sep='\t',names=user_age_attr)
user_occu = pd.read_csv('movie/user_occupation.dat', sep='\t',names =user_occ_header)
movie_genr = pd.read_csv('movie/movie_genre.dat')

user = user_age['user_id']
age = user_age['age']
occ = user_occu['occ']
list_tuples = list(zip(user,age, occ))
user_feature = pd.DataFrame(list_tuples, columns=['user','age', 'occ'])
# print(dframe)

user_id = []; item_id = [];  item_feature=[]
for row in movie.itertuples():
    user_id.append(row[1])
    item_id.append(row[2])
user_id = pd.Series(user_id)
item_id = pd.Series(item_id)

user_onehot = pd.get_dummies(user_id)
item_onehot = pd.get_dummies(item_id)
print(np.shape(user_onehot)) # (100000, 943)
print(np.shape(item_onehot)) # (100000, 1682)
"""
for row in movie.itertuples():
        dictionary = {'user': row[1], 'item': row[2], 'rating': row[3]}
        for row2 in dframe.itertuples():
            if row2[1] == row[1]:
                dictionary['age'] = row2[1]
                dictionary['occ'] = row2[2]
                # print(dictionary)
        dataframe.append(dictionary)
print(dataframe)

matrix = []

"""
"""

train = [
	{"user": "1", "item": "5", "age": 19},
	{"user": "2", "item": "43", "age": 33},
	{"user": "3", "item": "20", "age": 55},
	{"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()
X = v.fit_transform(train)
print(X.toarray())

y = np.repeat(1.0,X.shape[0])
fm = pylibfm.FM()
fm.fit(X,y)
fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))"""