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


class cal_loss():
    def __init__(self):
        super(cal_loss, self).__init__()

    def rmse(self,prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))


class prediction():
    def __init__(self):
        super(prediction, self).__init__()

    def predict(self, ratings, similarity, type='user'):
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1)
            # You use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
                [np.abs(similarity).sum(axis=1)]).T
        elif type == 'item':
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        return pred

class calculate():
    def __init__(self):
        super(calculate, self).__init__()

    def cos(self,train_data_matrix):
        user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
        item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
        # print("cos",np.shape(user_similarity))
        # print("cos",np.shape(item_similarity))
        return  user_similarity, item_similarity

    def pearson(self,train_data_matrix):
        # print(train_data_matrix)
        user_similarity = np.corrcoef(train_data_matrix)
        # numpy corrcoef - compute correlation matrix while ignoring missing data
        # item_similarity = ma.corrcoef(ma.masked_invalid(train_data_matrix.T))
        item_similarity = ma.corrcoef(train_data_matrix.T)

        return user_similarity, item_similarity




