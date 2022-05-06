# BPR MF

import sys
import numpy as np
import pandas as pd
from math import ceil
from tqdm import trange
from subprocess import call
from itertools import islice
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, dok_matrix
import  load_data
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold

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

items_col = 'item_id'
users_col = 'user_id'
ratings_col = 'rating'
threshold = 3
data = load_data.load_dataset()
movie = data.movie_info()

from scipy.sparse import csr_matrix
class BPR:
    def __init__(self, learning_rate=0.01, n_factors=15, n_iters=10,
                 batch_size=1000, reg=0.01, seed=1234, verbose=True):
        self.reg = reg
        self.seed = seed
        self.verbose = verbose
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # to avoid re-computation at predict
        self._prediction = None

    def fit(self, ratings):
        ratings = csr_matrix(ratings, shape=(ratings.shape))
        indptr = ratings.indptr
        indices = ratings.indices
        n_users, n_items = ratings.shape

        batch_size = self.batch_size
        if n_users < batch_size:
            batch_size = n_users
            sys.stderr.write('WARNING: Batch size is greater than number of users,'
                             'switching to a batch size of {}\n'.format(n_users))
        batch_iters = n_users // batch_size

        # initialize random weights
        rstate = np.random.RandomState(self.seed)
        self.user_factors = rstate.normal(size=(n_users, self.n_factors))
        self.item_factors = rstate.normal(size=(n_items, self.n_factors))

        # progress bar for training iteration if verbose is turned on
        loop = range(self.n_iters)
        if self.verbose:
            loop = trange(self.n_iters, desc=self.__class__.__name__)

        for _ in loop:
            for _ in range(batch_iters):
                sampled = self._sample(n_users, n_items, indices, indptr)
                sampled_users, sampled_pos_items, sampled_neg_items = sampled
                self._update(sampled_users, sampled_pos_items, sampled_neg_items)

        return self

    def _sample(self, n_users, n_items, indices, indptr):
        """sample batches of random triplets u, i, j"""
        sampled_pos_items = np.zeros(self.batch_size, dtype=np.int)
        sampled_neg_items = np.zeros(self.batch_size, dtype=np.int)
        sampled_users = np.random.choice(
            n_users, size=self.batch_size, replace=False)

        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user]:indptr[user + 1]]
            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

        return sampled_users, sampled_pos_items, sampled_neg_items

    def _update(self, u, i, j):
        user_u = self.user_factors[u]
        item_i = self.item_factors[i]
        item_j = self.item_factors[j]

        r_uij = np.sum(user_u * (item_i - item_j), axis=1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))

        # repeat the 1 dimension sigmoid n_factors times so
        # the dimension will match when doing the update
        sigmoid_tiled = np.tile(sigmoid, (self.n_factors, 1)).T

        # update using gradient descent
        grad_u = sigmoid_tiled * (item_j - item_i) + self.reg * user_u
        grad_i = sigmoid_tiled * -user_u + self.reg * item_i
        grad_j = sigmoid_tiled * user_u + self.reg * item_j
        self.user_factors[u] -= self.learning_rate * grad_u
        self.item_factors[i] -= self.learning_rate * grad_i
        self.item_factors[j] -= self.learning_rate * grad_j
        return self

    def predict(self):
        if self._prediction is None:
            self._prediction = self.user_factors.dot(self.item_factors.T)

        return self._prediction

    def _predict_user(self, user):
        user_pred = self.user_factors[user].dot(self.item_factors.T)
        return user_pred

    def recommend(self, ratings, N=5):
        n_users = ratings.shape[0]
        recommendation = np.zeros((n_users, N), dtype=np.uint32)
        for user in range(n_users):
            top_n = self._recommend_user(ratings, user, N)
            recommendation[user] = top_n

        return recommendation

    def _recommend_user(self, ratings, user, N):
        """the top-N ranked items for a given user"""
        scores = self._predict_user(user)
        liked = set(ratings[user].indices)
        count = N + len(liked)
        if count < scores.shape[0]:
            ids = np.argpartition(scores, -count)[-count:]
            best_ids = np.argsort(scores[ids])[::-1]
            best = ids[best_ids]
        else:
            best = np.argsort(scores)[::-1]

        top_n = list(islice((rec for rec in best if rec not in liked), N))
        return top_n

    def get_similar_items(self, N=5, item_ids=None):

        normed_factors = normalize(self.item_factors)
        knn = NearestNeighbors(n_neighbors=N + 1, metric='euclidean')
        knn.fit(normed_factors)
        if item_ids is not None:
            normed_factors = normed_factors[item_ids]

        _, items = knn.kneighbors(normed_factors)
        similar_items = items[:, 1:].astype(np.uint32)
        return similar_items

def score(model, ratings):
    auc = 0.0
    n_users, n_items = ratings.shape
    for user, row in enumerate(ratings):
        y_pred = model._predict_user(user)
        row =list(map(int, row))
        y_true = np.zeros(n_items)
        y_true[row] = 1
        auc += roc_auc_score(y_true, y_pred)
        print(y_pred)
        rms = sqrt(mean_squared_error(y_true, y_pred))

    auc /= n_users
    return auc


from sklearn.model_selection import KFold
seed = 80
kf = KFold(n_splits=5, shuffle=True)
from sklearn.model_selection import KFold
train_score=[]; test_score=[]
for train_index, test_index in kf.split(movie):
    train, test = movie.iloc[train_index], movie.iloc[test_index]
    X_train, X_test = data.create_matrix(train, test)
    bpr_params = {'reg': 0.01,
                  'learning_rate': 0.1,
                  'n_iters': 100,
                  'n_factors': 15,
                  'batch_size': 100}

    bpr = BPR(**bpr_params)
    bpr.fit(X_train)
    train_score.append(score(bpr, X_train))
    test_score.append(score(bpr, X_test))

trai_score = np.mean(train_score)
tes_score = np.mean(test_score)
print(trai_score)
print(tes_score)
