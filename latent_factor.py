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
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error


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
movi_header = ['item_id', 'gen']
movie_genr = pd.read_csv('movie/movie_genre.dat',sep='\t',names = movi_header)
user = user_age['user_id']
age = user_age['age']
occ = user_occu['occ']
list_tuples = list(zip(user,age, occ))
user_feature = pd.DataFrame(list_tuples, columns=['user_id','age', 'occ'])

# one hot method
movie_attr = np.zeros((1682,18),dtype=np.int)
for row in movie_genr.itertuples():
    movie_attr[row[1] - 1 ,row[2] -1] = 1
movie_attr = pd.DataFrame(movie_attr)
index =[]
a = [i for i in range(1682)]
movie_attr= movie_attr.assign(item_id=a)
# data = pd.merge(pd.merge(movie, user_feature, on='user_id'), movie_attr, on='item_id')
# print(data)
# print(data[0:100])

data = pd.merge(movie, user_feature, on='user_id')
data =data.drop(['timestamp','age','occ'], axis=1)
# origin
#one-hot encoder
columns=['user_id', 'item_id']

for i in columns:
    get_dummy_feature=pd.get_dummies(data[i])
    data=pd.concat([data, get_dummy_feature],axis=1)
    data=data.drop(i, axis=1)
print(data.head())


X= data.drop('rating', axis=1)
Y= data['rating']

X_train,X_val,Y_train,Y_val=train_test_split(X, Y, test_size=0.3, random_state=123)
print(np.shape(X_train), np.shape(Y_train))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(y, y_hat): #对每一个样本计算损失
    return np.log(1 + np.exp(-y * y_hat))

def df_logit(y, y_hat):
    return sigmoid(-y * y_hat) * (-y)


class FactorizationMachine(BaseEstimator):
    def __init__(self, k=5, learning_rate=0.01, iternum=5):
        self.w0 = None
        self.W = None
        self.V = None
        self.k = k
        self.alpha = learning_rate
        self.iternum = iternum

    def _FM(self, Xi):
        interaction = np.sum((Xi.dot(self.V)) ** 2 - (Xi ** 2).dot(self.V ** 2))
        y_hat = self.w0 + Xi.dot(self.W) + interaction / 2
        return y_hat[0]

    def _FM_SGD(self, X, y):
        m, n = np.shape(X)
        # 初始化参数
        self.w0 = 0
        self.W = np.random.uniform(size=(n, 1))
        self.V = np.random.uniform(size=(n, self.k))  # Vj是第j个特征的隐向量  Vjf是第j个特征的隐向量表示中的第f维

        for it in range(self.iternum):
            total_loss = 0
            for i in range(m):  # 遍历训练集
                y_hat = self._FM(Xi=X[i])  # X[i]是第i个样本  X[i,j]是第i个样本的第j个特征

                total_loss += logit(y=y[i], y_hat=y_hat)  # 计算logit损失函数值
                dloss = df_logit(y=y[i], y_hat=y_hat)  # 计算logit损失函数的外层偏导

                dloss_w0 = dloss * 1  # 公式中的w0求导，计算复杂度O(1)
                self.w0 = self.w0 - self.alpha * dloss_w0

                for j in range(n):
                    if X[i, j] != 0:
                        dloss_Wj = dloss * X[i, j]  # 公式中的wi求导，计算复杂度O(n)
                        self.W[j] = self.W[j] - self.alpha * dloss_Wj
                        for f in range(self.k):  # 公式中的vif求导，计算复杂度O(kn)
                            dloss_Vjf = dloss * (X[i, j] * (X[i].dot(self.V[:, f])) - self.V[j, f] * X[i, j] ** 2)
                            self.V[j, f] = self.V[j, f] - self.alpha * dloss_Vjf
            print('iter={}, loss={:.4f}'.format(it, total_loss / m))


        return self

    def _FM_predict(self, X):
        predicts, threshold = [], 0.5  # sigmoid阈值设置
        for i in range(X.shape[0]):  # 遍历测试集
            y_hat = self._FM(Xi=X[i])  # FM的模型方程
            predicts.append(-1 if sigmoid(y_hat) < threshold else 1)
        return np.array(predicts)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
            y = np.array(y)

        return self._FM_SGD(X, y)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
        return self._FM_predict(X)
    def predict_proba(self, X):
        pass

model = FactorizationMachine(k=5, learning_rate=0.01, iternum=20)
model.fit(X_train, Y_train)

y_pred = model.predict(X_train)
print(y_pred)
print(print())
print('training mean square error:',mean_squared_error(Y_train.values, y_pred))

y_pred = model.predict(X_val)
print('testing mean square error:',mean_squared_error(Y_val.values, y_pred))
