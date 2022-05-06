import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import load_data # from load_data.py
import numpy as np
header = ['user_id', 'item_id', 'rating', 'timestamp']
seed = 80
kf = KFold(n_splits=5, shuffle=True)
data = load_data.load_dataset()
data = data.movie_info()

sparse_features = ["item_id", "user_id", "timestamp"]#准备特征
target = ['rating']#准备标签
#特征数值化 data里面3个特征都用LabelEncoder处理一下 就是把特征里面的值 从0到n开始编号
for f in  sparse_features:
    transfor = LabelEncoder()
    data[f] = transfor.fit_transform(data[f])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique()) for feat in sparse_features]
linear_feature_columns = fixlen_feature_columns


dnn_feature_columns = fixlen_feature_columns
print(dnn_feature_columns)
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
# print(feature_names,type(feature_names))
rmse = []
for train_index, test_index in kf.split(data):
    train, test= data.iloc[train_index], data.iloc[test_index]
    # train, test = movie[train_index], movie[test_index]

    train_model_input = {name: train[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}
    # 使用DeepFM进行训练
    # linear_feature_columns线性部分用FM dnn_feature_columns高阶部分用DNN  task='regression'表示回归任务
    model = DeepFM(linear_feature_columns,dnn_feature_columns, task='regression')
    # 优化器用adam 评价指标用mse
    model.compile("adam", "mse", metrics=['mse'])
    # train_model_input作为训练集 rating作为标签值
    history = model.fit(train_model_input, train['rating'].values, batch_size=256, epochs=1, verbose=True,
                        validation_split=0.2)
    pred = model.predict(test_model_input, batch_size=256)
    # 输出RMSE或MSE
    mse = round(mean_squared_error(test['rating'].values, pred), 4)

    rmse.append(mse)

score = np.mean(rmse)

print("mean rmse :", score)
