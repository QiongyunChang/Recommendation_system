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
from sklearn.metrics import recall_score
from sklearn.metrics import ndcg_score
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
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# print(feature_names,type(feature_names))
rmse = []; ndcg_scoreee=[]
for train_index, test_index in kf.split(data):
    train, test= data.iloc[train_index], data.iloc[test_index]
    # train, test = movie[train_index], movie[test_index]

    train_model_input = {name: train[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}
    print("test_model",test_model_input)
    # 使用DeepFM进行训练
    '''
    # wide and deep 
    model = WDL(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64),
                           l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                           dnn_activation='relu', task='regression')

    '''

    '''
    #deep cross
    model = DCN(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64),
                           l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                          dnn_activation='relu', task='regression')
    '''
    '''
    #xDeepFM
    model = xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64),
                     l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                     dnn_activation='relu', task='regression')

    '''
    #'''
    #IPNN
    model = PNN(dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=True, use_outter=False, kernel_type='mat', task='regression', device='cpu', gpus=None)
    #'''

    '''
    # OPNN   
    model = PNN(dnn_feature_columns, dnn_hidden_units=(128, 128),
                l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001,
                seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=False,
                use_outter=True, kernel_type='mat', task='regression', device='cpu', gpus=None)
    '''

    ''' 
    model = PNN(dnn_feature_columns, dnn_hidden_units=(128, 128),
                l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001,
                seed=1024, dnn_dropout=0, dnn_activation='relu', 
                kernel_type='mat', task='regression', device='cpu', gpus=None)
    '''


    # xDeepFM

    '''#NFM
    model = NFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64),
                    l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                    dnn_activation='relu', task='regression')

    '''
    ''' 
    #AFM 
    model = AFM(linear_feature_columns, dnn_feature_columns,  use_attention=True, attention_factor=8, l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_att=1e-05, afm_dropout=0, init_std=0.0001, seed=1024, task='regression', device='cpu', gpus=None)
    '''


    '''#CCPM 
    model = CCPM(linear_feature_columns, dnn_feature_columns, conv_kernel_width=(6, 5), conv_filters=(4, 4), dnn_hidden_units=(256, ), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, dnn_dropout=0, init_std=0.0001, seed=1024, task='regression', device='cpu', dnn_use_bn=False, dnn_activation='relu', gpus=None)
    '''

    # model = DeepFM(linear_feature_columns,dnn_feature_columns, task='regression')
    # 优化器用adam 评价指标用mse
    model.compile("adam", "mse", metrics=['mse'])
    # train_model_input作为训练集 rating作为标签值
    history = model.fit(train_model_input, train['rating'].values, batch_size=256, epochs=4, verbose=True,
                        validation_split=0.2)
    pred = model.predict(test_model_input, batch_size=256)

    # print(test['rating'].values, pred)
    # print(np.shape(test['rating'].values), np.shape(pred))


    # 输出RMSE或MSE
    mse = round(mean_squared_error(test['rating'].values, pred), 4)
    pred = pred.flatten()

    # nsamples, nx, ny = test.shape
    # d2_train_dataset = test.reshape((nsamples, nx * ny))
    ndcg = ndcg_score([test['rating'].values], [pred], k=10)
    rmse.append(mse)
    ndcg_scoreee.append(ndcg)

score = np.mean(rmse)
ndcg_sco = np.mean(ndcg)
print("mean rmse :", score)
print("mean ndcg :", ndcg_sco)
