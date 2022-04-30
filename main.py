import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import model_selection as cv
import load_data # from load_data.py
import calculate as cs # calculate_similarity
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import MF

# random seed
seed = 80
kf = KFold(n_splits=5, shuffle=True)
# load data and create training/ test matrix
data = load_data.load_dataset()
movie = data.movie_info()
# train_data, test_data = cv.train_test_split(movie, test_size=0.25)
result = 0
UCFs=[]; ICFs=[];  UCFp=[]; ICFp=[]; average_RMSE=[]

for train_index, test_index in kf.split(movie):
    train, test = movie.iloc[train_index], movie.iloc[test_index]
    # train, test = movie[train_index], movie[test_index]
    train_data_matrix, test_data_matrix = data.create_matrix(train, test)
    #CF
    # cosine_similarity
    data_cal = cs.calculate() # calculation similarity
    data_cal_predict = cs.prediction() # make prediction
    user_similarity_cos, item_similarity_cos= data_cal.cos(train_data_matrix)
    item_prediction_cos = data_cal_predict.predict(train_data_matrix, item_similarity_cos, 'item')
    user_prediction_cos = data_cal_predict.predict(train_data_matrix, user_similarity_cos, 'user')

    # pearson correlation
    user_similarity_pearson, item_similarity_pearson = data_cal.pearson(train_data_matrix)
    item_prediction_pearson = data_cal_predict.predict(train_data_matrix, item_similarity_pearson, 'item')
    user_prediction_pearson = data_cal_predict.predict(train_data_matrix, user_similarity_pearson, 'user')

    # Latent Factor Model
    latent_factor_loss = MF.main(train_data_matrix, test_data_matrix )
    average_RMSE.append(latent_factor_loss)
    # for row in train_data_matrix:
    #     user_tensor = torch.LongTensor([row])
    #     item_tensor = torch.LongTensor([key[1] for key in data.keys()])
    #     rating_tensor = torch.FloatTensor([val for val in data.values()])
    #
    #

    data_cal_loss = cs.cal_loss()  # calculation similarity
    UCFs.append(data_cal_loss.rmse(user_prediction_cos, test_data_matrix))
    ICFs.append(data_cal_loss.rmse(item_prediction_cos, test_data_matrix))
    UCFp.append(data_cal_loss.rmse(user_prediction_pearson, test_data_matrix))
    ICFp.append(data_cal_loss.rmse(item_prediction_pearson, test_data_matrix))

UCFs = np.mean(UCFs)
ICFs = np.mean(ICFs)
UCFp = np.mean(UCFp)
ICFp = np.mean(ICFp)
average_RMSE = np.mean(average_RMSE)
print('User-based CF RMSE cosine : ' + str(data_cal_loss.rmse(user_prediction_cos, test_data_matrix)))
print('Item-based CF RMSE cosine: ' + str(data_cal_loss.rmse(item_prediction_cos, test_data_matrix)))

print('User-based CF RMSE Pearson: ' + str(data_cal_loss.rmse(user_prediction_pearson, test_data_matrix)))
print('Item-based CF RMSE Pearson: ' + str(data_cal_loss.rmse(item_prediction_pearson, test_data_matrix)))

d = 1
result = {
          "Method": ["UCF-s", "UCF-p", "ICF-p","ICF-p"],
          "RMSE": [UCFs, UCFp, ICFs, ICFp]
          # "Recal@10":score,
          # "NDCG@10":score
          }
result = pd.DataFrame(result)
result.to_csv(f'result{d}.csv',index=False)

