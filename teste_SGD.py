import os
import json
import glob
import xgboost
import scores as s
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

path = 'D:/Users/NKings/Documents/PUC/Data_Science/base_FOODS/'
# df = pd.read_csv(path + 'FOODS_1_001_CA_1_validation.csv')

extra_drop=['d', 'item_id', 'store_id', 'state_id', 'cat_id', 'dept_id', 'date', 'wm_yr_wk', 'weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']

# define qual regressor deve ser usado [xgb, sgd]
regressor = 'sgd'

all_files = glob.glob(os.path.join(path, '*.csv'))

RMSSE_list = []
MASE_list = []
MAPE_list = []
MAPE_ConstantErrorForZeros = []
MAPE_IgnoringZeros = []
MAE_list = []
MSE_list = []
RMSE_list = []
R2_list = []
count = 1

for produto in all_files:
	 
	df_from_each_file = pd.read_csv(produto)
	# df_from_each_file.replace(' ', 0, inplace=True)
	df_from_each_file.fillna(0, inplace=True)

	y = df_from_each_file['Qtd_dia'].values
	X = df_from_each_file.drop(['Qtd_dia'] + extra_drop, axis=1).values

	index = int(X.shape[0]-28)
	X_train = X[:index,:]
	X_test = X[index:,:]
	y_train = y[:index]
	y_test = y[index:]

	print("X_train", X_train.shape)
	print("X_test", X_test.shape)
	print("y_train", y_train.shape)
	print("y_test", y_test.shape)

	print('Treinando modelo:', count, '/', X_train.shape[0])
	count += 1

	if regressor == 'sgd':
		model = SGDRegressor(alpha=0.1,
				average=False,
				early_stopping=False,
				epsilon=0.1,
				eta0=0.01,
				fit_intercept=True,
				l1_ratio=0.15,
				learning_rate='constant',
				loss='squared_loss',
				max_iter=10000,
				n_iter_no_change=5,
				penalty='l2',
				power_t=0.25,
				random_state=None,
				shuffle=True,
				tol=0.001,
				validation_fraction=0.1,
				verbose=0,
				warm_start=False)

		model.fit(X_train, y_train)

		print("OBTENDO PREDICTS...")
	    #OBTENDO PREDICTS
		y_pred = model.predict(X_test)

	else:
		model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)

		model.fit(X_train, y_train)

		print("OBTENDO PREDICTS...")
	    #OBTENDO PREDICTS
		y_pred = model.predict(X_test)

	# MÉTRICAS
	score = s.Score(['rmsse', 'mase', 'mape', 'mae', 'mse', 'rmse', 'r2'])
	metrics = score.get_scores(y_test, y_pred, X_test)
	print("Métricas: ", metrics)

	# Salva as métricas de cada produto
	with open(str(produto) + '_' + regressor + '.json', 'w') as outfile:
	    json.dump(metrics, outfile)

	# armazena as métricas de todos os produtos em listas
	RMSSE_list = metrics['rmsse']
	MASE_list = metrics['mase']
	MAPE_list = metrics["mape"]
	# MAPE_ConstantErrorForZeros = metrics["MAPEConstantErrorForZeros"]
	# MAPE_IgnoringZeros = metrics["MAPEIgnoringZeros"]
	MAE_list = metrics["mae"]
	MSE_list = metrics["mse"]
	RMSE_list = metrics["rmse"]
	R2_list = metrics["r2"]

RMSSE_medio = np.mean(RMSSE_list)
MASE_medio = np.mean(MASE_list)
MAPE_medio = np.mean(MAPE_list)
MAE_medio = np.mean(MAE_list)
MSE_medio = np.mean(MSE_list)
RMSE_medio = np.mean(RMSE_list)
R2_medio = np.mean(R2_list)

# exibe as média das métricas
print('\n\n')
print('******** MÉDIA DAS MÉTRICAS ********')
print('RMSSE medio', np.mean(RMSSE_list))
print('MASE medio', np.mean(MASE_list))
print('MAPE medio', np.mean(MAPE_list))
# print('MAPE ConstantErrorForZeros', np.mean(MAPE_ConstantErrorForZeros))
# print('MAPE IgnoringZeros', np.mean(MAPE_IgnoringZeros))
print('MAE medio', np.mean(MAE_list))
print('MSE medio', np.mean(MSE_list))
print('RMSE medio', np.mean(RMSE_list))
print('R2 medio', np.mean(R2_list))

#salvar_metrics = [RMSSE_medio, MASE_medio, MAPE_medio, MAE_medio, MSE_medio, RMSE_medio, R2_medio]
dic_metrics = {'RMSSE_medio': [RMSSE_medio],
				  ' MASE_medio': [MASE_medio],
				  'MAPE_medio': [MAPE_medio],
				  'MAE_medio': [MAE_medio],
				  'MSE_medio': [MSE_medio],
				  'RMSE_medio': [RMSE_medio],
				  ' R2_medio':  [R2_medio]
				  }
df_metrics = pd.DataFrame(dic_metrics)
df_metrics.to_csv('result_metrics_media_' + regressor + '.txt', index=False)

