import json
import scores as s
import numpy as np
import pandas as pd
import autoSVR as svr
from sklearn import preprocessing as skp

path = 'D:/Users/NKings/Documents/PUC/Data_Science/Dataset/Dataset_ind/'


df = pd.read_csv(path + 'FOODS_1_001_CA_1_validation.csv')
# df.fillna(-99999, inplace=True) # retira os valores NaN e substitui por -99999

# produtos = df['ID']

# Nenhum historico eh utilizado, apenas o valor atual,
# mas pode configurar essa variavel para adicionar outras N medicoes
look_back = 1

forcast_col = 'Qtd_dia'
forcast_name = 'TARGET'

# Normalizando dados
scalerY = skp.MinMaxScaler(feature_range=(0, 1))
scalerX = skp.MinMaxScaler(feature_range=(0, 1))

# listas para armazenar as metricas
# MAPE_list = []
# MAE_list = []
# MSE_list = []
# RMSE_list = []
# R2_list = []

# skipDone = False
# lastID = 'FOODS_3_827_CA_4_validation'

# Cria colunas novas para o tamanho de look_back
for i in range(look_back):
    df[forcast_col + "_" + str(i+1)] = df[forcast_col].shift(-(i+1))

# cria a coluna de target
df[forcast_name] = df[forcast_col].shift(-look_back-1)
df.dropna(inplace=True)

X = np.array(df)
X = np.array(df.drop([forcast_name], 1))
X = scalerX.fit_transform(X)

y = np.array(df[forcast_name]).reshape(-1, 1)
y = scalerY.fit_transform(y)

index = int(X.shape[0]*0.7)
X_train = X[:index,:]
X_test = X[index:,:]
y_train = y[:index]
y_test = y[index:]

print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)

y_test_real = scalerY.inverse_transform(y)[index:]

######################################################
# TREINAMENTO DO MODELO
######################################################

model = svr.gridSVR()
model.fit(X_train, y_train.reshape(-1,), True)


print("OBTENDO PREDICTS...")
#OBTENDO PREDICTS
trainPredictNorm = model.predict(X_train)
testPredictNorm = model.predict(X_test)


trainPredict = scalerY.inverse_transform(trainPredictNorm.reshape(-1, 1))
testPredict = scalerY.inverse_transform(testPredictNorm.reshape(-1, 1))


###########################################################################
# METRICAS
###########################################################################

score = s.Score(['rmsse', 'mase', 'mape', 'mae', 'mse', 'rmse', 'r2'])
metrics = score.get_scores(y_test, testPredict, X_test)
metrics_norm = score.get_scores(y_test, testPredictNorm, X_test)
print("Métricas: ", metrics)
print("Métricas normalizadas: ", metrics_norm)

with open('results/metrics.json', 'w') as outfile:
    json.dump(metrics, outfile)

with open('results/metrics_norm.json', 'w') as outfile:
    json.dump(metrics_norm, outfile)

# MAPE_list = metrics_norm["mape"]
# MAE_list = metrics_norm["mae"]
# MSE_list = metrics_norm["mse"]
# RMSE_list = metrics_norm["rmse"]
# R2_list = metrics_norm["r2"]

