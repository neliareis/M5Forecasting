import os
import glob
import pandas as pd


path = 'D:/Users/NKings/Documents/PUC/Data_Science/Dataset/'


def split_df(path, state):
    X = pd.read_csv(path + 'sales_train_validation.csv')
    count = 1
    lista = []
    df_result = pd.DataFrame()
    for i in X.itertuples():
        stt = i.state_id
        if stt == state:
            count += 1
            lista.append(i)
    
    df_result = pd.DataFrame(lista, index=None)
    df_result.to_csv('teste.csv', index=False)
    print(count)

split_df(path, 'CA')
