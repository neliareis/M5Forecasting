import pandas as pd


path = 'D:/Users/NKings/Documents/PUC/Data_Science/Dataset/'

# Divide a base por estado
def split_df(path, state):
    '''Recebe o caminho da base e a sigla de um estado (CA, TX, WI)'''

    X = pd.read_csv(path + 'sales_train_validation.csv')
    count = 1
    lista = []
    df_result = pd.DataFrame()
    for i in X.itertuples():
        stt = i.state_id
        if stt == state:
            count += 1
            lista.append(i)
    
    df_result = pd.DataFrame(lista)
    df_result.drop(['Index'], axis=1, inplace=True)
    
    df_result.to_csv('state_' + state + '.csv', index=False)
    print('Done!')

split_df(path, 'CA')
