import sklearn.metrics as sm
import math
import numpy as np

class Score:
    def __init__(self, metrics=['mape']):
        self._metrics = metrics
        self._available_metrics = ['mape', 'mae', 'mse', 'rmse', 'r2']

    #CALCULANDO A PORCENTAGEM MEDIA DE ERRO ABSOLUTO ( mean_absolute_percentage_error - MAPE)
    def mape(self, y_true, y_pred):
        MAPE = np.mean(np.abs((y_true - y_pred) / y_true))

        if(np.isnan(MAPE)):
            MAPE = 0
        if(np.isinf(MAPE)):
            MAPE = 0

        return MAPE

    # CALCULANDO MEDIA DE ERRO ABSOLUTO (MAE)
    def mae(self, y_true, y_pred):
        return sm.mean_absolute_error(y_true, y_pred)

    # CALCULANDO MEDIA DE ERRO QUADRATICO (mean squared error - MSE)
    def mse(self, y_true, y_pred):
        return  sm.mean_squared_error(y_true, y_pred)

    # CALCULANDO A RAIZ MEDIA DE ERRO QUADRATICO (calculate root mean squared error - RMSE)
    def rmse(self, y_true, y_pred):
        return math.sqrt(sm.mean_squared_error(y_true, y_pred))

    #CALCULANDO O COEFFICIENTE DE DETERMINACAO (R-SQUARED)
    def r2(self, y_true, y_pred):
        return sm.r2_score(y_true, y_pred)

    def f(self, x, y_true, y_pred):
        return {
            'mape' : self.mape(y_true, y_pred),
            'mae' : self.mae(y_true, y_pred),
            'mse' : self.mse(y_true, y_pred),
            'rmse' : self.rmse(y_true, y_pred),
            'r2' : self.r2(y_true, y_pred)
        }[x]

    def get_scores(self, y_true, y_pred):
        dict = {}
        for m in self._metrics:
            score = self.f(m, y_true, y_pred)
            dict[m] = score

        return dict
