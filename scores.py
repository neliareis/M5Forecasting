import sklearn.metrics as sm
import math
import numpy as np

class Score:
    def __init__(self, metrics=['mape']):
        self._metrics = metrics
        self._available_metrics = ['rmsse', 'mase', 'mape', 'mae', 'mse', 'rmse', 'r2']

    #CALCULANDO A PORCENTAGEM MEDIA DE ERRO ABSOLUTO ( mean_absolute_percentage_error - MAPE)
    def mape(self, y_true, y_pred):

        MAPE = np.mean(np.abs((y_true - y_pred) / y_true))

        if(np.isnan(MAPE)):
            MAPE = 0
        if(np.isinf(MAPE)):
            MAPE = 0

        return MAPE

    # def MAPEConstantErrorForZeros(self, y_true, y_pred):
    #     print("EUUU", y_true)
    #     if(float(y_true) == 0):
    #         error = 0.5
    #     else:
    #         error = abs((float(y_true)-float(y_pred))/float(y_true))
    #     return error

    # def MAPEIgnoringZeros(self, y_true, y_pred):
    #     if(float(y_true) != 0):
    #         error = abs((float(y_true)-float(y_pred))/float(y_true))
    #         return error
    #     else:
    #         return 0

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

    def mase(self, training_series, y_true, y_pred):
        """
        Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.

        parameters:
        training_series: the series used to train the model, 1d numpy array
        y_true: the test series to predict, 1d numpy array or float
        y_pred: the prediction of y_true, 1d numpy array (same size as y_true) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
        """

        #print("Needs to be tested.")
        n = training_series.shape[0]
        d = np.abs(  np.diff( training_series) ).sum()/(n-1)

        errors = np.abs(y_true - y_pred )
        return errors.mean()/d

    def rmsse(self, training_series, y_true, y_pred):
        ''' RMSSE - MEAN SQUARED SCALED ERROR '''

        n = training_series.shape[0]
        d = np.abs(  np.diff( training_series) ).sum()/(n-1)

        errors = ((y_true - y_pred) ** 2)
        
        return errors.mean()/d

    def f(self, x, y_true, y_pred, training_series):
        return {
            'rmsse': self.rmsse(training_series, y_true, y_pred),
            'mase': self.mase(training_series, y_true, y_pred),
            'mape' : self.mape(y_true, y_pred),
            # 'MAPEConstantErrorForZeros': self.MAPEConstantErrorForZeros(y_true, y_pred),
            # 'MAPEIgnoringZeros': self.MAPEIgnoringZeros(y_true, y_pred),
            'mae' : self.mae(y_true, y_pred),
            'mse' : self.mse(y_true, y_pred),
            'rmse' : self.rmse(y_true, y_pred),
            'r2' : self.r2(y_true, y_pred)
        }[x]

    def get_scores(self, y_true, y_pred, training_series):
        dict = {}
        for m in self._metrics:
            score = self.f(m, y_true, y_pred, training_series)
            dict[m] = score

        return dict
