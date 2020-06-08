import numpy as np
from sklearn import preprocessing, svm
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.model_selection import GridSearchCV

class gridSVR:
    def __init__(self, kernels=['rbf'], c=[-15, 15, 1], gamma=[0, 0, 1], epsilon=[-9, 0, 2]):
        self._kernels = kernels
        self._c_begin, self._c_end, self._c_step = c
        self._g_begin, self._g_end, self._g_step =  gamma
        self._p_begin, self._p_end, self._p_step = epsilon

        self._c_qtd = int((abs(self._c_begin) + abs(self._c_end)) / self._c_step) + 1
        self._g_qtd = int((abs(self._g_begin) + abs(self._g_end)) / self._g_step) + 1
        self._p_qtd = int((abs(self._p_begin) + abs(self._p_end)) / self._p_step) + 1

        self._model = None

    def fit(self, X_train=None, y_train=None, verbose=False):
        print("Iniciando GridSearchCV para SVR...")

        if(verbose):
            print('Search Space - kernel', self._kernels)
            print('Search Space - C', np.logspace(self._c_begin, self._c_end, self._c_qtd))
            print('Search Space - gamma', np.logspace(self._g_begin, self._g_end, self._g_qtd))
            print('Search Space - epsilon', np.logspace(self._p_begin, self._p_end, self._p_qtd))

        gsc = GridSearchCV(
                estimator = svm.SVR(),
                param_grid = {
                    'kernel': self._kernels,
                    'C': np.logspace(self._c_begin, self._c_end, self._c_qtd),
                    'gamma': np.logspace(self._g_begin, self._g_end, self._g_qtd),
                    'epsilon': np.logspace(self._p_begin, self._p_end, self._p_qtd)
                },
                cv=5, scoring="neg_mean_absolute_error", verbose=0, n_jobs=2)

        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_

        if(verbose):
            print("Best Params:", best_params)

        self._model = svm.SVR(kernel = best_params["kernel"],
                        C = best_params["C"],
                        epsilon = best_params["epsilon"],
                        gamma = best_params["gamma"],
                        coef0 = 0.1, shrinking = True,
                        tol = 0.001, cache_size = 200, verbose=False, max_iter = -1)

        self._model.fit(X_train,y_train)

    def predict(self, X=None):
        return self._model.predict(X)
