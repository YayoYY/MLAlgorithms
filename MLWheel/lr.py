#!/usr/bin/env
# -*- coding: utf-8 -*-

'''

Logistic Regression with BGD method

1. import "lr" from "MLWheel" module

	>>> from MLWheel import lr

2. initialize a LogisticRegreeession model

	>>> clf = lr.LogisticRegreession(features, alpha, max_iter)

    features : Features' name.
    alpha: Learning rate, default value: 0.1. Decrease alpha can avoid overfitting.
    max_iter: Max iteration num, default value: 100. Decrease max_iter can avoid overfitting.
    tol: Tolerance for stopping criteria, default value: 1e-4.

3. use your data fit a model

	>>> clf.fit(X, y)

    X is an array of shape (n_samples, n_features)
    y is an array of shape (n_samples, 1)

'''

__author__ = 'YayoYY'

import numpy as np

class LogisticRegression(object):

    def __init__(self, features, alpha=1, max_iter=100, tol=1e-4):
        self.__features = features
        feature_num = len(features)
        self.__w = np.zeros(feature_num)
        self.__b = 0
        self.__alpha = alpha
        self.__max_iter = max_iter
        self.__tol = tol

    @property
    def features_(self):
        features = {}
        for i,item in enumerate(self.__features):
            features[item] = self.__w[i]
        return features

    @property
    def classes_(self):
        return self.__classes

    @property
    def iter_num_(self):
        return self.__iter_num

    @property
    def coef_(self):
        return self.__w
    
    def __calc_z(self, X):
        return np.dot(self.__w, X) + self.__b
    
    def __calc_y(self, X):
        z = self.__calc_z(X)
        return 1/(1 + np.exp(-z))

    def predict(self, X):
        z = self.__calc_z(X.T)
        return 1/(1 + np.exp(-z))
    
    def __calc_dz(self, y, y_hat):
        return y_hat - y
    
    def __calc_dw(self, m, X, dz):
        return 1/m * np.dot(X, dz.reshape(m,1))
    
    def __calc_db(self, m, dz):
        return 1/m * np.sum(dz)
    
    def __calc_J(self, m, y, y_hat):
        return -1/m * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

    def __bgd(self, X, y):
        self.__iter_num = 0
        J = float('inf')
        for i in range(self.__max_iter):
            m = len(y)
            y_hat = self.__calc_y(X)
            dz = self.__calc_dz(y, y_hat)
            dw = self.__calc_dw(m, X, dz).reshape(X.shape[0],)
            db = self.__calc_db(m, dz)
            J_new = self.__calc_J(m, y, y_hat)
            if np.abs(J - J_new) <= self.__tol:
                break
            else:
                J = J_new
                self.__iter_num += 1
                self.__w = self.__w - self.__alpha * dw
                self.__b = self.__b - self.__alpha * db
        
    def fit(self, X, y):
        self.__classes = np.array(np.unique(y))
        self.__bgd(X.T, y)
        return self