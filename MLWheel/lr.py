#!/usr/bin/env
# -*- coding: utf-8 -*-

'''

1. import "lr" from "MLWheel" module

	>>> from MLWheel import lr

2. initialize a LogisticRegreeession model

	>>> clf = lr.LogisticRegreession(features, alpha, T)

    features: your features' name
    alpha: learning rate, default value:0.01. decrease alpha can avoid overfitting.
    T: iter_num, default value:100

3. use your data fit a model

	>>> 

'''

__author__ = 'YayoYY'

import numpy as np

class LogisticRegression(object):
    
    def __init__(self, features, alpha=0.01, iter_num=100):
        self.features = features
        feature_num = len(features)
        self.w = np.zeros(feature_num).reshape(feature_num,1)
        self.b = 0
        self.alpha = alpha
        self.iter_num = iter_num

    @property
    def model(self):
        model = {}
        for i,item in enumerate(features):
            model[item] = self.w[i]
        return model
    
    def __calc_z(self, X):
        return np.dot(self.w.T, X) + self.b
    
    def __calc_y(self, X):
        z = self.__calc_z(X)
        return 1/(1 + np.exp(-z))

    def predict(self, X):
        z = self.__calc_z(X.T)
        return 1/(1 + np.exp(-z))
    
    def __calc_dz(self, y, y_hat):
        return y_hat - y
    
    def __calc_dw(self, m, X, dz):
        return 1/m * np.dot(X, dz.T)
    
    def __calc_db(self, m, dz):
        return 1/m * np.sum(dz)
    
    def __calc_J(self, m, y, y_hat):
        return -1/m * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

    def __bgd(self, X, y):
        J = []
        for i in range(self.iter_num):
            m = len(y)
            y_hat = self.__calc_y(X)
            dz = self.__calc_dz(y, y_hat)
            dw = self.__calc_dw(m, X, dz)
            db = self.__calc_db(m, dz)
            J_new = self.__calc_J(m, y, y_hat)
            J.append(J_new)
            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db
        
    def fit(self, X, y):
        X = X.T
        y = y.reshape(1, len(y))
        self.__bgd(X, y)
        return self