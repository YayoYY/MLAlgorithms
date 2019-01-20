#!/usr/bin/env
# -*- coding: utf-8 -*-

'''

Perceptron with SGD method

1. import "perceptron" from "MLWheel" module

	>>> from MLWheel import perceptron

2. initialize a Perceptron model

	>>> clf = perceptron.Perceptron(features, eta, max_iter, dual)

    features : Features' name.

    eta: Learning rate, default value: 0.1. Decrease eta can avoid overfitting.
    
    max_iter: Max iteration num, default value: 100. Decrease max_iter can avoid overfitting.
    
    dual: Dual or primal formulation, default value: False. Prefer dual=False when n_samples > n_features. 

3. method
    
    (1) use your data fit a model

    	>>> clf.fit(X, y)

        X is an array of shape (n_samples, n_features)
        y is an array of shape (n_samples,) ps. -1 for n-sample, +1 for p-sample
        return type self (an Perceptron instance)

    (2) predict

        >>> clf.predict(X)

        X is an array of shape (n_samples, n_features)
        return type is an array of shape (n_samples,)

4. attributes

    features_: dict, [featurename : weight]

    iter_num_: iteration number

'''

__author__ = 'YayoYY'

import numpy as np
import time

class Perceptron():

	def __init__(self, features, eta=0.1, max_iter=100, dual=False):
		self.__features = features
		feature_num = len(features)
		self.__eta = eta
		self.__max_iter = max_iter
		self.__w = np.zeros(feature_num)
		self.__b = 0
		self.__dual = dual

	@property
	def features_(self):
		features = {}
		for i,item in enumerate(self.__features):
			features[item] = self.__w[i]
		return features

	@property
	def iter_num_(self):
		return self.__iter_num

	def time_record(func):
		def wrapper(*args, **kw):
			time1 = time.time()
			result = func(*args, **kw)
			time2 = time.time()
			# print(func, 'COST', time2-time1)
			return result
		return wrapper

	def fit(self, X, y):
		X = X.T
		if self.__dual:
			self.__alpha = np.zeros(X.shape[1])
			self.__Gram = self.__calc_Gram(X)
			self.__iter_num = 0
			for i in range(self.__max_iter):
				y_hat = self.__calc_y_dual(y)
				j = self.__select_one_wrong(y, y_hat)
				if j:
					self.__iter_num = self.__iter_num + 1
					self.__sgd_dual(j, y)
				else:
					break
			self.__w = np.dot(self.__alpha * y, X.T)
		else:
			self.__iter_num = 0
			for i in range(self.__max_iter):
				y_hat = self.__calc_y(X)
				j = self.__select_one_wrong(y, y_hat)
				if j:
					self.__iter_num = self.__iter_num + 1
					self.__sgd(j, X, y)
				else:
					break

	@time_record
	def predict(self, X):
		return np.sign(np.dot(self.__w.reshape(1, X.shape[1]), X.T) + self.__b).reshape((X.shape[0],))

	@time_record
	def __calc_y_dual(self, y):
		return np.sign(np.dot((self.__alpha * y).reshape((1, len(y))), self.__Gram) + self.__b).reshape((len(y),))
	
	@time_record
	def __calc_y(self, X):
		return np.sign(np.dot(self.__w.reshape(1, X.shape[0]), X) + self.__b).reshape((X.shape[1],))

	@time_record
	def __calc_Gram(self, X):
		return np.dot(X.T, X)

	@time_record
	def __select_one_wrong(self, y, y_hat):
		if list(np.where(y != y_hat)[0]):
			return np.random.choice(np.where(y != y_hat)[0])
		else:
			return None

	@time_record
	def __sgd_dual(self, j, y):
		self.__alpha[j] = self.__alpha[j] + self.__eta
		self.__b = self.__b + y[j]

	@time_record
	def __sgd(self, j, X, y):
		self.__w = self.__w + self.__eta * y[j] * X[:, j].reshape((len(self.__w),))
		self.__b = self.__b + y[j]



