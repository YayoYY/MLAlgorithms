#!/usr/bin/env
# -*- coding: utf-8 -*-

__author__ = 'YayoYY'

import numpy as numpy

class SVM():

	def __init__(self, **params):
		self.__params = params
		self.__features = params.get('features', None)
		self.__C = params.get('C', 1.0)
		self.__kernel = params.get('kernel', 'linear')
		self.__max_iter = params.get('max_iter', 100)
		self.__degree = params.get('degree', 3)
		self.__gamma = params.get('gamma', 1/len(self.__features))
		self.__coef0 = params.get('coef0', 0)
		self.__b = 0

	@property
	def features_(self):
		if self.__kernel == 'linear':
			return [feature + ': ' +  coef for feature,coef in zip(self.__features, self.__coef)]
		else:
			return 'Only linear kernel can output coef_'

	@property
	def support_(self):
		return np.where(self.__alpha != 0)[0]

	@property
	def n_support_(self):
		return self.__n_support

	@property
	def support_vectors_(self):
		return self.__X[[self.support_]]

	@property
	def get_params(self):
		return self.__params

	def fit(self, X, y):
		self.__X = X.T
		self.__y = y
		self.__alpha = np.zeros(len(y))
		self.__Gram = self.__calc_Gram(self.__X, self.__X)
		self.__smo()


	def predict(self, X):
		Gram = self.__calc_Gram(self.__X, X.T)
		return np.sign(self.__calc_g(Gram))

	def __smo(self):
		for i in range(self.__max_iter):
			g = self.__calc_g(self.__X)
			yg = self.__y * g
			voi_1 = np.logical_and(self.__alpha == 0, yg < 1)
			voi_2 = np.logical_and(np.logical_and(self.__alpha > 0, self.__alpha < self.__C), yg != 1)
			voi_3 = np.logical_and(self.__alpha == self.__C, yg > 1)
			E = self.__calc_E()
			if np.any(voi_1) and np.any(voi_2) and np.any(voi_3) == False:
				break
			elif np.any(voi_2):
				max_voi = max(np.abs(yg[voi_2]))
				alpha_1 = np.random.choice(np.where(np.logical_and(voi_2, np.abs(yg) == max_voi))[0])
			elif np.any(voi_1):
				max_voi = max(np.abs(yg[voi_1]))
				alpha_1 = np.random.choice(np.where(np.logical_and(voi_1, np.abs(yg) == max_voi))[0])
			elif np.any(voi_3):
				max_voi = max(np.abs(yg[voi_3]))
				alpha_1 = np.random.choice(np.where(np.logical_and(voi_3, np.abs(yg) == max_voi))[0])
			if E[alpha_1] > 0:
				alpha_2 = np.random.choice(np.where(E == min(E))[0])
			else:
				alpha_2 = np.random.choice(np.where(E == max(E))[0])
			

	def __calc_E(self):
		return self.__calc_g(self.__Gram) - self.__y

	def __calc_g(self, Gram):
		return (np.dot((self.__alpha * self.__y).reshape(1, len(self.__y)), Gram) + self.__b).reshape((len(self.__y),))

	def __calc_Gram(self, X1, X2):
		if self.__kernel == 'rbf':
			pass
		elif self.__kernel == 'poly':
			pass
		elif self.__kernel == 'sigmoid':
			pass
		else:
			return np.dot(X1.T, X2)
	