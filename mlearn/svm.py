#!/usr/bin/env
# -*- coding: utf-8 -*-

'''

Support Vector Machine with SMO method

1. import "svm" from "MLWheel" module

	>>> from MLWheel import svm

2. initialize a Perceptron model

	>>> clf = svm.SVM(**params)

    features : Features' name.

    C: Penalty parameter. Decrease C can avoid overfitting.

    kernel: kernel function type, ['linear', 'rbf', 'poly', 'sigmoid'], default value: 'linear'
    
    max_iter: Max iteration num, default value: 100. Decrease max_iter can avoid overfitting.
    
    degree: Parameter for kernel functions.

    gamma: Parameter for kernel functions.

    coef0: Parameter for kernel functions.

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
	
	features_: Only linear kernel has this attribute. Dict type. [feature: coef]

	support_: Index of support vectors.

	support_vectors_: DataFrame of support vectors.

	params_: Params of the model.

    iter_num_: iteration number

'''

__author__ = 'YayoYY'

import numpy as np
import pandas as pd

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
			return [str(feature) + ': ' +  str(coef) for feature,coef in zip(self.__features, self.__coef)]
		else:
			return 'Error: Only linear kernel can output coef_'

	@property
	def support_(self):
		return self.__support

	@property
	def n_support_(self):
		return self.__n_support

	@property
	def support_vectors_(self):
		sv = pd.DataFrame(self.__X[:,self.__support].T, columns=self.__features, index=self.__support)
		return sv

	@property
	def params_(self):
		return self.__params

	@property
	def iter_num_(self):
		return self.__iter_num
	
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
		self.__iter_num = 0
		for i in range(self.__max_iter):
			g = self.__calc_g(self.__Gram)
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
			eta = self.__Gram[alpha_1, alpha_1] + self.__Gram[alpha_2, alpha_2] - 2*self.__Gram[alpha_1, alpha_2]
			alpha_2_new = self.__alpha[alpha_2] + self.__y[alpha_2]*(E[alpha_1] - E[alpha_2])/eta
			if self.__y[alpha_1] == self.__y[alpha_2]:
				L = max([1, self.__alpha[alpha_1]+self.__alpha[alpha_2]-self.__C])
				H = min([self.__alpha[alpha_1]+self.__alpha[alpha_2], self.__C])
			else:
				L = max([0, self.__alpha[alpha_2]-self.__alpha[alpha_1]])
				H = min([self.__C, self.__C-self.__alpha[alpha_2]+self.__alpha[alpha_1]])
			alpha_2_old = self.__alpha[alpha_2]
			alpha_2_new = np.array([L, alpha_2_new, H])[np.array([alpha_2_new<L, alpha_2_new>=L and alpha_2_new <=H, alpha_2_new>H])][0]
			alpha_1_old = self.__alpha[alpha_1]
			alpha_1_new = self.__alpha[alpha_1] + self.__y[alpha_1]*self.__y[alpha_2]*(alpha_2_old-alpha_2_new)
			self.__alpha[alpha_1] = alpha_1_new
			self.__alpha[alpha_2] = alpha_2_new
			b1_new = -E[alpha_1] - self.__y[alpha_1]*self.__Gram[alpha_1,alpha_1]*(alpha_1_new-alpha_1_old)-self.__y[alpha_2]*self.__Gram[alpha_2,alpha_1]*(alpha_2_new-alpha_2_old)+self.__b
			b2_new = -E[alpha_2] - self.__y[alpha_2]*self.__Gram[alpha_2,alpha_2]*(alpha_2_new-alpha_2_old)-self.__y[alpha_1]*self.__Gram[alpha_1,alpha_2]*(alpha_1_new-alpha_1_old)+self.__b
			if b1_new  == b2_new:
				self.__b = b1_new
			else:
				self.__b = (b1_new + b2_new)/2
			self.__iter_num = self.__iter_num + 1
		self.__coef = np.dot((self.__alpha * self.__y).reshape(1, len(self.__y)), self.__X.T).reshape((self.__X.shape[0],))
		self.__support = np.where(self.__alpha == self.__C)[0]
		self.__n_support = {}
		self.__n_support['+1'] = np.sum(self.__y[self.__support] == 1)
		self.__n_support['-1'] = np.sum(self.__y[self.__support] == -1)


	def __calc_E(self):
		return self.__calc_g(self.__Gram) - self.__y

	def __calc_g(self, Gram):
		return (np.dot((self.__alpha * self.__y).reshape(1, len(self.__y)), Gram) + self.__b).reshape((Gram.shape[1],))

	def __calc_Gram(self, X1, X2):
		Gram = np.zeros((X1.shape[1], X2.shape[1]))
		if self.__kernel == 'rbf':
			for i in range(X1.shape[0]):
				Gram[i, :] = np.exp(self.__gamma * np.linalg.norm(X1[:,i].reshape((len(X1[:,i]),1)) - X2, keepdims=True, axis=0, ord=2))
			return Gram
		elif self.__kernel == 'poly':
			return np.power(self.__gamma * np.dot(X1.T, X2) + self.__coef0, self.__degree)
		elif self.__kernel == 'sigmoid':
			return np.tanh(self.__gamma * np.dot(X1.T, X2) + self.__coef0)
		else:
			return np.dot(X1.T, X2)
	