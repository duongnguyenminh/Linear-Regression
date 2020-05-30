from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
one = np.ones((X.shape[0], 1)) # tra ve row matrix
Xbar = np.concatenate((one, X), axis =1) #noi 2 matrix one voi X voi axis =1

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185,2)
y0 = w_0 + w_1*x0
#
#
#
from sklearn import linear_model
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)
print((regr.coef_).T)
