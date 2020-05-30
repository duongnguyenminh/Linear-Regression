# Linear-Regression
# using basic:

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
w=A^(-1)*b

# using linear_model :
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)
print((regr.coef_).T)
w=regr.coef_

