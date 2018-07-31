"""
linear_regression1.py

This script should demo Linear Regression with the wine data.

Demo:
~/anaconda3/bin/python linear_regression1.py
"""

import pandas as pd
import numpy  as np
import sklearn
from sklearn import linear_model

wine_df = pd.read_csv('winequality-red.csv', sep=';')

wine_a = np.array(wine_df)

x_a = wine_a[:,:-1] # All col cept last.

y_a = wine_a[:,-1]  # Last col

linr_mod = linear_model.LinearRegression()

linr_mod.fit(x_a, y_a)

linr_mod.intercept_
linr_mod.coef_

linr_mod.predict(x_a[:4])

'''
>>> linr_mod.predict(x_a[:4])
array([5.03285045, 5.13787975, 5.20989474, 5.69385794])
'''

yhat_a = linr_mod.predict(x_a)

# I should use sklearn to calculate R-Squared:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
r2sklearn_f = sklearn.metrics.r2_score(y_a, yhat_a)
# I should see:
'''
>>> r2sklearn_f
0.36055170303868833
'''
# Which means that Linear Regression works poorly with this data.
# I'd like to see a value above 0.7

'bye'
