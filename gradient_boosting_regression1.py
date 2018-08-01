"""
gradient_boosting_regression1.py

This script should demo Gradient Boosting Regression with the wine data.

Ref:
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
https://stats.stackexchange.com/questions/173390/gradient-boosting-tree-vs-random-forest
https://medium.com/@aravanshad/gradient-boosting-versus-random-forest-cfa3fa8f0d80
http://fastml.com/what-is-better-gradient-boosted-trees-or-random-forest/

Demo:
~/anaconda3/bin/python gradient_boosting_regression1.py
"""

import pandas as pd
import numpy  as np
import sklearn
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

wine_df = pd.read_csv('winequality-red.csv', sep=';')

wine_a = np.array(wine_df)

x_a = wine_a[:,:-1] # All col cept last.

y_a = wine_a[:,-1]  # Last col

gb_mod = GradientBoostingRegressor()

gb_mod.fit(x_a, y_a)

gb_mod.feature_importances_
# I should see:
'''
>>> gb_mod.feature_importances_
array([0.09123461, 0.12110511, 0.08286125, 0.07060833, 0.07465385,
       0.04954871, 0.1092546 , 0.08724339, 0.0687083 , 0.10210527,
       0.14267659])
''' 

# I should look at 4 predictions:
gb_mod.predict(x_a[:4])

'''
>>> gb_mod.predict(x_a[:4])
array([5.10369558, 5.21892487, 5.3282596 , 5.61580978])
'''

# I should get more predictions:
yhat_a = gb_mod.predict(x_a)

# I should use sklearn to calculate R-Squared:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
r2sklearn_f = sklearn.metrics.r2_score(y_a, yhat_a)
# I should see:
'''
>>> r2sklearn_f
0.6097015222504552
'''
# Which means that GradientBoostingRegressor is probably better than Linear Regression
# assuming the training data is similar to the test data.
# Still, I'd like to see a value above 0.7.

'bye'
