# 1) Import libraries and modules
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

# 2) Load data from remote url
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

# Uncomment this to print first 5 rows of data
# print data.head()

# Uncomment this to print the data shape
# print data.shape

# Uncomment this to print data overview summary stats
# print data.describe()

# 3) Split into training and test sets using sklearn
# Separate target from training features
y = data.quality
X = data.drop('quality', axis=1)

# Split data into training and testing sets, 20% of data in test, with random state
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

# 4) Data preprocessing
# Preprocess the data using Scikit-Learn preprocessing, only for the training set
# X_train_scaled = preprocessing.scale(X_train)
# print X_train_scaled.std(axis=0)

# Better way, preprocess the data using Scikit-Learn Transformer API (training data + model same way, and sounds much cooler)
# First, fit the transformer API
# scaler = preprocessing.StandardScaler().fit(X_train)

# Apply that transformer to the training data
 #X_train_scaled = scaler.transform(X_train)
# print X_train_scaled.mean(axis=0)
# print X_train_scaled.std(axis=0)

# Then apply that same tansformer to test data
# X_test_scaled = scaler.transform(X_test)
# print X_test_scaled.mean(axis=0)
# print X_test_scaled.std(axis=0)

# An even better way, which sets up a cross validation pipeline that transforms that data using StandardScaler, and fits a model using RandomForestRegressor
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# 5) Declare hyperparameters that you want to tune
# List hyperparameters that can be tuned
# print pipeline.get_params()

# Declare which hyperparameters to tune through cross-validation
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# 6) Tune the model using a cross-validation pipeline
# Split data into folds (equal parts)
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
# Fit the model
clf.fit(X_train, y_train)
# Print out the best parameters results
# print clf.best_params_

# 7) Refit the model using the entire training set via GridSearchCV, for performance improvements
# Check if model will be retrained (if result from below is "True" you are good)
# print clf.refit

# 8) Evaluate the model pipeline on the test data
# Predict new data set
y_pred = clf.predict(X_test)

# Use metrics from before to evaluate performance of the model
# For R-squared
# print r2_score(y_test, y_pred)

# For mean squared error (MSE)
print mean_squared_error(y_test, y_pred)

# 9) Save model for future and load it back
# joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')