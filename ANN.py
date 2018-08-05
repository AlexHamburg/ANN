#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: ANN with Keras
@author: Oleksandr Trunov
"""
# Data Preparation
# Dependency
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Read Data
# This part is based on feature of a data
dataset = pd.read_csv("")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Cleaning, Dummy-Coding of Data
# This part is based on feature of a data
LabelEncoder_X = LabelEncoder()
X[:, 1] = LabelEncoder_X.fit_transform(X[:, 1])
X[:, 2] = LabelEncoder_X.fit_transform(X[:, 2])
OneHotEncoder_ = OneHotEncoder(categorical_features = [1])
X = OneHotEncoder_.fit_transform(X).toarray()
X = X[:, 1:]

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Stranard scale of features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating ANN
# Dependency
from keras.models import Sequential
from keras.layers import Dense, Dropout

# ANN (classifier problem)
classifier = Sequential()

# Adding input layer and hidden layer (hidden as rectified linear unit function)
# Dropout for reduce of overfitting ANN (p grows if you have overfitting after trying)
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(p = 0.1))

# Adding second hidden layer (hidden as rectified linear unit function)
# Dropout for reduce of overfitting ANN (p grows if you have overfitting after trying)
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(p = 0.1))

# Adding output layer (sigmoid function) - if we have more than 2 classes - soft max function
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

# Compile a model (binary outcome -> loss = binary uderscore cross entropy, 
# not binary -> categorical underscore cross entropy)
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fit with features
classifier.fit(x = X_train, y = Y_train, batch_size = 10, epochs = 100)

# Prediction and slit into true if over 50 percent
y_prediction = classifier.predict(X_test)
y_prediction = (y_prediction > 0.5)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_prediction)

# Evaluating of model k-Fold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    from keras.models import Sequential
    from keras.layers import Dense
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accur = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10 ,n_jobs = -1)
accur_mean = accur.mean()
variance = accur.std()

# Grid Search of parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    from keras.models import Sequential
    from keras.layers import Dense
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
param = {"batch_size": [10, 25, 32], "epochs": [100, 500], "optimizer": ["adam", "rmsprop"]}
gridSearchCV_Object = GridSearchCV(estimator = classifier, param_grid = param, scoring = "accuracy", cv = 10)
gridSearchCV_Object = gridSearchCV_Object.fit(X_train, Y_train)
best_param = gridSearchCV_Object.best_params_
best_accur = gridSearchCV_Object.best_score_