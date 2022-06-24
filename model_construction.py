#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:53:53 2022

@author: samuel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import  accuracy_score, confusion_matrix, recall_score, precision_score, mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import pickle

# -------------- Loading Data Set -------------- #

data = pd.read_csv('/Users/samuel/Desktop/Stroke_Prediction/health.csv')

# -------------- Preprocess Data -------------- #

def preprocess(df):
    
    df = df.copy()
    
    # Drop id column
    df = df.drop('id', axis=1)
    
    # Replace null values of bmi with median
    df['bmi'] = df['bmi'].fillna( df['bmi'].median() )
    
    # One hot encoding
    for i in ['gender', 'work_type', 'smoking_status']:
        df = pd.concat([ df, pd.get_dummies( df[i] ) ], axis=1)
        df = df.drop( i, axis=1 ) # Drop columns not needed after encoding
    
    # Binary encoding
    df['ever_married'] = df.ever_married.apply( lambda x: 1 if x == 'Yes' else 0 )
    df['Residence_type'] = df.Residence_type.apply( lambda x: 1 if x == 'Urban' else 0 )

    # Create X and y for training
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    return X, y, df

X, y, df = preprocess( data )

# Handling class imbalance within data using synthetic minority oversampling technique (SMOTE)

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# -------------- Splitting Data/Dropping Variables -------------- #

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=1)

X_train = X_train.drop(['Male', 'Female', 'Other', 'Govt_job', 'Never_worked', 'Private', 'Self-employed', 'Residence_type', 'Unknown', 'ever_married'], axis=1)
X_test = X_test.drop(['Male', 'Female', 'Other', 'Govt_job', 'Never_worked', 'Private', 'Self-employed', 'Residence_type', 'Unknown', 'ever_married'], axis=1)


# -------------- Model Fitting -------------- #

models = {'   K Nearest Neighbors': KNeighborsClassifier(),
          '         Random Forest': RandomForestClassifier(random_state=1),
          'Support Vector Machine': SVC(random_state=1, probability=True)
         }

for method, model in models.items():
    model.fit(X_train, y_train)

for method, model in models.items():
    
    prediction = model.predict(X_test)
    recall = recall_score(y_test, prediction)*100
    precision = precision_score(y_test, prediction)*100
    probability = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, prediction)
    auc = roc_auc_score(y_test, probability)
    
    print(method + ' Recall rate: {:.3f}% \n'.format(recall) + method + ' Precision rate: {:.3f}% \n'.format(precision) 
          + 'Area under ROC curve: {} \n'.format(auc) + method + ' Accuracy: {:.3f} \n'.format(acc))
    
    sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt='g', cmap='Oranges')
    plt.show()
    
    prob_prediction = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, prob_prediction)
    
    plt.plot(fpr, tpr, marker='.', label=method)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()

# Appears that Random Forest Classifier provides best results.
# Based on precision score and recall rate

# -------------- Checking for overfitting -------------- #
fr = RandomForestClassifier(random_state=1)
fr.fit(X_train, y_train)

kf = KFold(n_splits=10)
score = cross_val_score(fr, X_test, y_test, cv=kf)
print("\n Cross Validation Scores for " + method + " are {}".format(score))
print("Average Cross Validation Scores are {} \n".format(score.mean()))

# -------------- Final model -------------- #

final_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, min_samples_split=2, random_state=1)
final_model.fit(X_train, y_train)

pred = final_model.predict(X_test)

accuracy_score(y_test, pred)
precision_score(y_test, pred)
recall_score(y_test, pred)

mean_absolute_error(y_test, pred)

# -------------- Pickling Model -------------- #

pickl = {'model': final_model}
pickle.dump(pickl, open('stroke_model' + '.p', 'wb'))

file_name = 'stroke_model.p'
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']


model.predict(np.array(X_test.iloc[142,:]).reshape(1,-1))

list(X_test.iloc[1,:])



