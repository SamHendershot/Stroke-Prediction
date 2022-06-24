#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 17:27:33 2022

@author: samuel
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


data = pd.read_csv('/Users/samuel/Desktop/Stroke Prediction/health.csv')

def preprocess(df):
    df = df.copy()
    
    # Drop id column
    df = df.drop('id', axis=1)
    
    # Replace null values of bmi with median
    df['bmi'] = df['bmi'].fillna( df['bmi'].median() )
    
    # Create X and y for training
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    # Create object classes from binary values
    for i in ['hypertension', 'heart_disease']:
        df[i] = df[i].apply(lambda x: 'yes' if x == 1 else 'no')
    
    # Split data into categorical and numeric
    
    df_num = df._get_numeric_data()
    df_cat = df.drop( df_num.columns, axis=1 )
    
    return X, y, df_num, df_cat

X, y, num, cat = preprocess( data )

# ------------- PipeLine ------------- #

class removeOutlier(BaseEstimator, TransformerMixin):
    def __init__( self, outliers=['bmi', 'avg_glucose_level'] ):
        self.outliers = outliers
        
    def fit( self, df ):
        return self
    
    def transform( self, df ):
        pass
            

class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__( self, feature_names ):
        self._feature_names = feature_names
    
    def fit( self, X, y = None ):
        return self
    
    def transform( self, X, y = None ):
        return X[self._feature_names].values



cat_pipeline = Pipeline([ ('cat_selector', FeatureSelector(cat) ),
                          ('one_hot_encoder', OneHotEncoder(sparse=False))])

num_pipeline = Pipeline([ ('num_selector', FeatureSelector(num) ),
                          ('std_scaler', StandardScaler()) ])

full_pipeline = FeatureUnion( transformer_list= [ ( 'num_pipeline', num_pipeline ), ( 'cat_pipeline', cat_pipeline )])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

X_train_processed = full_pipeline.fit_transform( X_train )
X_test_processed = full_pipeline.transform( X_test )




