import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
df=pd.read_csv('tested.csv')
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2)
for train_index,test_index in split.split(df,df[['Survived','Pclass','Sex']]):
    train_set=df.loc[train_index]
    test_set=df.loc[test_index]

class AgeImputer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        imputer=SimpleImputer(strategy='mean')
        X['Age']=imputer.fit_transform(X[['Age']])
        return X

class FeatureEncoder(BaseEstimator,TransformerMixin):

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        encoder=OneHotEncoder()
        matrix=encoder.fit_transform(X[['Embarked']]).toarray()

        col_names=["C","S","Q","N"]

        for i in range(len(matrix.T)):
            X[col_names[i]]=matrix.T[i]
        
        col_names = ["Female", "Male"]
        matrix=encoder.fit_transform(X[['Sex']]).toarray()
        for i in range(len(matrix.T)):
            X[col_names[i]] = matrix.T[i]

        return X

class FeatureDropper(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X.drop(['Sex','N','Embarked','Name','Ticket','Cabin'],axis=1, errors='ignore')

pipeline=Pipeline([("ageimputer",AgeImputer()),("featureencoder",FeatureEncoder()),("featuredropper",FeatureDropper())])

train_set=pipeline.fit_transform(train_set)
test_set=pipeline.fit_transform(test_set)

