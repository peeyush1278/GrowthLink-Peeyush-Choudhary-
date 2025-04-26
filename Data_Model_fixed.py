import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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
    def __init__(self):
        self.embarked_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.sex_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.embarked_cols = ["C", "S", "Q"]
        self.sex_cols = ["Female", "Male"]

    def fit(self,X,y=None):
        self.embarked_encoder.fit(X[['Embarked']])
        self.sex_encoder.fit(X[['Sex']])
        return self
    
    def transform(self,X):
        X = X.copy()
        embarked_matrix = self.embarked_encoder.transform(X[['Embarked']])
        for i, col in enumerate(self.embarked_cols):
            X[col] = embarked_matrix[:, i]
        # Add a column for missing Embarked values if any
        if embarked_matrix.shape[1] < len(self.embarked_cols):
            for col in self.embarked_cols[embarked_matrix.shape[1]:]:
                X[col] = 0

        sex_matrix = self.sex_encoder.transform(X[['Sex']])
        for i, col in enumerate(self.sex_cols):
            X[col] = sex_matrix[:, i]
        return X

class FeatureDropper(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X.drop(['Sex','Embarked','Name','Ticket','Cabin'],axis=1, errors='ignore')

pipeline=Pipeline([("ageimputer",AgeImputer()),("featureencoder",FeatureEncoder()),("featuredropper",FeatureDropper())])

train_set=pipeline.fit_transform(train_set)
test_set=pipeline.transform(test_set)

X=train_set.drop('Survived',axis=1)
y=train_set['Survived']

scaler=StandardScaler()
X_data=scaler.fit_transform(X)
y_data=y.to_numpy()

clf=RandomForestClassifier()

param_grid=[
    {"n_estimators":[10,100,200,500],"max_depth":[None,5,10],"min_samples_split":[2,3,4]}
]

grid_search=GridSearchCV(clf,param_grid,cv=3,scoring="accuracy",return_train_score=True)
grid_search.fit(X_data,y_data)
final_clf=grid_search.best_estimator_
print(final_clf)

X_test=test_set.drop('Survived',axis=1)
y_test=test_set['Survived']

X_data_test=scaler.transform(X_test)
y_data_test=y_test.to_numpy()

print(final_clf.score(X_data_test,y_data_test))
