import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

Dataset=pd.read_csv('tested.csv')
#showing the Dataset information
print(Dataset.info())

# showing the total null value in each column
print(Dataset.isnull().sum())

#filling the missing value
Dataset['Age'].fillna(Dataset['Age'].median(), inplace=True)

#showing the total null value in each column
print(Dataset.isnull().sum())

#Now doing the same for fare
Dataset['Fare'].fillna(Dataset['Fare'].median(), inplace=True)

#showing the total null value in each column
print(Dataset.isnull().sum())

#so now we are having more no of cabin null value so we just drop the cabin column from dataset
Dataset.drop('Cabin', axis=1, inplace=True)

#checking if the column is droped or not
print(Dataset.columns)

#converting categorical data into the numerical data 
Dataset['Sex'] = Dataset['Sex'].map({'male': 0, 'female': 1})

#printing the dataset
print(Dataset['Sex'])

Dataset=pd.get_dummies(Dataset,columns=['Embarked'],drop_first=True)

# Extract title from Name 
Dataset['Title'] = Dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplify and encode titles
Dataset['Title'] = Dataset ['Title'].replace(['Lady', 'Countess','Capt','Col','Don','Dr', 
                                   'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
Dataset['Title'] = Dataset ['Title'].replace(['Mlle', 'Ms'], 'Miss')
Dataset['Title'] = Dataset ['Title'].replace('Mme', 'Mrs')

# Map titles
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
Dataset['Title'] = Dataset ['Title'].map(title_mapping).fillna(0)

print(Dataset['Title'])

#dropping irrelevent columns
Dataset.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

print(Dataset.info())

#creating numerical col
numeric_cols=Dataset.select_dtypes(include=['int64','float64']).columns

#now creating the boxplot for looking for outliers in data
Dataset[numeric_cols].boxplot()
plt.show()

#creating the histogram
Dataset.hist(bins=50, figsize=(12,8))
plt.show()

#creating scatter plot
scatter_matrix(Dataset[numeric_cols], figsize=(12,8))
plt.show()


