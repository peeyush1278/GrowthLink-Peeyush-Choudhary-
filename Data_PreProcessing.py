import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataset=pd.read_csv('tested.csv')

print(Dataset.describe())
print(Dataset.info())
print(Dataset.head())
print(Dataset.shape)
print(Dataset.isnull().sum())
print(Dataset.columns)
print(Dataset.dtypes)
print(Dataset.tail)