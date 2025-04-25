import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

Dataset=pd.read_csv('tested.csv')
#showing the Dataset information
print(Dataset.info())


