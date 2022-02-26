import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

autodata = pd.read_csv('../datas/Auto.csv',na_values='?')
autodata.head()

autodata.describe()

#Quantitative predictor
quantitative_predictors = autodata.select_dtypes(include=['number']).columns
quantitative_predictors

#Qualitative predictor
qualitative_predictors= autodata.select_dtypes(exclude=['number']).columns
qualitative_predictors

#Range of each quantitative predictor
autodata.describe().loc['max']-autodata.describe().loc['min']

#Mean and standard deviation of each quantitative predictor
#mean
autodata.describe().loc['mean']
#standard deviation 
autodata.describe().loc['std']

#Range, mean, and standard deviation of each predictor when remove the 10th through 85th observation
#remove the 10th through 85th observations.
autodata_subset = autodata.drop(autodata.index[10:85])
autodata_subset.describe()
#range
autodata_subset.describe().loc['max'] - autodata_subset.describe().loc['min']
#mean
autodata_subset.describe().loc['mean']
#std
autodata_subset.describe().loc['std']

#Investigate the predictors graphically
scatter_matrix(autodata,figsize=(12,12))