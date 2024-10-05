import os
import sys
import pandas as pd
import numpy as np
from numpy import arange
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random
from statsmodels.nonparametric.smoothers_lowess import lowess 
from loess.loess_2d import loess_2d
from enum import Enum
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

if len (sys.argv) != 3:
    print ("Required parameters: TrainFileName TestFileName")
    exit ()

sTrainFileName = sys.argv [1]
sTestFileName  = sys.argv [2]

test_DNAm_df = pd.read_pickle(sTestFileName)
train_DNAm_df = pd.read_pickle(sTrainFileName)
print(test_DNAm_df.shape)
print(test_DNAm_df.head())
print(train_DNAm_df.shape)
print(train_DNAm_df.head())

train_data = train_DNAm_df.values
print("train_data has shape", train_data.shape)

X = train_data[:, :-1]
print("X has shape", X.shape)
y = train_data[:, -1]
print("y has shape", y.shape)

model = Lasso() # define model

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define grid
grid = dict()
grid['alpha'] = arange(0.01, 1, 0.01)

# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# perform the search
results = search.fit(X, y)

# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
