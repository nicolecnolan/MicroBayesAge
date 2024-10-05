import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random
from statsmodels.nonparametric.smoothers_lowess import lowess 
from loess.loess_2d import loess_2d
from enum import Enum
from matplotlib.lines import Line2D

if len (sys.argv) != 3:
    print ("Required parameters: InputFileName OutputFileName")
    exit ()

sInputFileName  = sys.argv [1]
sOutputFileName = sys.argv [2]

# Load and Reformat Data
def reformat_col_names(col_name):
    new_name = ""
    if(col_name < 0):
        new_name = 'Age'
    else:
        new_name = "cg"
        num_zeros = 8 - len(str(int(col_name)))
        for i in range(0,num_zeros):
            new_name = new_name + "0"
        new_name = new_name + (str(int(col_name)))
    return(new_name)

full_DNAm_df = pd.read_pickle(sInputFileName).transpose()
full_DNAm_df = full_DNAm_df.rename(columns=full_DNAm_df.iloc[0]).drop(full_DNAm_df.index[0]) # move row 0 cg site numbers up to column title
full_DNAm_df.rename(columns=reformat_col_names, inplace=True)
full_DNAm_df.drop(full_DNAm_df.index[0], inplace=True) # delete row 0 correlation values
full_DNAm_df = full_DNAm_df.assign(SampleID=range(1,len(full_DNAm_df)+1))
full_DNAm_df = full_DNAm_df.set_index("SampleID")
ages = full_DNAm_df['Age']
full_DNAm_df = full_DNAm_df.drop('Age', axis=1).assign(Age=ages)
print(full_DNAm_df.shape)
print(full_DNAm_df.head())

# remove NA values
full_DNAm_df.dropna(axis=0, subset='Age', inplace=True) # drop samples w/ NA values in Age column
full_DNAm_df.fillna(value=0, inplace=True)
print(full_DNAm_df.shape)
print(full_DNAm_df.head())

# write to pickle file
print("Writing to pickle file")
full_DNAm_df.to_pickle(sOutputFileName)
