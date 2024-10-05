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

if len (sys.argv) != 4:
    print ("Required parameters: InputFileName TrainFileName TestFileName")
    exit ()

sInputFileName  = sys.argv [1]
sTrainFileName  = sys.argv [2]
sTestFileName   = sys.argv [3]

full_DNAm_df = pd.read_pickle(sInputFileName)

# split into test/train
print("Splitting data into testing and training groups")

df_50_1 = full_DNAm_df.sample(frac=0.50)
df_50_2 = full_DNAm_df.drop(df_50_1.index, axis=0)

test_DNAm_df_1 = df_50_1.sample(frac=0.20)
test_DNAm_df_2 = df_50_2.sample(frac=0.20)

df_40_1 = df_50_1.drop(test_DNAm_df_1.index, axis=0)
df_40_2 = df_50_2.drop(test_DNAm_df_2.index, axis=0)

df_20_1 = df_40_1.sample(frac=0.50)
df_20_2 = df_40_2.sample(frac=0.50)

df_20_3 = df_40_1.drop(df_20_1.index, axis=0)
df_20_4 = df_40_2.drop(df_20_2.index, axis=0)

test_DNAm_df_3 = df_20_1.sample(frac=0.50)
test_DNAm_df_4 = df_20_1.drop(test_DNAm_df_3.index, axis=0)

test_DNAm_df_5 = df_20_2.sample(frac=0.50)
test_DNAm_df_6 = df_20_2.drop(test_DNAm_df_5.index, axis=0)

test_DNAm_df_7 = df_20_3.sample(frac=0.50)
test_DNAm_df_8 = df_20_3.drop(test_DNAm_df_7.index, axis=0)

test_DNAm_df_9 = df_20_4.sample(frac=0.50)
test_DNAm_df_0 = df_20_4.drop(test_DNAm_df_9.index, axis=0)

train_DNAm_df_1 = full_DNAm_df.drop(test_DNAm_df_1.index, axis=0)
train_DNAm_df_2 = full_DNAm_df.drop(test_DNAm_df_2.index, axis=0)
train_DNAm_df_3 = full_DNAm_df.drop(test_DNAm_df_3.index, axis=0)
train_DNAm_df_4 = full_DNAm_df.drop(test_DNAm_df_4.index, axis=0)
train_DNAm_df_5 = full_DNAm_df.drop(test_DNAm_df_5.index, axis=0)
train_DNAm_df_6 = full_DNAm_df.drop(test_DNAm_df_6.index, axis=0)
train_DNAm_df_7 = full_DNAm_df.drop(test_DNAm_df_7.index, axis=0)
train_DNAm_df_8 = full_DNAm_df.drop(test_DNAm_df_8.index, axis=0)
train_DNAm_df_9 = full_DNAm_df.drop(test_DNAm_df_9.index, axis=0)
train_DNAm_df_0 = full_DNAm_df.drop(test_DNAm_df_0.index, axis=0)

# print to pickle files
print("Printing training and testing data to pickle files")

train_DNAm_df_1.to_pickle("1_" + sTrainFileName)
train_DNAm_df_2.to_pickle("2_" + sTrainFileName)
train_DNAm_df_3.to_pickle("3_" + sTrainFileName)
train_DNAm_df_4.to_pickle("4_" + sTrainFileName)
train_DNAm_df_5.to_pickle("5_" + sTrainFileName)
train_DNAm_df_6.to_pickle("6_" + sTrainFileName)
train_DNAm_df_7.to_pickle("7_" + sTrainFileName)
train_DNAm_df_8.to_pickle("8_" + sTrainFileName)
train_DNAm_df_9.to_pickle("9_" + sTrainFileName)
train_DNAm_df_0.to_pickle("0_" + sTrainFileName)

test_DNAm_df_1.to_pickle("1_" + sTestFileName)
test_DNAm_df_2.to_pickle("2_" + sTestFileName)
test_DNAm_df_3.to_pickle("3_" + sTestFileName)
test_DNAm_df_4.to_pickle("4_" + sTestFileName)
test_DNAm_df_5.to_pickle("5_" + sTestFileName)
test_DNAm_df_6.to_pickle("6_" + sTestFileName)
test_DNAm_df_7.to_pickle("7_" + sTestFileName)
test_DNAm_df_8.to_pickle("8_" + sTestFileName)
test_DNAm_df_9.to_pickle("9_" + sTestFileName)
test_DNAm_df_0.to_pickle("0_" + sTestFileName)

