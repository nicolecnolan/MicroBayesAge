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

if len (sys.argv) != 2:
    print ("Required parameters: TrainFileName")
    exit ()

sTrainFileName = sys.argv [1]

sYoungFileName = "young_" + sTrainFileName
sOldFileName   = "old_"   + sTrainFileName

sFullReferenceMatrixName  = sTrainFileName.replace('.pickle', '_reference_model')
sYoungReferenceMatrixName = sYoungFileName.replace('.pickle', '_reference_model')
sOldReferenceMatrixName   = sOldFileName.replace  ('.pickle', '_reference_model')

print("Loading training and testing data")
train_DNAm_df = pd.read_pickle(sTrainFileName)

young_df = train_DNAm_df[train_DNAm_df["Age"] <= 25]
print(young_df.shape)
old_df = train_DNAm_df[train_DNAm_df["Age"] > 25]
print(old_df.shape)

young_df.to_pickle(sYoungFileName)
old_df.to_pickle(sOldFileName)

# Construct Reference: Beta Values
from BayesAge import construct_reference

print("Constructing full reference")
construct_reference(training_matrix = sTrainFileName,
reference_name = sFullReferenceMatrixName,
output_path = "output_files/",
zero_met_replacement = 0.001,
one_met_replacement = 0.999,
min_age = 0,
max_age = 100,
age_step = 1,
tau = 0.7)

print("Constructing reference for young samples")
construct_reference(training_matrix = sYoungFileName,
reference_name = sYoungReferenceMatrixName,
output_path = "output_files/",
zero_met_replacement = 0.001,
one_met_replacement = 0.999,
min_age = 0,
max_age = 25,
age_step = 1,
tau = 0.7)

print("Constructing reference for old samples")
construct_reference(training_matrix = sOldFileName,
reference_name = sOldReferenceMatrixName,
output_path = "output_files/",
zero_met_replacement = 0.001,
one_met_replacement = 0.999,
min_age = 26,
max_age = 100,
age_step = 1,
tau = 0.7)
