import glob
import os
import re
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

if len (sys.argv) < 3:
    print ("Required Parameter: Filename CutoffAge")
    exit ()

sFileName = str(sys.argv [1])
split_age = int(sys.argv [2])

def ReadFile(sFileName):
    if os.path.exists(sFileName):
        with open(sFileName, "r") as f:
            return pd.read_pickle(sFileName)
    else:
        return None

# Error Analysis
def calculate_error(real_ages, predicted_ages, split_age):
    actual = real_ages.tolist()
    predicted = predicted_ages.tolist()

    n = len(actual)
    total_absolute_error = 0
    total_bias_error = 0
    total_squared_error = 0

    senior_bias_error = 0
    senior_absolute_error = 0
    senior_squared_error = 0

    junior_bias_error = 0
    junior_absolute_error = 0
    junior_squared_error = 0

    for i in range(n):
        error = predicted[i] - actual[i]
        total_bias_error += error
        total_absolute_error += abs(error)
        total_squared_error += error ** 2

        if actual[i] > split_age:
            senior_bias_error += error
            senior_absolute_error += abs(error)
            senior_squared_error += error ** 2
        else:
            junior_bias_error += error
            junior_absolute_error += abs(error)
            junior_squared_error += error ** 2


    mae = total_absolute_error / n
    rmse = math.sqrt(total_squared_error / n)
    mbe = total_bias_error / n

    j_mae = junior_absolute_error / n
    j_rmse = math.sqrt(junior_squared_error / n)
    j_mbe = junior_bias_error / n

    s_mae = senior_absolute_error / n
    s_rmse = math.sqrt(senior_squared_error / n)
    s_mbe = senior_bias_error / n

    return mae, rmse, mbe, j_mae, j_rmse, j_mbe, s_mae, s_rmse, s_mbe


errors = []
junior_errors = []
senior_errors = []
real_ages = pd.DataFrame()

predictions_df = ReadFile(sFileName)

if predictions_df is not None and predictions_df['Fold'].max() == 9:
    predictions_df.sort_values(by=['Real_Age'], inplace=True)
    
    print('Reading', sFileName)
    
    real_ages = predictions_df['Real_Age']
    min_age = real_ages.min()
    max_age = real_ages.max()
    # Calculate the stage 1 and stage 2 mae and rmse across all ten folds
    mae_1, rmse_1, mbe_1, j_mae_1, j_rmse_1, j_mbe_1, s_mae_1, s_rmse_1, s_mbe_1 = calculate_error(predictions_df['Real_Age'], predictions_df['Predicted_Age'], split_age)

    errors += [[mae_1, rmse_1, mbe_1]]
    junior_errors += [[j_mae_1, j_rmse_1, j_mbe_1]]
    senior_errors += [[s_mae_1, s_rmse_1, s_mbe_1]]
else:
    print(sFileName + " is incomplete")

df_errors = pd.DataFrame(errors, columns=['MAE1', 'RMSE1', 'MBE1'])
df_junior_errors = pd.DataFrame(junior_errors, columns=['MAE1', 'RMSE1', 'MBE1'])
df_senior_errors = pd.DataFrame(senior_errors, columns=['MAE1', 'RMSE1', 'MBE1'])

print("\nOverall Statistics")
print(df_errors)

print("\nJunior Cohort Statistics")
print(df_junior_errors)

print("\nSenior Cohort Statistics")
print(df_senior_errors)
