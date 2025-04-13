import glob
import os
import re
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

if len (sys.argv) > 1:
    sPrefix = str(sys.argv [1])
else:
    sPrefix = ""


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

for sFileName in glob.glob (sPrefix + "Predictions_SplitAge_*.pickle"):
    predictions_df = ReadFile(sFileName)
    
    if predictions_df is not None and predictions_df['Fold'].max() == 9:
        predictions_df.sort_values(by=['Real_Age'], inplace=True)
        
        if sPrefix == "f":
            split_age = int (re.sub(r"fPredictions_SplitAge_(\d+)\.pickle", r"\1", sFileName))
        elif sPrefix == "m":
            split_age = int (re.sub(r"mPredictions_SplitAge_(\d+)\.pickle", r"\1", sFileName))
        else:
            split_age = int (re.sub(r"Predictions_SplitAge_(\d+)\.pickle", r"\1", sFileName))
        
        print('Reading', sFileName, "with split age", split_age)
        
        real_ages = predictions_df['Real_Age']
        min_age = real_ages.min()
        max_age = real_ages.max()

        # Calculate the stage 1 and stage 2 mae and rmse across all ten folds
        mae_1, rmse_1, mbe_1, j_mae_1, j_rmse_1, j_mbe_1, s_mae_1, s_rmse_1, s_mbe_1 = calculate_error(predictions_df['Real_Age'], predictions_df['Predicted_Age_1'], split_age)
        mae_2, rmse_2, mbe_2, j_mae_2, j_rmse_2, j_mbe_2, s_mae_2, s_rmse_2, s_mbe_2 = calculate_error(predictions_df['Real_Age'], predictions_df['Predicted_Age_2'], split_age)

        errors += [[split_age, mae_1, mae_2, rmse_1, rmse_2, mbe_1, mbe_2]]
        junior_errors += [[split_age, j_mae_1, j_mae_2, j_rmse_1, j_rmse_2, j_mbe_1, j_mbe_2]]
        senior_errors += [[split_age, s_mae_1, s_mae_2, s_rmse_1, s_rmse_2, s_mbe_1, s_mbe_2]]

    else:
        print('Skipping incomplete file', sFileName)

df_errors = pd.DataFrame(errors, columns=['SplitAge', 'MAE1', 'MAE2', 'RMSE1', 'RMSE2', 'MBE1', 'MBE2'])
df_junior_errors = pd.DataFrame(junior_errors, columns=['SplitAge', 'MAE1', 'MAE2', 'RMSE1', 'RMSE2', 'MBE1', 'MBE2'])
df_senior_errors = pd.DataFrame(senior_errors, columns=['SplitAge', 'MAE1', 'MAE2', 'RMSE1', 'RMSE2', 'MBE1', 'MBE2'])

# Plot MAE results
plt.figure(figsize=(12, 8))
min_index = df_errors['MAE2'].idxmin()
plt.plot(df_errors['SplitAge'], df_errors['MAE2'], label='Stage 2 MAE', color='black')
plt.plot(df_errors['SplitAge'].iloc[min_index], df_errors['MAE2'].iloc[min_index], label='Lowest MAE', marker='x', markersize=10, color='black')

# Add labels and title
plt.xlabel('Junior Cohort Maximum Age (Inclusive)')
plt.xticks(df_errors['SplitAge'])
plt.ylabel('Age Prediction MAE')
plt.title('Prediction MAE by Cohort Division Age')

# Add legend
plt.legend()

# Display the plot
plt.show()

# Plot RMSE results
plt.figure(figsize=(12, 8))
min_index = df_errors['RMSE2'].idxmin()
plt.plot(df_errors['SplitAge'], df_errors['RMSE2'], label='Stage 2 RMSE', color='black')
plt.plot(df_errors['SplitAge'].iloc[min_index], df_errors['RMSE2'].iloc[min_index], label='Lowest RMSE', marker='x', markersize=10, color='black')

# Add labels and title
plt.xlabel('Junior Cohort Maximum Age (Inclusive)')
plt.xticks(df_errors['SplitAge'])
plt.ylabel('Age Prediction RMSE')
plt.title('Prediction RMSE by Cohort Division Age')

# Add legend
plt.legend()

# Display the plot
plt.show()

# Plot MBE results
plt.figure(figsize=(12, 8))
min_index = df_errors['MBE2'].idxmin()
plt.plot(df_errors['SplitAge'], df_errors['MBE2'], label='Stage 2 MBE2', color='black')
plt.plot(df_errors['SplitAge'].iloc[min_index], df_errors['MBE2'].iloc[min_index], label='Lowest MBE', marker='x', markersize=10, color='black')

# Add labels and title
plt.xlabel('Junior Cohort Maximum Age (Inclusive)')
plt.xticks(df_errors['SplitAge'])
plt.ylabel('Age Prediction MBE')
plt.title('Prediction MBE by Cohort Division Age')

# Add legend
plt.legend()

# Display the plot
plt.show()

print("\nOverall Statistics")
print(df_errors)

print("\nJunior Cohort Statistics")
print(df_junior_errors)

print("\nSenior Cohort Statistics")
print(df_senior_errors)
