import glob
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro

def ReadLinesFromFile(sFileName):
    if os.path.exists(sFileName):
        with open(sFileName, "r") as f:
            return f.readlines()
    else:
        return None

results = []

for sFileName in glob.glob ("ErrorsByFold_SplitAge_*.csv"):
    listLines = ReadLinesFromFile(sFileName)
    
    if listLines and len(listLines) == 11:
        split_age = int (re.sub(r"ErrorsByFold_SplitAge_(\d+)\.csv", r"\1", sFileName))
        print('Reading', sFileName, "with split age", split_age)
        df = pd.read_csv(sFileName, delimiter=',')
        
        # Take the average of the stage 1 and stage 2 mae and rmse across all ten folds
        mae_1 = df['MAE1'].mean()
        mae_2 = df['MAE2'].mean()
        rmse_1 = df['RMSE1'].mean()
        rmse_2 = df['RMSE2'].mean()

        results += [[split_age, mae_1, mae_2, rmse_1, rmse_2]]
    
    else:
        print('Skipping incomplete file', sFileName)


df_errors = pd.DataFrame(results, columns=['SplitAge', 'MAE1', 'MAE2', 'RMSE1', 'RMSE2'])

# Plot MAE results
plt.figure(figsize=(12, 8))
#plt.plot(df_errors['SplitAge'], df_errors['MAE1'], label='Stage 1 MAE', color='grey')
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
#plt.plot(df_errors['SplitAge'], df_errors['RMSE1'], label='Stage 1 RMSE', color='grey')
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

