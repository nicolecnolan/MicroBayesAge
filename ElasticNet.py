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
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

if len (sys.argv) != 5:
    print ("Required parameters: LambdaParameter L1Ratio TrainFileName TestFileName")
    exit ()

sLambdaParameter = sys.argv [1]
sL1Ratio         = sys.argv [2]
sTrainFileName   = sys.argv [3]
sTestFileName    = sys.argv [4]

# Optimized ElasticNet paramaters: 'alpha': 0.001, 'l1_ratio': 0.72
lambda_param = float(sLambdaParameter)
l1_param = float(sL1Ratio)

# Optimized LOWESS parameter: 'tau': 0.7
tau = 0.7

# Load training dataset
train_DNAm_df = pd.read_pickle(sTrainFileName)
print(train_DNAm_df.shape)
print(train_DNAm_df.head())

train_data = train_DNAm_df.values
print("train_data has shape", train_data.shape)

X = train_data[:, :-1]
print("X has shape", X.shape)
y = train_data[:, -1]
print("y has shape", y.shape)

# Define the model
model = ElasticNet(alpha=lambda_param, l1_ratio=l1_param)

# Fit the model to the training data
model.fit(X, y)

# Load the test dataset
test_DNAm_df = pd.read_pickle(sTestFileName)
print(test_DNAm_df.shape)
print(test_DNAm_df.head())

test_data = test_DNAm_df.values
test_data = test_data[:, :-1]
print("test_data has shape", test_data.shape)

if( "SampleID" not in test_DNAm_df.columns ):
    test_DNAm_df = test_DNAm_df.reset_index()
    print(test_DNAm_df.head())

# Make age predictions
print("Calculating predictions")
age_predictions = model.predict(test_data)
print(age_predictions)

predictions = {}
for sample in test_DNAm_df.index:
    sample_id = str(int(test_DNAm_df.iloc[sample]["SampleID"]))
    predictions[int(sample_id)] = age_predictions[sample]

print(test_DNAm_df.head())
full_prediction_df = pd.DataFrame.from_dict(predictions, orient='index', columns=["Age"])
print(full_prediction_df.head())
full_prediction_df.reset_index(inplace=True, names="Sample")
print(full_prediction_df.head())

# Prediction Analysis
def calculate_error(test_df, prediction_df, sTitle):
    merged_df = pd.merge(test_df, prediction_df, left_on="SampleID", right_on="Sample", how="inner")
    mini_merged_df = merged_df[["Sample", "Age_x", "Age_y"]]
    print(mini_merged_df)

    x = mini_merged_df["Age_x"]
    y = mini_merged_df["Age_y"]
    r_squared = r2_score(x, y)

    model = sm.OLS(y, sm.add_constant(x))

    results = model.fit()
    print(results.params)
    print(results.summary())

    plt.rc("figure", figsize=(8, 5))
    plt.rc("font", size=12)

    plt.scatter(x=x, y=y, c='tan')

    plt.title("ElasticNet Predictions for " + sTitle, fontname = "Arial", fontsize = 20)
    plt.xlabel("Real Age")
    plt.ylabel("Predicted Age")
    plt.annotate("R^2 = " + str(round(r_squared, 3)), xy=(0.15, 0.80), xycoords='figure fraction')

    a,b = np.polyfit(x, y, 1)
    y_fit = a*x+b

    plt.annotate("y = " + str(round(a, 3)) + "x + " + str(round(b, 3)), xy=(0.15, 0.75), xycoords='figure fraction')

    plt.plot(x, y_fit, color='k')
    plt.show()

    # Plot deviance from best fit
    deviance = y - y_fit

    smoothed_deviance = lowess(deviance.to_numpy(), x.to_numpy(), frac=tau)

    fig = plt.figure(figsize=(10, 10), layout="tight")

    sub = fig.add_subplot()

    line1 = Line2D(smoothed_deviance[:,0], smoothed_deviance[:,1], color = "red", linewidth=3)
    sub.add_line(line1)
    sub.legend(["LOWESS fit"])

    sub.axhline(y=0, color='black', linewidth=1)
    
    sub.scatter(x, deviance, color = 'tan')

    sub.set_title("Deviation from Linear Fit of ElasticNet Predictions for " + sTitle, fontname = "Arial", fontsize = 20)

    sub.set_xlabel("Real Age", fontname = "Times New Roman", fontsize=20)
    sub.set_ylabel("Deviation from Linear Fit", fontname = "Times New Roman", fontsize=20)

    sub.set_ylim([-35, 25])

    z = np.polyfit(x, deviance, 1)
    p = np.poly1d(z)
    sub.plot(x, p(x), color="k", linewidth=3, linestyle="--")

    plt.show()

    actual = mini_merged_df['Age_x'].tolist()
    predicted = mini_merged_df['Age_y'].tolist()

    n = len(actual)
    total = 0
    for i in range(n):
        total += abs(actual[i] - predicted[i])

    error = total/n
    return error

# Residual Analysis
def plot_residuals(test_df, prediction_df, sTitle):
    merged_df = pd.merge(test_df, prediction_df, left_on="SampleID", right_on="Sample", how="inner")
    mini_merged_df = merged_df[["Sample", "Age_x", "Age_y"]]
    mini_merged_df['Residual'] = mini_merged_df['Age_x'] - mini_merged_df['Age_y']
    print(mini_merged_df)

    real_ages = mini_merged_df['Age_x'].to_numpy()
    residuals = mini_merged_df['Residual'].to_numpy()
    smoothed_residuals = lowess(residuals, real_ages, frac=tau)

    fig = plt.figure(figsize=(10, 10), layout="tight")

    x = mini_merged_df['Age_x']
    y = mini_merged_df['Residual']

    sub = fig.add_subplot()

    line1 = Line2D(smoothed_residuals[:,0], smoothed_residuals[:,1], color = "red", linewidth=3)
    sub.add_line(line1)
    sub.legend(["LOWESS fit"])

    sub.axhline(y=0, color='black', linewidth=1)
    
    sub.scatter(x,y, color = 'tan')

    sub.set_title("Residual Plot of ElasticNet Predictions for " + sTitle + " Patients", fontname = "Arial", fontsize = 20)

    sub.set_xlabel("Real Age", fontname = "Times New Roman", fontsize=20)
    sub.set_ylabel("Residuals", fontname = "Times New Roman", fontsize=20)

    sub.set_ylim([-35, 25])
    
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    sub.plot(x, p(x), color="k", linewidth=3, linestyle="--")

    plt.show()

print("Mean absolute error for all patients: " + str(calculate_error(test_DNAm_df, full_prediction_df, "All Patients")))

plot_residuals(test_DNAm_df, full_prediction_df, "All")

