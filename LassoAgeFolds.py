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
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso

if len (sys.argv) != 2:
    print ("Required parameters: LambdaParameter")
    exit ()

sLambdaParameter = sys.argv [1]

# Optimized LASSO paramaters: 'alpha': 0.01
lambda_param = float(sLambdaParameter)

# Optimized LOWESS parameter: 'tau': 0.7
tau = 0.7


def processFold(nFold):
    sTrainFileName = str(nFold) + "_train_DNAm_matrix.pickle"
    sTestFileName  = str(nFold) + "_test_DNAm_matrix.pickle"

    test_DNAm_df = pd.read_pickle(sTestFileName)
    train_DNAm_df = pd.read_pickle(sTrainFileName)

    train_data = train_DNAm_df.values

    X = train_data[:, :-1]
    y = train_data[:, -1]

    model = Lasso(alpha=lambda_param) # define model

    model.fit(X, y) # fit model

    test_data = test_DNAm_df.values
    test_data = test_data[:, :-1]

    if( "SampleID" not in test_DNAm_df.columns ):
        test_DNAm_df = test_DNAm_df.reset_index()

    age_predictions = model.predict(test_data)

    predictions = {}
    for sample in test_DNAm_df.index:
        sample_id = str(int(test_DNAm_df.iloc[sample]["SampleID"]))
        predictions[int(sample_id)] = age_predictions[sample]

    full_prediction_df = pd.DataFrame.from_dict(predictions, orient='index', columns=["Age"])
    full_prediction_df.reset_index(inplace=True, names="Sample")

    predictions_df = pd.merge(test_DNAm_df, full_prediction_df, left_on="SampleID", right_on="Sample", how="inner")
    predictions_df = predictions_df[["Sample", "Age_x", "Age_y"]]
    predictions_df.rename(columns={"Age_x": "Real_Age", "Age_y": "Predicted_Age"}, inplace=True)

    predictions_df['Fold'] = nFold

    return predictions_df


def AppendPredictionsToFile(sFileName, wip_df, predictions_df):
    print("Wrote predictions to file " + sFileName)

    if wip_df is None:
        output_df = predictions_df
    else:
        output_df = pd.concat([wip_df, predictions_df])

    output_df.to_pickle(sFileName)


def ReadPredictionsFromFile(sFileName):
    if os.path.exists(sFileName):
        wip_df = pd.read_pickle(sFileName)
        return wip_df
    else:
        return None

sOutputFileName = "Predictions_LASSO.pickle"
wip_df = ReadPredictionsFromFile(sOutputFileName)

if wip_df is None:
    nStartFold = 0
elif wip_df['Fold'].max() == 9:
    print(sOutputFileName + " is full")
    sys.exit()
else:
    nStartFold = wip_df['Fold'].max() + 1

for nFold in range(nStartFold, 10):
    predictions_df = processFold(nFold)
    AppendPredictionsToFile(sOutputFileName, wip_df, predictions_df)
    wip_df = ReadPredictionsFromFile(sOutputFileName)
    print(wip_df)

