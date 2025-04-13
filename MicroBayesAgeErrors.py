import os
import sys
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import random
from statsmodels.nonparametric.smoothers_lowess import lowess 
from loess.loess_2d import loess_2d
from enum import Enum
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from scipy.stats import beta
from scipy.stats import norm

if len (sys.argv) != 2:
    print ("Required Parameter: SplitAge")
    exit ()

split_age  = int(sys.argv [1])

beta_dist = False # otherwise will use normal distribution
Modes = Enum('Modes', ['AGE','METH', 'COMP','MULTI'])
var_regression_mode = Modes.METH
tau = 0.7
CpG_parameter = 16
min_age = 0
max_age = 100
age_step = 1

# Construct Reference: Variance and bin the samples into buckets of 1 yr
def construct_reference(sample_df, beta_ref_df, low_age_bound, top_age_bound, age_steps):

    highest_age = int(sample_df["Age"].max())
    num_bins = highest_age + 1 - low_age_bound
    bin_edges = np.linspace(sample_df["Age"].min(),sample_df["Age"].max()+1, num_bins + 1)
    sample_df["age_bin"] = pd.cut(sample_df["Age"], bins = bin_edges, labels = np.arange(start=low_age_bound, stop=highest_age+1), right=False)

    # fit the variance to a function of age and/or methylation using lo(w)ess
    var_ref = {"Age": age_steps}
    CpG_list = sample_df.columns
    num_rows = round(len(CpG_list)/4)

    for CpG_site in beta_ref_df.index:
        ages = np.array([])
        methylations = np.array([])
        variances = np.array([])
        for bin in range (0,num_bins):
            replicates = sample_df[sample_df["age_bin"]==bin]
            if len(replicates) > 0:
        
                variance = np.var(replicates[CpG_site])
                mean_methylation = np.mean(replicates[CpG_site])
                mean_age = np.mean(replicates["Age"])
        
                variances = np.append(variances, variance)
                methylations = np.append(methylations, mean_methylation)
                ages = np.append(ages, mean_age)

        ages_to_predict = np.array(age_steps)
        methylations_to_predict = np.array(beta_ref_df.loc[CpG_site][:len(age_steps)])

        if var_regression_mode == Modes.AGE:
            predicted_variance = lowess(variances, ages, xvals=ages_to_predict, frac=tau)

        elif var_regression_mode == Modes.METH:
            predicted_variance = lowess(variances, methylations, xvals=methylations_to_predict, frac=tau)
            smoothed_variance = lowess(variances, methylations, frac=tau)

            predicted_meth = lowess(methylations, ages, xvals=ages_to_predict, frac=tau)
            smoothed_meth = lowess(methylations, ages, frac=tau)

            z = np.polyfit(ages, methylations, 1)
            p = np.poly1d(z)

        elif var_regression_mode == Modes.MULTI:
            predicted_variance, wout = loess_2d(x=ages, y=methylations, z=variances, xnew=ages_to_predict, ynew=methylations_to_predict, frac=tau)
            
        var_ref[CpG_site] = predicted_variance

    # construct the reference dataframe
    var_ref_df = pd.DataFrame.from_dict(var_ref).set_index("Age").transpose()
    return var_ref_df


# Predict each sample
def predict_sample(sample, beta_ref_df, var_ref_df, age_steps):
    global bFirst # remove
    age_probabilities = []
    for age in age_steps:
        prob_list_one_age = []
        
        for CpG in beta_ref_df.index:

            x = sample[CpG]
            iAge = age - age_steps.start
            mean = beta_ref_df.loc[CpG].iloc[iAge]
            variance = var_ref_df.loc[CpG].iloc[iAge]

            # replace extreme values
            if variance <= 0:
                variance = 0.000001
            
            if beta_dist:
                # calculate the shape parameters
                a = (mean*(-(mean*mean) + mean - (variance)))/(variance)
                b = ((mean - 1)*((mean*mean)-mean+variance))/variance

                # replace extreme values
                if a < 0:
                    a = 0.000001
                if b < 0:
                    b = 0.000001
                
                # get the posterior probability
                prob_list_one_age.append(np.log(beta.pdf(x, a, b)))

            else:
                prob = norm.pdf(x, mean, math.sqrt(variance))
                prob_list_one_age.append(np.log(prob))

        # sum the posterior probability for each CpG site
        age_probabilities.append((np.sum(prob_list_one_age), age))
    # return the maximum likelihood age
    return max(age_probabilities)[1]


# Prediction Analysis
def calculate_error(test_df, prediction_df, sTitle):
    merged_df = pd.merge(test_df, prediction_df, left_on="SampleID", right_on="Sample", how="inner")
    mini_merged_df = merged_df[["Sample", "Age_x", "Age_y"]]

    x = mini_merged_df["Age_x"]
    y = mini_merged_df["Age_y"]
    r_squared = r2_score(x, y)

    model = sm.OLS(y, sm.add_constant(x))

    results = model.fit()

    a,b = np.polyfit(x, y, 1)
    y_fit = a*x+b

    # Plot deviance from best fit
    deviance = y - y_fit

    smoothed_deviance = lowess(deviance.to_numpy(), x.to_numpy(), frac=tau)

    z = np.polyfit(x, deviance, 1)
    p = np.poly1d(z)

    actual = mini_merged_df['Age_x'].tolist()
    predicted = mini_merged_df['Age_y'].tolist()

    n = len(actual)
    total = 0
    for i in range(n):
        total += abs(actual[i] - predicted[i])

    mae = total/n
    rmse = math.sqrt((total ** 2) / n)

    return mae, rmse


def processFold(nFold):

    sTrainFileName = str(nFold) + "_train_DNAm_matrix.pickle"
    sTestFileName  = str(nFold) + "_test_DNAm_matrix.pickle"

    sFullReferenceMatrixName  =            sTrainFileName.replace('.pickle', '_reference_model.pickle')
    sYoungReferenceMatrixName = "young_" + sTrainFileName.replace('.pickle', '_reference_model.pickle')
    sOldReferenceMatrixName   = "old_"   + sTrainFileName.replace('.pickle', '_reference_model.pickle')

    print ("Reading pickle file", sTrainFileName)
    train_DNAm_df = pd.read_pickle(sTrainFileName)

    print ("Reading pickle file", sTestFileName)
    test_DNAm_df = pd.read_pickle(sTestFileName)

    reference_data = "output_files/" + sFullReferenceMatrixName
    print ("Reading pickle file", sFullReferenceMatrixName)
    reference_df = pd.read_pickle(reference_data)

    full_beta_ref_df = reference_df.loc[reference_df["spearman_rank"].abs().nlargest(CpG_parameter).index]
    age_steps = range(min_age,max_age+1,age_step)

    full_var_ref_df = construct_reference(train_DNAm_df, full_beta_ref_df, min_age, max_age, age_steps)

    if "SampleID" not in test_DNAm_df.columns:
        test_DNAm_df = test_DNAm_df.reset_index()

    predictions = {}
    for sample in test_DNAm_df.index:
        sample_id = str(int(test_DNAm_df.iloc[sample]["SampleID"]))
        predictions[int(sample_id)] = predict_sample(test_DNAm_df.iloc[sample], full_beta_ref_df, full_var_ref_df, age_steps)

    full_prediction_df = pd.DataFrame.from_dict(predictions, orient='index', columns=["Age"])
    full_prediction_df.reset_index(inplace=True, names="Sample")

    mae_1, rmse_1 = calculate_error(test_DNAm_df, full_prediction_df, "All Patients")

    young_subset_test_df = test_DNAm_df.loc[test_DNAm_df["Age"] <= split_age]
    mask = full_prediction_df["Sample"].isin(young_subset_test_df["SampleID"])
    young_subset_prediction_df = full_prediction_df[mask]

    old_subset_test_df = test_DNAm_df.loc[test_DNAm_df["Age"] > split_age]
    mask = full_prediction_df["Sample"].isin(old_subset_test_df["SampleID"])
    old_subset_prediction_df = full_prediction_df[mask]

    # Creating subset of only young samples
    young_train_df = train_DNAm_df.loc[train_DNAm_df["Age"] <= split_age]

    young_reference_data = "output_files/" + sYoungReferenceMatrixName
    print ("Reading pickle file", sYoungReferenceMatrixName)
    young_reference_df = pd.read_pickle(young_reference_data)

    young_beta_ref_df = young_reference_df.loc[young_reference_df["spearman_rank"].abs().nlargest(CpG_parameter).index]
    age_steps = range(min_age,split_age+1,age_step)

    young_test_df = full_prediction_df.loc[full_prediction_df["Age"] <= split_age]

    mask = test_DNAm_df["SampleID"].isin(young_test_df["Sample"])
    young_test_df = test_DNAm_df[mask]

    # Predicting young samples
    young_var_ref_df = construct_reference(young_train_df, young_beta_ref_df, min_age, split_age, age_steps)

    if "SampleID" not in young_test_df.columns:
        young_test_df = young_test_df.reset_index()

    young_predictions = {}
    for sample in young_test_df.index:
        sample_id = str(int(young_test_df.loc[sample]["SampleID"]))
        young_predictions[int(sample_id)] = predict_sample(young_test_df.loc[sample], young_beta_ref_df, young_var_ref_df, age_steps)

    young_prediction_df = pd.DataFrame.from_dict(young_predictions, orient='index', columns=["Age"])
    young_prediction_df.reset_index(inplace=True, names="Sample")

    # Creating subset of only old samples
    old_train_df = train_DNAm_df.loc[train_DNAm_df["Age"] > split_age]

    old_reference_data = "output_files/" + sOldReferenceMatrixName
    print ("Reading pickle file", sOldReferenceMatrixName)
    old_reference_df = pd.read_pickle(old_reference_data)

    old_beta_ref_df = old_reference_df.loc[old_reference_df["spearman_rank"].abs().nlargest(CpG_parameter).index]
    age_steps = range(split_age+1,max_age+1,age_step)

    old_test_df = full_prediction_df[full_prediction_df["Age"] > split_age]

    mask = test_DNAm_df["SampleID"].isin(old_test_df["Sample"])
    old_test_df = test_DNAm_df[mask]

    # Predicting old samples
    old_var_ref_df = construct_reference(old_train_df, old_beta_ref_df, split_age+1, max_age, age_steps)

    if "SampleID" not in old_test_df.columns:
        old_test_df = old_test_df.reset_index()

    old_predictions = {}
    for sample in old_test_df.index:
        sample_id = str(int(old_test_df.loc[sample]["SampleID"]))
        old_predictions[int(sample_id)] = predict_sample(old_test_df.loc[sample], old_beta_ref_df, old_var_ref_df, age_steps)

    old_prediction_df = pd.DataFrame.from_dict(old_predictions, orient='index', columns=["Age"])
    old_prediction_df.reset_index(inplace=True, names="Sample")

    combined_prediction_df = pd.concat([young_prediction_df, old_prediction_df], sort=False)

    mae_2, rmse_2 = calculate_error(test_DNAm_df, combined_prediction_df, "All Patients")

    return (nFold, mae_1, rmse_1, mae_2, rmse_2)


def AppendLineToFile(sFileName, sLine):
    print("Wrote line to file " + sFileName)
    with open(sFileName, "a") as f:
        print(sLine, file=f)

def ReadLinesFromFile(sFileName):
    if os.path.exists(sFileName):
        with open(sOutputFileName, "r") as f:
            return f.readlines()
    else:
        return None

sOutputFileName = "ErrorsByFold_SplitAge_" + str(split_age) + ".csv"

listLines = ReadLinesFromFile(sOutputFileName)
if not listLines:
    AppendLineToFile(sOutputFileName, "Fold,MAE1,RMSE1,MAE2,RMSE2") # write CSV file column titles
    nStartFold = 0
elif len(listLines) == 11:
    print(sOutputFileName + " is full")
    sys.exit()
else:
    nStartFold = len(listLines) - 1

for nFold in range(nStartFold, 10):
    tup = processFold(nFold)
    sLine = ",".join([ str(obj) for obj in tup ])
    AppendLineToFile(sOutputFileName, sLine)
