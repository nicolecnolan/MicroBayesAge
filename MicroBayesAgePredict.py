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

if len (sys.argv) != 3:
    print ("Required parameters: TrainFileName TestFileName")
    exit ()

sTrainFileName = sys.argv [1]
sTestFileName  = sys.argv [2]

sFullReferenceMatrixName  =            sTrainFileName.replace('.pickle', '_reference_model.pickle')
sYoungReferenceMatrixName = "young_" + sTrainFileName.replace('.pickle', '_reference_model.pickle')
sOldReferenceMatrixName   = "old_"   + sTrainFileName.replace('.pickle', '_reference_model.pickle')

beta_dist = False # otherwise will use normal distribution
Modes = Enum('Modes', ['AGE','METH', 'COMP','MULTI'])
var_regression_mode = Modes.METH
tau = 0.7
CpG_parameter = 16
min_age = 0
split_age = 25
max_age = 100
age_step = 1

main_color = 'gray' # all = gray, male = saddlebrown, female = mediumpurple
young_color = 'green' # all = green, male = darkorange, female = salmon
old_color = 'steelblue' # all = steelblue, male = darkgoldenrod, female = maroon

print ("Reading pickle file", sTrainFileName)
train_DNAm_df = pd.read_pickle(sTrainFileName)
print(train_DNAm_df.head())
print(train_DNAm_df.shape)

print ("Reading pickle file", sTestFileName)
test_DNAm_df = pd.read_pickle(sTestFileName)
print(test_DNAm_df.head())
print(test_DNAm_df.shape)

reference_data = "output_files/" + sFullReferenceMatrixName
print ("Reading pickle file", sFullReferenceMatrixName)
reference_df = pd.read_pickle(reference_data)
print(reference_df.head())
full_beta_ref_df = reference_df.loc[reference_df["spearman_rank"].abs().nlargest(CpG_parameter).index]
age_steps = range(min_age,max_age+1,age_step)


# Construct Reference: Variance
# Bin the samples into buckets of 1 yr

def construct_reference(sample_df, beta_ref_df, low_age_bound, top_age_bound):

    if low_age_bound == min_age and top_age_bound != max_age:
        color = young_color
    elif top_age_bound == max_age and low_age_bound != min_age:
        color = old_color
    else:
        color = main_color

    print("Grouping samples by age")
    highest_age = int(sample_df["Age"].max())
    num_bins = highest_age + 1 - low_age_bound
    print("num_bins =", num_bins)
    bin_edges = np.linspace(sample_df["Age"].min(),sample_df["Age"].max()+1, num_bins + 1)
    sample_df["age_bin"] = pd.cut(sample_df["Age"], bins = bin_edges, labels = np.arange(start=low_age_bound, stop=highest_age+1), right=False)
    print(sample_df.head())

    # fit the variance to a function of age and/or methylation using lo(w)ess
    var_ref = {"Age": age_steps}
    CpG_list = sample_df.columns
    num_rows = round(len(CpG_list)/4)

    fig = plt.figure(figsize=(15, 15), layout="tight")
    #fig.subplots_adjust(hspace = 5, top=1)
    plotnum = 0


    for CpG_site in beta_ref_df.index:
        ages = np.array([])
        methylations = np.array([])
        variances = np.array([])
        for bin in range (0,num_bins):
            replicates = sample_df[sample_df["age_bin"]==bin]
            if(len(replicates) > 0):
        
                variance = np.var(replicates[CpG_site])
                mean_methylation = np.mean(replicates[CpG_site])
                mean_age = np.mean(replicates["Age"])
        
                variances = np.append(variances, variance)
                methylations = np.append(methylations, mean_methylation)
                ages = np.append(ages, mean_age)

        ages_to_predict = np.array(age_steps)
        methylations_to_predict = np.array(beta_ref_df.loc[CpG_site][:len(age_steps)])

        plotnum = plotnum + 1

        if( var_regression_mode == Modes.AGE ):
            predicted_variance = lowess(variances, ages, xvals=ages_to_predict, frac=tau)

            plt.subplot(num_rows,4,plotnum)
            plt.scatter(ages_to_predict, predicted_variance)
            plt.title(CpG_site)
            plt.xlabel('Age')
            plt.ylabel('Variance')

        elif( var_regression_mode == Modes.METH ):
            predicted_variance = lowess(variances, methylations, xvals=methylations_to_predict, frac=tau)
            smoothed_variance = lowess(variances, methylations, frac=tau)

            predicted_meth = lowess(methylations, ages, xvals=ages_to_predict, frac=tau)
            smoothed_meth = lowess(methylations, ages, frac=tau)

            sub1 = fig.add_subplot(4,4,plotnum)
            #plt.scatter(methylations_to_predict, predicted_variance)
            sub1.set_title(CpG_site, fontname = "Arial", fontsize = 20)
            sub1.set_ylabel("Methylation Level", fontname = "Times New Roman", fontsize=20)
            sub1.set_xlabel("Real Age", fontname = "Times New Roman", fontsize=20)
            
            #sub1.plot(predicted_methylation[:,0], predicted_methylation[:,1], color = "palevioletred", label = "LOWESS Predicted Methylation")
            line1 = Line2D(smoothed_meth[:,0], smoothed_meth[:,1], color = "red", linewidth=3)
            sub1.add_line(line1)
            sub1.legend(["LOWESS fit"])
            sub1.scatter(ages, methylations, color = color, label = "Observed Age")
            z = np.polyfit(ages, methylations, 1)
            p = np.poly1d(z)
            sub1.plot(ages, p(ages), color="black", linewidth=3, linestyle="--")

        elif( var_regression_mode == Modes.MULTI ):
            predicted_variance, wout = loess_2d(x=ages, y=methylations, z=variances, xnew=ages_to_predict, ynew=methylations_to_predict, frac=tau)
            
            ax = fig.add_subplot(8,2, plotnum, projection='3d')
            ax.scatter(ages_to_predict, methylations_to_predict, predicted_variance)
            ax.set_title(CpG_site)
            ax.set_xlabel('Age')
            ax.set_ylabel('Methylation')
            ax.set_zlabel('Variance')
        
        var_ref[CpG_site] = predicted_variance
    plt.show()

    for k in var_ref.keys():
        print("var_ref["+str(k)+"] value has len = "+str(len(var_ref[k])))

    # construct the reference dataframe
    print("Initializing reference dataframe")
    var_ref_df = pd.DataFrame.from_dict(var_ref).set_index("Age").transpose()
    return var_ref_df


full_var_ref_df = construct_reference(train_DNAm_df, full_beta_ref_df, min_age, max_age)
print(full_var_ref_df.head())


# Age Prediction
from scipy.stats import beta
from scipy.stats import norm

import math
# Predict each sample
def predict_sample(sample, beta_ref_df, var_ref_df):
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
            if(variance <= 0):
                variance = 0.000001
            
            if(beta_dist):
                # calculate the shape parameters
                a = (mean*(-(mean*mean) + mean - (variance)))/(variance)
                b = ((mean - 1)*((mean*mean)-mean+variance))/variance

                # replace extreme values
                if(a < 0):
                    a = 0.000001
                if(b < 0):
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

if( "SampleID" not in test_DNAm_df.columns ):
    test_DNAm_df = test_DNAm_df.reset_index()
    print(test_DNAm_df.head())

print("Calculating predictions")
predictions = {}
for sample in test_DNAm_df.index:
    sample_id = str(int(test_DNAm_df.iloc[sample]["SampleID"]))
    predictions[int(sample_id)] = predict_sample(test_DNAm_df.iloc[sample], full_beta_ref_df, full_var_ref_df)

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

    # Plot Real Age vs Predicted Age
    plt.rc("figure", figsize=(8, 5))
    plt.rc("font", size=12)

    colors = np.where(y > 25, old_color, young_color)
    
    plt.scatter(x=x, y=y, c=colors)

    plt.title("BayesAge Predictions for " + sTitle, fontname = "Arial", fontsize = 20)
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

    if "Old" == sTitle:
        color = old_color
    elif "Young" == sTitle:
        color = young_color
    else:
        color = main_color

    smoothed_deviance = lowess(deviance.to_numpy(), x.to_numpy(), frac=tau)

    fig = plt.figure(figsize=(10, 10), layout="tight")

    sub = fig.add_subplot()

    line1 = Line2D(smoothed_deviance[:,0], smoothed_deviance[:,1], color = "red", linewidth=3)
    sub.add_line(line1)
    sub.legend(["LOWESS fit"])
    
    sub.axhline(y=0, color='black', linewidth=1)

    sub.scatter(x, deviance, color = color)

    sub.set_title("Deviation from Linear Fit of BayesAge Predictions for " + sTitle, fontname = "Arial", fontsize = 20)

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

    if "Old" == sTitle:
        color = old_color
    elif "Young" == sTitle:
        color = young_color
    else:
        color = main_color

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

    sub.scatter(x,y, color = color)

    sub.set_title("Residual Plot of BayesAge Predictions for " + sTitle + " Patients", fontname = "Arial", fontsize = 20)

    sub.set_xlabel("Real Age", fontname = "Times New Roman", fontsize=20)
    sub.set_ylabel("Residuals", fontname = "Times New Roman", fontsize=20)

    sub.set_ylim([-35, 25])

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    sub.plot(x, p(x), color="k", linewidth=3, linestyle="--")

    plt.show()


print("Mean absolute error for all patients: " + str(calculate_error(test_DNAm_df, full_prediction_df, "All Patients")))

young_subset_test_df = test_DNAm_df.loc[test_DNAm_df["Age"] <= split_age]
mask = full_prediction_df["Sample"].isin(young_subset_test_df["SampleID"])
young_subset_prediction_df = full_prediction_df[mask]
print("Mean absolute error for young patients: " + str(calculate_error(young_subset_test_df, young_subset_prediction_df, "Young Patients")))

old_subset_test_df = test_DNAm_df.loc[test_DNAm_df["Age"] > split_age]
mask = full_prediction_df["Sample"].isin(old_subset_test_df["SampleID"])
old_subset_prediction_df = full_prediction_df[mask]
print("Mean absolute error for old patients: " + str(calculate_error(old_subset_test_df, old_subset_prediction_df, "Old Patients")))

plot_residuals(test_DNAm_df, full_prediction_df, "All")

# Creating subset of only young samples
young_train_df = train_DNAm_df.loc[train_DNAm_df["Age"] <= split_age]

young_reference_data = "output_files/" + sYoungReferenceMatrixName
print ("Reading pickle file", sYoungReferenceMatrixName)
young_reference_df = pd.read_pickle(young_reference_data)
print(young_reference_df.head())
young_beta_ref_df = young_reference_df.loc[young_reference_df["spearman_rank"].abs().nlargest(CpG_parameter).index]
age_steps = range(min_age,split_age+1,age_step)

young_test_df = full_prediction_df.loc[full_prediction_df["Age"] <= split_age]
print(young_test_df.shape)
print(young_test_df.head())

mask = test_DNAm_df["SampleID"].isin(young_test_df["Sample"])
young_test_df = test_DNAm_df[mask]


#Predicting young samples
young_var_ref_df = construct_reference(young_train_df, young_beta_ref_df, min_age, split_age)
print(young_var_ref_df.head())

if( "SampleID" not in young_test_df.columns ):
    young_test_df = young_test_df.reset_index()
    print(young_test_df.head())

print("Predicting young samples")
print(young_beta_ref_df.info())
print("age_steps:", age_steps)
young_predictions = {}
for sample in young_test_df.index:
    sample_id = str(int(young_test_df.loc[sample]["SampleID"]))
    young_predictions[int(sample_id)] = predict_sample(young_test_df.loc[sample], young_beta_ref_df, young_var_ref_df)

young_prediction_df = pd.DataFrame.from_dict(young_predictions, orient='index', columns=["Age"])
young_prediction_df.reset_index(inplace=True, names="Sample")
print(young_prediction_df.head())
print("Mean absolute error for young samples: " + str(calculate_error(young_test_df, young_prediction_df, "Young Patients")))
plot_residuals(young_test_df, young_prediction_df, "Young")

#Creating subset of only old samples
old_train_df = train_DNAm_df.loc[train_DNAm_df["Age"] > split_age]

old_reference_data = "output_files/" + sOldReferenceMatrixName
print ("Reading pickle file", sOldReferenceMatrixName)
old_reference_df = pd.read_pickle(old_reference_data)
print(old_reference_df.head())
old_beta_ref_df = old_reference_df.loc[old_reference_df["spearman_rank"].abs().nlargest(CpG_parameter).index]
age_steps = range(split_age+1,max_age+1,age_step)

old_test_df = full_prediction_df[full_prediction_df["Age"] > split_age]
print(old_test_df.shape)
print(old_test_df.head())

mask = test_DNAm_df["SampleID"].isin(old_test_df["Sample"])
old_test_df = test_DNAm_df[mask]
print(old_test_df.shape)
print(old_test_df.head())

#Predicting old samples
old_var_ref_df = construct_reference(old_train_df, old_beta_ref_df, split_age+1, max_age)
print(old_var_ref_df.head())

if( "SampleID" not in old_test_df.columns ):
    old_test_df = old_test_df.reset_index()
    print(old_test_df.head())

print("Predicting old samples")
old_predictions = {}
for sample in old_test_df.index:
    sample_id = str(int(old_test_df.loc[sample]["SampleID"]))
    old_predictions[int(sample_id)] = predict_sample(old_test_df.loc[sample], old_beta_ref_df, old_var_ref_df)

old_prediction_df = pd.DataFrame.from_dict(old_predictions, orient='index', columns=["Age"])
old_prediction_df.reset_index(inplace=True, names="Sample")
print(old_prediction_df.head())
print("Mean absolute error for old samples: " + str(calculate_error(old_test_df, old_prediction_df, "Old Patients")))
plot_residuals(old_test_df, old_prediction_df, "Old")

combined_prediction_df = pd.concat([young_prediction_df, old_prediction_df], sort=False)
print(combined_prediction_df)

print("Mean absolute error for all samples: " + str(calculate_error(test_DNAm_df, combined_prediction_df, "All Patients")))
plot_residuals(test_DNAm_df, combined_prediction_df, "All")

