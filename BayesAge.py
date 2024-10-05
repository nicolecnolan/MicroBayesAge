"""
BayesAge v1.0 (10/28/2023)

BayesAge is a framework for epigenetic age predictions. This is an extension to scAge (Trapp et al.).
BayesAge utilizes maximum likelihood estimation (MLE) to infer ages, models count data using binomial distributions,
and uses LOWESS smoothing to capture the non-linear dynamics between methylation and age.
For more information on the algorithm, please consult Mboning et al., Frontiers in Bioinformatics (2024).
If you use this software, please cite our work along with Trapp's work. 

The BayesAge pipeline consists of three key steps:
    1) Computing the nonlinear models for each CpG within a DNAm matrix using LOWESS fit.
    2) Loading in and processing sample methylation cgmap files
    3) Predicting age of samples given number of cytosines and coverage from cgmap files.

and can be executed with the following functions:
    1) construct_reference
    2) process_cgmap
    3) bdAge

For more details on the algorithm and how to run the functions, 
please refer to the GitHub page @ https://github.com/lajoycemboning/BayesAge

Copyright, Lajoyce Mboning, 2024.

"""

# import packages and check dependencies
try:
    from scipy.stats import binom
    from typing import Union, List, Tuple, Optional, Dict
    import pathlib as Path
    import numpy as np
    import pandas as pd
    import os
    import time
    from datetime import datetime
    import scipy.stats as stats
    import statsmodels.api as sm
    from scipy.stats import spearmanr
    from multiprocessing import Pool
    lowess = sm.nonparametric.lowess
    import tqdm
    from tqdm.contrib.concurrent import process_map
    import warnings

    warnings.filterwarnings("ignore")

except ImportError as e:
    print(f"One or more required packages is not installed: {e}")
    print("Please verify dependencies and try again.")
    exit()


def commas(value: Union[int, float]) -> str:
    """
    Summary:
    ----------
    This function formats an integer or a float into a comma-separated string
    (i.e. 1000 -> "1,000")

    Parameters
    ----------
    value : int or float value

    Returns
    ----------
    value_with_comma : string of comma-separated number (i.e. 1000 -> "1,000")
    """
    if not isinstance(value, (int, float)):
        raise ValueError("Input value must be an int or float.")
    value_with_comma = f"{value:,}"
    return value_with_comma


def get_range(value_list: List[Union[int, float]]) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Summary:
    ----------
    This function returns the range (minimum value, maximum value) of a given list or array

    Parameters
    ----------
    value_list: list of floats or ints

    Returns
    ----------
    min_max_tuple: tuple of the form (min_value, max_value)

    """
    min_max_tuple = (min(value_list), max(value_list))

    return min_max_tuple


def load_cgmap_file(args):
    """
    Summary:
    ----------
    This function acts as the workhorse for the parallelization function process_cgmap_file.
    It takes as input a tuple of arguments from process_cgmap and either returns a processed 
    methylation matrix or writes it to a specified output path.

    Parameters:
    -----------
    args: a tuple of arguments supplied in parallelization function load_cgmap_file.
    in the form (file, cgmap_directory, split, write_path)
        file -- the name of the .CGmap file
        CGmap_directory -- the path to the directory containing cgmap files
        split -- an optional argument dictating how the file name should be split to name samples
                i.e. if split is ".", then "130010.CGmap" becomes "130010".
        write_path -- the full path of the directory in which to store processed .tsv files.

    Returns:
    -----------
    if write_path == None:
        (sample, cgmap): a tuple containing
                    sample -- the name/identifier of a sample
                    CGmap -- the processed counts of reads of methylated cytosines and counts of all cytosines.
                        (col 1 = genomic position, col 2 = the counts of reads of methylated cytosines, col 3 = counts of all cytosines)

    else:
        CGmap matrix is written as a .tsv file with only the counts of reads of methylated cytosines and counts of all cytosines in write_path.

    """
    # load arguments

    file = args[0]
    cgmap_directory = args[1]
    split = args[2]
    write_path = args[3]

    # list of autosomes for filtering
    autosome_list = [str(x) for x in range(1, 20)]

    # split sample name from file name with desired split
    sample = file.split(split)[0]

    # read cgmap file

    cgmap = pd.read_csv(
        cgmap_directory + file,
        sep="\t",
        header=None,
        names=["Chr", "Nuc", "Pos1", "Cont", "Dinuc", "Meth", "Mc", "Nc"],
        dtype={"Chr": "str", "Pos1": "str"},
    )

    # iloc purely integer-location based indexing for selection by position
    if (
        "chr" in cgmap.iloc[0, 0]
    ):  # Check for if column is labeld "Chr15" instead of "15"
        cgmap["Chr"] = cgmap["Chr"].str.replace("chr", "")
    # print (cgmap)

    # filter autosomes
    cgmap = cgmap[cgmap["Chr"].isin(autosome_list)]

    # create ChrPos column
    cgmap["ChrPos"] = "chr" + cgmap["Chr"] + "_" + cgmap["Pos1"]

    # set index to ChrPos
    cgmap = cgmap.set_index("ChrPos")

    # Sort genomic positions
    cgmap[["Chr", "Pos1"]] = cgmap[["Chr", "Pos1"]].astype("int")
    cgmap = cgmap.sort_values(["Chr", "Pos1"])

    # minimizing the processed dataframe to only necessary columns
    cgmap = cgmap.drop(["Chr", "Nuc", "Pos1", "Cont", "Dinuc", "Meth"], axis=1)

    # remove duplicate indices if there are any (there should be any)
    cgmap = cgmap[~cgmap.index.duplicated(keep="first")]

    # determining whteter to return a tuple or write data to a .tsv
    if write_path == None:
        # return tuple of sample name and cgmap dataframe
        return (sample, cgmap)
    else:
        # write to file
        cgmap.to_csv(write_path + sample + ".tsv", sep="\t")
    del cgmap


def process_cgmap_file(
    cgmap_directory: str,
    output_path: Optional[str] = "./sc_data_processed/",
    n_cores: int = 1,
    split: str = ".",
    chunksize: int = 1,
) -> Optional[Dict[str, pd.DataFrame]]:

    #
    """
    Process CGmap files in parallel and return a dictionary of sample methylome matrices.

    Parameters:
    ----------
    cgmap_directory: str
        The path to the directory containing .CGmap files.
    output_path: str, optional
        The path to the output directory in which to write .tsv files. If None, data is returned.
    n_cores: int
        The number of CPU cores to use for parallel processing.
    split: str
        The symbol/letter/number to split by when generating sample names from files.
    chunksize: int
        The number of elements to feed to each worker during parallel processing.

    Returns:
    ----------
    if output_path is None:
        sc_dict: dict
            A dictionary with sample names/identifiers as keys and binary methylation matrices as values.
    """
    start_time = time.time()

    print("----------------------------------------------------------")
    # get list of files in directory
    print("Loading .cgmap files from '%s'" % cgmap_directory)
    cgmap_list = sorted(os.listdir(cgmap_directory))
    print("Number of BSBolt .CGmap files = %s" % len(cgmap_list))
    print("First .CGmap file name: '%s'" % cgmap_list[0])
    print("----------------------------------------------------------\n")

    # create tuple of arguments for load_cgmap_file
    file_tuples = []
    for file in cgmap_list:
        file_tuples.append((file, cgmap_directory, split, output_path))

    print("----------------------------------------------------------")
    print("Starting parallel loading and processing of .CGmap files...")
    # parallelization function with tqdm progress bar
    results = process_map(
        load_cgmap_file,
        file_tuples,
        max_workers=n_cores,
        chunksize=chunksize,
        desc="sample loading progress ",
        unit=" sample methylomes",
    )
    print("\nParallel loading complete!")

    # determine whether to write data or return a dictionary
    if output_path == None:  # return dictionary of sample matrices
        sample_dict = {}
        for result_tuple in results:
            sample_dict[result_tuple[0]] = result_tuple[1]
        print("Returning a dictionary, as no output path was given.")
        return sample_dict
    else:  # sample matrices are written to output_path inside of load_cgmap_file
        print("Processed matrices written to '%s'" % output_path)
    print("----------------------------------------------------------\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time elapsed to process CGmap files = %.3f seconds" % elapsed_time)
    print("\nprocess_cgmap run complete!")

# Sample Usage:
# sample_data = process_cgmap_file(cgmap_directory="./data/cgmaps/", output_path=None, n_cores=4)


def construct_reference(
    training_matrix,
    reference_name,
    output_path="./predictions/",
    zero_met_replacement=0.001,
    one_met_replacement=0.999,
    min_age=-20,
    max_age=60,
    age_step=0.1,
    tau=0.7,
):
    '''
    Parameters:
    ----------
    training_matrix = full path to the reference matrix you want to create the reference matrix
    for where samples(rows) and CpG sites (columns) with some additional metadata columns (at least "Age")
    reference_name = name of the processed reference matrix dataset.
    output_path = the full path where to output the reference matrix file.
    zero_met_replacement = value to replace the extreme methylation probabilities with.
    one_met_replacement = value to replace the extreme methylation.
    min_age = minimum age of the reference model.
    max_age = maximum age of the reference model.
    age_step = age step between minimum age and maximum age.
    tau = Tau value of the lowess fit.

    Returns:
    ----------
    Can return the reference matrix on top of saving it in the output_path as a .tsv file.

    Notes:
        Spearman metrics are calculated using spearmanr from the scipy.stats library.
        The lowess fit is done using sm.nonparametric.lowess 
    '''

    start_time = time.time()
    # Read the reference data into a DataFrame
    training_cv_df = pd.read_pickle(training_matrix)

    # List of columns (chromosome sites)
    chromosome_sites = [
        x for x in training_cv_df.columns if x.startswith("cg")]

    # Generate a range of age values
    age_steps = np.arange(min_age, max_age + age_step, age_step)

    # Age values from the reference data
    age_training = training_cv_df["Age"].values.flatten()

    final_training_matrix = []
    num_sites = len(chromosome_sites)

    for i, cpg_site in enumerate(chromosome_sites):
        
        # Print progress
        print (i, "of", num_sites)

        # Methylation levels for the current site
        cpg_site_meth = training_cv_df[cpg_site].values.flatten()

        # Perform lowess smoothing
        predicted_meth_levels = lowess(
            cpg_site_meth, age_training, xvals=age_steps, frac=tau)

        # Calculate Spearman rank correlation
        rho, p = spearmanr(age_training, cpg_site_meth)

        spearman_rank = rho
        p_value = p

        # Replace extreme methylation probabilities
        final_pred_meth = [zero_met_replacement if meth <=
                           0 else one_met_replacement if meth >= 1 else meth for meth in predicted_meth_levels]

        final_cpg_meth_tuple = (
            cpg_site,) + tuple(final_pred_meth) + (spearman_rank, p_value)

        final_training_matrix.append(final_cpg_meth_tuple)

    # Create column names for the DataFrame
    df_columns = ["ChrPos"] + \
        [str(num) for num in age_steps] + ["spearman_rank", "p_value"]

    df = pd.DataFrame(final_training_matrix, columns=df_columns)

    df.set_index("ChrPos", inplace=True)

    # Save the DataFrame to a pickle file
    df.to_pickle(output_path + reference_name + ".pickle")

    print("\nReference model dataset written to '%s'" % output_path)

    number_of_samples = training_cv_df.shape[0] - 1
    # write report file detailing input matrix.
    with open(output_path + f'{reference_name}.report.txt', 'w') as writer:
        writer.write("BayesAge reference report for %s\n" % reference_name)
        now = datetime.now()
        write_datetime = now.strftime("%m/%d/%Y %H:%M:%S")
        writer.write("Reference file created: %s\n\n" % write_datetime)
        writer.write("Number of input samples = %s\n" % number_of_samples)
        writer.write("Number of input CpGs= %s\n" % len(chromosome_sites))

    print("Report file generated at '%s.report.txt'" % output_path)
    print("-----------------------------------------------------\n\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\nTime to run construct reference: %0.3f seconds" % elapsed_time)

    print("\n The reference is constructed and saved!")

    return df  # Optionally return the DataFrame


def compute_binomial_probabilities(args):
    """
    Summary:
    ----------
    The bdAge function is the main function of the pipeline for age prediction. 

    refer to bdAge for addtional information

    Paramters:
    --------------
    args: tuple, in the form()

    Returns:
    --------------
    binomial_probabilities_ouput: tuple, in the form ()

    """

    # Get arguments
    # sample,
    # sample_path,
    # corr_regr_data,
    # selection_mode,
    # CpG_parameter_num,
    # min_age,
    # max_age,
    # age_step,

    sample_name = args[0]
    sample_dir_df = args[1]
    reference_data = args[2]
    selection_mode = args[3]
    CpG_parameter = args[4]
    min_age = args[5]
    max_age = args[6]
    age_step = args[7]
    # n_cores = 1,
    # chunksize = 5

    # Determine whether the input is pandas dataframe or string (file path)
    input_type = str(type(sample_dir_df))
    if input_type == "<class 'str'>":
        sample_dir_df = pd.read_csv(sample_dir_df, sep="\t", index_col=0)
    elif input_type == "<class 'pandas.core.frame.DataFrame'>":
        pass

    start = time.time()

    # Intersecting CpGs between bulk and reference dataset
    ref_sc_intersect_df = pd.concat(
        [reference_data, sample_dir_df], axis=1, join="inner"
    )

    # profiling mode selection
    if selection_mode == "percentile":  # ex: top 1% age-associated CpGs per sample
        quantile = ref_sc_intersect_df["spearman_rank"].abs().quantile(
            q=CpG_parameter)
        SpearmanR_abs_top = ref_sc_intersect_df[
            ref_sc_intersect_df["spearman_rank"].abs() >= quantile
        ]
    elif selection_mode == "numCpGs":  # ex: top 500 age-associated CpGs per sample
        SpearmanR_abs_top = (
            ref_sc_intersect_df["spearman_rank"].abs().nlargest(CpG_parameter)
        )
    elif selection_mode == "cutoff":  # ex: CpGs above a cutoff of r ≥ 0.7 per sample
        SpearmanR_abs_top = ref_sc_intersect_df[
            ref_sc_intersect_df["spearman_rank"].abs() >= CpG_parameter
        ]

    ref_sc_intersect_df_subset = ref_sc_intersect_df.loc[SpearmanR_abs_top.index, :]

    # subset dataframe to chosen highly age-correlated CpGs
    ref_sc_intersect_df_subset = ref_sc_intersect_df.loc[SpearmanR_abs_top.index, :]

    # isolate selected sites
    selected_sites = list(ref_sc_intersect_df_subset.index)

    # get age steps from min_age to max_age (inclusive of both)
    age_steps = np.arange(min_age, max_age + age_step, age_step)
    # age_steps = np.arange(int(min_age), int(max_age) + int(age_step), int(age_step))

    # create list to store probability profiles
    list_of_profile_probabilities_per_age = []

    # Loop through each age step
    for age in age_steps:
        # Create list to hold probability for all chosen CpGs for a given age
        probability_list_one_age = []
        # loop through each age site
        for site in selected_sites:

            methylation_probability = ref_sc_intersect_df_subset.loc[site,
                                                                     f"{age}"]

            # number of successes
            ns = ref_sc_intersect_df_subset.loc[site, "Mc"]

            # number of attempts
            na = ref_sc_intersect_df_subset.loc[site, "Nc"]

            binomial_probability = binom.pmf(
                k=ns, n=na, p=methylation_probability)

            probability_list_one_age.append(np.log(binomial_probability))

        # compute log-likelihood sum and appened to list
        # this is equivalent to computing the product of probabilities
        # but neatly avoid underflow errors that result when multiplying
        # many small numbers together
        list_of_profile_probabilities_per_age.append(
            np.sum(probability_list_one_age))

    # transform into dataframe with age steps
    age_probability_df = pd.DataFrame(
        {"Pr": list_of_profile_probabilities_per_age}, index=age_steps
    )

    # print(age_probability_df)

    # compute highest likelihood age among age steps
    max_probability_age = round(float(age_probability_df.idxmax()), 2)

    # compute maximum probability
    max_probability = float(age_probability_df["Pr"].max())

    end = time.time()

    # Compute sample characteristics
    mean_meth_count = sample_dir_df["Mc"].mean()
    mean_coverage = sample_dir_df["Nc"].mean()
    coverage = len(sample_dir_df)
    num_intersections = len(ref_sc_intersect_df)

    # Return tuple output
    binomial_probabilities_output = (
        sample_name,
        max_probability_age,
        age_probability_df,
        ref_sc_intersect_df_subset,
        mean_meth_count,
        mean_coverage,
        coverage,
        num_intersections
    )

    return binomial_probabilities_output


def bdAge(sample_dir_or_dict,
          sample_set_name,
          reference_data,
          output_path="./predictions/",
          selection_mode="percentile",
          CpG_parameter=1,
          min_age=0,
          max_age=100,
          age_step=1,
          n_cores=1,
          chunksize=5,
          ):
    """
    Summary:
    -----------
    This is the main function of the pipeline. 

    Parameters:
    ----------
    sample_dir_or_dict = str or dict, either the directory containing processed cgmap files
                             as .tsv/.tsv.gz files (i.e. generated by process_cgmap_file),
                             or a dictionary of labeled methylation matrices
    sample_set_name = str, the desired name of the sample data
                          this is used in setting the name of the output file
    reference_data = str, the full file path to the desired reference data (i.e: .csv, .tsv)

    output_path = path of directory to store the files with predicted ages

    selection_mode = str, one of [numCpGs, percentile, cutoff] where
                         percentile -- selects the top x% age-associated CpGs per sample/subject
                         numCpGs -- selects a defined number of age-associated CpGs per sample/subject
                         cutoff -- selects only CpGs with a Spearman Rank ≥ cutoff
     : float, the parameter to feed in given a specific selection mode
                          ex1: selection_mode == percentile --> CpG_parameter = 1 (Top 1% percentile)
                          ex2: selection_mode == numCpGs --> CpG_parameter = 1000 (1000 CpGs/subject/sample)
                          ex3: selection_mode == cutoff --> CpG_parameter = 0.7 (Only CpGs with spearman_r ≥ 0.7)
    CpG_parameter = Parameter to specifically choose the number of CpG sites. This is a parameter accompanying selection mode.
    min_age =  the minimum age for which to build a probability profile
    max_age = the maximum age for which to build a probability profile
    age_step = the step value for computing probability profiles
              (i.e. if age_step == 1, likelihoods will be calculated for every 1 month) or year
    n_cores = int, the number of cores to use for parallel processing
              
    Returns:
    ----------
    2 files are created: 1 .report.txt file containing the parameters used in running the algorithm
                         and 1 .tsv file containing the results and predictions for each sample/subject
    """
    start = time.time()
    print("bdAge algorithm starting!\n")

    print("----------------------------------------")
    print("Profiling epigenetic age in '%s' " % sample_set_name)

    # Determine input data is a path to a directory or a dictionary of data

    input_type = str(type(sample_dir_or_dict))

    if input_type == "<class 'str'>":
        sample_cov_dir = sample_dir_or_dict
        print(
            "Loading processed CGmap methylation files from '%s'..." % sample_cov_dir
        )
        sample_files = sorted(os.listdir(sample_cov_dir))

        # check if files are gzipped
        if ".gz" in sample_files[0]:
            add_gz = True
        else:
            add_gz = False
            samples = [sample.split(".tsv")[0] for sample in sample_files]

    elif input_type == "<class 'dict'>":
        print("Using sample stored in dictionary...")
        samples = list(sample_dir_or_dict.keys())

    print("Number of samples to analyze: %s" % len(samples))

    # get name of the reference dataset from the path input
    training_dataset_name = reference_data.split("/")[-1].split(".tsv")[0]

    print("\nbdAge parameters: ")
    print("---------------------------------------------------------")
    print("Using reference training data: %s" % training_dataset_name)

    # If the reference file is there, loads it in
    try:
        corr_regr_data = pd.read_csv(reference_data, sep="\t", index_col=0)

    # If the reference file cannot be found, an arror is thrown
    except:
        raise NameError(
            "Reference training set not found, please verify input directory."
        )

    print(
        "Shape of reference matrix: {} CpGs, {} metric columns".format(
            commas(corr_regr_data.shape[0]), corr_regr_data.shape[1]
        )
    )
    print("\n")
    print("Using %s cores with chunksize of %s" % (n_cores, chunksize))
    print("\n")
    print("Setting minimum age to %s month(s)/year(s)" % min_age)
    print("Setting maximum age to %s month(s)/year(s)" % max_age)
    print("Using age step of %s month(s)/year(s)" % age_step)
    print("\n")
    print("Using profiling mode: %s" % selection_mode)
    if selection_mode == "percentile":
        print(
            "---> Profiling top %s%% age-related CpGs by absolute Spearman rank"
            % (str(CpG_parameter))
        )
        # for example, providing a value of 1 means selecting the top 1% (absolute highest) age correlated CpGs
        CpG_parameter_num = (100 - CpG_parameter) / 100
    elif selection_mode == "numCpGs":
        # For example, providing a value of 1000 means selecting the top 1000 (absolute highest) age correlated CpGs.
        print(
            "---> Profiling top %s age-related CpGs by absolute Spearman rank"
            % CpG_parameter
        )
        CpG_parameter_num = CpG_parameter
    elif selection_mode == "cutoff":
        # For exampe, providing a value of 0.7 means CpGs with an absolute age-correlation greated than or equal to 0.7
        print(
            "---> Profiling top age-related CpGs above an absolute correlation cutoff of %s" % CpG_parameter
        )
        CpG_parameter_num = CpG_parameter
    else:
        raise ValueError(
            "Incorrect selection mode, must be one of ['percentile', 'numCpGs', 'cutoff']"
        )

    print("---------------------------------------------------------------------")

    # Create a tuple of arguments for parallel processing
    list_of_arguments_parallel_bdAge = []

    # If the full path to processed sample methylation files is given
    if input_type == "<class 'str'>":
        for sample in samples:
            if add_gz == True:
                sample_path = sample_dir_or_dict + sample + ".tsv.gz"
            elif add_gz == False:
                sample_path = sample_dir_or_dict + sample + ".tsv"
            list_of_arguments_parallel_bdAge.append(
                (
                    sample,
                    sample_path,
                    corr_regr_data,
                    selection_mode,
                    CpG_parameter_num,
                    min_age,
                    max_age,
                    age_step,
                )
            )

    # or if single-cell data is provided as a labeled dictionary
    elif input_type == "<class 'dict'>":
        for sample in sample_dir_or_dict:
            list_of_arguments_parallel_bdAge.append(
                (
                    sample,
                    sample_path[sample],
                    corr_regr_data,
                    selection_mode,
                    CpG_parameter_num,
                    min_age,
                    max_age,
                    age_step,
                )
            )
    print("\n\n----------------------------------------------------------")
    print("Starting parallel processing of all samples with %s cores!\n" % n_cores)

    # compute probabilities using parallel procesing with a progress bar using tqdm
    results = process_map(compute_binomial_probabilities,
                          list_of_arguments_parallel_bdAge,
                          max_workers=n_cores,
                          chunksize=chunksize,
                          desc="bdAge progress ",
                          unit=" Age predictions",
                          )

    # process output data into a final dataframe
    sample_data_dict = {}

    for sample in results:
        # Get name of the sample
        sample_name = sample[0]

        # Get preidcted age of the sample
        sample_age = sample[1]

        # Ge the list of age steps that were tested
        ages_tested = list(
            np.around(sample[2].index.values.astype("float"), 2))

        # Get the likelihood for each age step
        ages_likelihoods = list(sample[2]["Pr"])

        # Get CpGs chosen by the ranking algorithm
        CpGs_chosen = list(sample[3].index)

        # Get the number of CpGs that were selected
        numCpGs_selected = len(CpGs_chosen)

        # Isolated the Spearman rank correlations of chosen CpGs
        correlations = list(sample[3]["spearman_rank"])

        # Isolate the number of methylated cytosine count of chosen CpGs
        methylation_count = list(sample[3]["Mc"])

        # Isolate the number of methylated cytosine + non-methylated cytosine count of chosen CpGs
        coverage_count = list(sample[3]["Nc"])

        # Get mean methylation count of the sample
        mean_meth_count = sample[4]

        # Get mean coverage count of the sample
        mean_coverage = sample[5]

        # Get CpG coverage of the sample
        coverage = sample[6]

        # Get the number of intersections between sample and reference data
        num_intersections = sample[7]

        # Combine all the data into a list and save to dictionary
        sample_data_dict[sample_name] = [
            sample_age,
            mean_meth_count,
            mean_coverage,
            coverage,
            num_intersections,
            ages_tested,
            ages_likelihoods,
            CpGs_chosen,
            numCpGs_selected,
            correlations,
            methylation_count,
            coverage_count,
        ]

        # Create dataframe from dictionary
        sample_predictions_df = pd.DataFrame.from_dict(sample_data_dict,
                                                       columns=[
                                                           "Predicted Age",
                                                           "Mean Meth Count",
                                                           "Mean Coverage",
                                                           "Sample Coverage",
                                                           "Intersections",
                                                           "Ages Tested",
                                                           "Age Likelihood",
                                                           "Selected CpGs",
                                                           "Number CpGs",
                                                           "Correlations",
                                                           "Methylation Count",
                                                           "Coverage Count",
                                                       ],
                                                       orient="index",
                                                       )

    # Create descriptive name for output file
    # Ex: name-train(Thompson_Liver_BL6)-mode(percentile)-param(top_1_pct).tsv
    # The most crucial parameters (the training data, the selection mode, and the selection parameters) are automatically encoded in the output file name
    # Addtional data about the run is written to a .report.txt file
    base_output_name = (
        output_path
        + sample_set_name
        + "-train("
        + training_dataset_name
        + ")-mode("
        + selection_mode
    )

    if selection_mode == "percentile":
        output_file_name = base_output_name + \
            ")-param(top_%s_pct).tsv" % CpG_parameter
    if selection_mode == "numCpGs":
        output_file_name = base_output_name + \
            ")-param(%sCpGs).tsv" % CpG_parameter
    if selection_mode == "cutoff":
        output_file_name = (
            base_output_name + ")-param(above_%s_cutoff).tsv" % CpG_parameter
        )

    # check if output path directory exists, otherwise creates it
    print("\n")
    try:
        os.listdir(output_path)
    except:
        print("Output path does not exist, creating directory...")
        os.makedirs(output_path)

    # save prediction dataframe to a .tsv file
    sample_predictions_df = sample_predictions_df.rename_axis(index="Sample")
    sample_predictions_df.to_csv(output_file_name, sep="\t")

    end = time.time()
    print("\nPredictions stored in '%s'" % output_path)
    print("----------------------------------------------------------")
    print("\nTime elapsed to generate bdAge results = %0.3f seconds\n" %
          (end - start))
    print("BdAge run complete!")

    # write a report file containing the parameters that were used
    # in the current bdAge run
    with open("%s.report.txt" % output_file_name[:-4], "w") as writer:
        writer.write("bdAge report for '%s'\n" % output_file_name)
        now = datetime.now()
        write_datetime = now.strftime("%m/%d/%Y %H:%M:%S")
        writer.write("Files created: %s\n\n" % write_datetime)
        if input_type == "<class 'dict'>":
            writer.write("Methylation data loaded in from dictionary\n")
        if input_type == "<class 'str'>":
            writer.write("Methylation data loaded in from '%s'\n" %
                         sample_dir_or_dict)
        writer.write("Training dataset used: %s\n" % training_dataset_name)
        writer.write("Selection mode: %s\n" % selection_mode)
        if selection_mode == "percentile":
            writer.write(
                "CpG parameter: top %s%% age-associated CpGs\n" % CpG_parameter
            )
        elif selection_mode == "numCpGs":
            writer.write(
                "CpG parameter: top %s age-associated CpGs\n" % commas(
                    CpG_parameter)
            )
        elif selection_mode == "cutoff":
            writer.write(
                "CpG parameter: CpGs with Pearson correlation ≥ %s\n"
                % commas(CpG_parameter)
            )
        writer.write("Minimum age: %s\n" % min_age)
        writer.write("Maximum age: %s\n" % max_age)
        writer.write("Age step: %s\n" % age_step)
        writer.write("Time to generate results: %0.3f seconds" % (end - start))
