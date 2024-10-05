# Nonlinear Bayesian Age Prediction Using Microarray Data

## Prerequisites

Ensure Python 3.12 is installed.

Ensure Git is installed.

Install required Python packages.

    pip install loess
    pip install matplotlib
    pip install numpy
    pip install pandas
    pip install scikit-learn
    pip install scipy
    pip install statsmodels
    pip install tqdm
    pip install typing_extensions

## How to Use

Clone this repo.

    git clone https://github.com/nicolecnolan/MicroBayesAge

Change to the cloned repo folder.

    cd MicroBayesAge

GitHub imposes a file size limit. The input dataset has been split into a collection of small Python [pickle](https://docs.python.org/3/library/pickle.html) files. Ensure that you have approximately 23 GB of disk space available before downloading the data files using the provided command.

    python DownloadData.py

Combine all of the small input data files into a combined full dataset pickle file.

    python CombineDataFiles.py fulldata.pickle data_files/*.pickle

You should now have a pickle file named `fulldata.pickle` in the `MicroBayesAge` folder.

Here is the expected checksum and file size for the full dataset.

    cksum fulldata.pickle

    3534348034 23158912815 fulldata.pickle

Reformat the full dataset by dropping unnecessary columns, removing the prefix from cg site numbers, transposing the entire dataset, correcting column titles and column data types, and relocating the last row (Age) to be the first row.

This reformatting process produces a dataset consisting of only the CpG site methylation data and ages.

    python FormatFullDataset.py fulldata.pickle full_DNAm.pickle

Repeat the reformating process and also construct two sex-specific datasets consisting of the CpG site methylation data and ages.

    python FormatDataBySex.py fulldata.pickle male_DNAm.pickle female_DNAm.pickle

You should now have three files containing methylation datasets named `full_DNAm.pickle`, `male_DNAm.pickle`, and `female_DNAm.pickle`.

Here are the expected checksums and file sizes for the three files.

    cksum *_DNAm.pickle

    211548061 11844910543 female_DNAm.pickle
    30464008 23156359799 full_DNAm.pickle
    2934127506 11113304143 male_DNAm.pickle

Calculate the Spearman rank correlation for each CpG site methylation level with respect to age.

    python CalculateCorrelations.py full_DNAm.pickle full_DNAm_corr.pickle
    python CalculateCorrelations.py male_DNAm.pickle male_DNAm_corr.pickle
    python CalculateCorrelations.py female_DNAm.pickle female_DNAm_corr.pickle

Here are the expected checksums and file sizes for the three correlation data files.

    cksum *_DNAm_corr.pickle

    1874409934 11846942875 female_DNAm_corr.pickle
    3720539650 23158392131 full_DNAm_corr.pickle
    3833801190 11115336475 male_DNAm_corr.pickle

Eliminate the CpG sites with methylation values that show little to no correlation with age, based on the Spearman ranks.

    python SelectCorrelatedSites.py full_DNAm_corr.pickle full_DNAm_high_corr.pickle
    python SelectCorrelatedSites.py male_DNAm_corr.pickle male_DNAm_high_corr.pickle
    python SelectCorrelatedSites.py female_DNAm_corr.pickle female_DNAm_high_corr.pickle

Here are the expected checksums and file sizes for the three data files containing only highly correlated sites.

    cksum *_DNAm_high_corr.pickle

    2699969861 67493587 female_DNAm_high_corr.pickle
    3882648007 137944371 full_DNAm_high_corr.pickle
    2286307798 67046167 male_DNAm_high_corr.pickle

Load the datasets into matrices suitable for analysis and age prediction.

    python LoadDataset.py full_DNAm_high_corr.pickle full_DNAm_matrix.pickle
    python LoadDataset.py male_DNAm_high_corr.pickle m_full_DNAm_matrix.pickle
    python LoadDataset.py female_DNAm_high_corr.pickle f_full_DNAm_matrix.pickle

Here are the expected checksums and file sizes for the three data files containing the matrices.

    cksum *_DNAm_matrix.pickle

    2348096087 66789921 f_full_DNAm_matrix.pickle
    3834292930 135028436 full_DNAm_matrix.pickle
    1707184763 65541234 m_full_DNAm_matrix.pickle

You can now generate histograms of the ages of the samples in your datasets.

    python AgeHistogram.py full_DNAm_matrix.pickle
    python AgeHistogram.py m_full_DNAm_matrix.pickle
    python AgeHistogram.py f_full_DNAm_matrix.pickle

Subdivide the data into random training and testing groups for ten-fold cross-validation.

    python SubdivideDataTenFold.py full_DNAm_matrix.pickle train_DNAm_matrix.pickle test_DNAm_matrix.pickle
    python SubdivideDataTenFold.py m_full_DNAm_matrix.pickle m_train_DNAm_matrix.pickle m_test_DNAm_matrix.pickle
    python SubdivideDataTenFold.py f_full_DNAm_matrix.pickle f_train_DNAm_matrix.pickle f_test_DNAm_matrix.pickle

In your MicroBayesAge folder, there should now be 10 training data files and 10 testing data files, numbered 0 - 9, for each of the three input files.

Select one of the training datasets and use it to train the model.

    python ConstructTrainingReferences.py 0_train_DNAm_matrix.pickle

Then generate age predictions for the corresponding testing dataset.

    python MicroBayesAgePredict.py 0_train_DNAm_matrix.pickle 0_test_DNAm_matrix.pickle

Using the `ConstructTrainingReferences.py` and `MicroBayesAgePredict.py` as shown, you can train the model on any of the training data files you have produced so far and then make age predictions for their corresponding testing data files.
