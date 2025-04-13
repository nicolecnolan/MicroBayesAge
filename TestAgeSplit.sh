#!/bin/sh

if [ "$1" = "" ] ; then
    echo Required parameter: AgeSplit
    exit 1
fi

echo Using AgeSplit = $1

python ConstructTrainingReferences.py 0_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 1_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 2_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 3_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 4_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 5_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 6_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 7_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 8_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 9_train_DNAm_matrix.pickle $1

python MicroBayesAgeFolds.py $1
