#!/bin/sh

if [ "$1" = "" ] ; then
    echo Required parameter: AgeSplit
    exit 1
fi

echo Using AgeSplit = $1

python ConstructTrainingReferences.py 0f_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 1f_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 2f_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 3f_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 4f_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 5f_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 6f_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 7f_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 8f_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 9f_train_DNAm_matrix.pickle $1

python MicroBayesAgeFolds.py $1
