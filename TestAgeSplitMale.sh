#!/bin/sh

if [ "$1" = "" ] ; then
    echo Required parameter: AgeSplit
    exit 1
fi

echo Using AgeSplit = $1

python ConstructTrainingReferences.py 0m_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 1m_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 2m_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 3m_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 4m_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 5m_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 6m_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 7m_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 8m_train_DNAm_matrix.pickle $1
python ConstructTrainingReferences.py 9m_train_DNAm_matrix.pickle $1

python MicroBayesAgeFolds.py $1
