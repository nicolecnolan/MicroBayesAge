@echo off

if "%1"=="" (
    echo Required parameter: AgeSplit
    exit /b
)

echo Using AgeSplit = %1

echo on
py ConstructTrainingReferences.py 0m_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 1m_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 2m_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 3m_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 4m_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 5m_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 6m_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 7m_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 8m_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 9m_train_DNAm_matrix.pickle %1

py MicroBayesAgeFolds.py %1 m
