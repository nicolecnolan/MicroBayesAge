@echo off

if "%1"=="" (
    echo Required parameter: AgeSplit
    exit /b
)

echo Using AgeSplit = %1

echo on
py ConstructTrainingReferences.py 0f_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 1f_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 2f_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 3f_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 4f_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 5f_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 6f_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 7f_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 8f_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 9f_train_DNAm_matrix.pickle %1

py MicroBayesAgeFolds.py %1 f
