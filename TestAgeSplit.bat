@echo off

if "%1"=="" (
    echo Required parameter: AgeSplit
    exit /b
)

echo Using AgeSplit = %1

echo on
py ConstructTrainingReferences.py 0_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 1_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 2_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 3_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 4_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 5_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 6_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 7_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 8_train_DNAm_matrix.pickle %1
py ConstructTrainingReferences.py 9_train_DNAm_matrix.pickle %1

py MicroBayesAgeFolds.py %1
