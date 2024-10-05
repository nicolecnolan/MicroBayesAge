import pandas
import sys

if len (sys.argv) != 3:
    print ("Required parameters: InputFileName OutputFileName")
    exit ()

sInputFileName  = sys.argv [1]
sOutputFileName = sys.argv [2]

print ("Reading pickle file", sInputFileName)
df = pandas.read_pickle(sInputFileName)

df = df.drop(['Sample_ID', 'Health_Complications', 'Sex'], axis=1) # Drop unnecessary columns

def reformat (sample_df):

    # Change cg site numbers to remove the cg prefix

    cgNumbers = []
    for oldName in sample_df.columns:
        if oldName == 'Age':
            cgNumbers += [-1]
        else:
            cgNumbers += [int(oldName[2:])]

    newColTitles = [ i for i in range(len(sample_df.columns)) ]
    sample_df.columns = newColTitles

    df1 = pandas.DataFrame([cgNumbers], columns=newColTitles)
    sample_df = pandas.concat([df1, sample_df], ignore_index=True)

    # Transpose entire dataframe

    sample_df = sample_df.T

    # Fix column titles - first column title = "cpg", rest = prefix with "p" and turn into str

    newColTitles = []

    for oldName in sample_df.columns:
        if oldName == 0:
            newColTitles += ['cpg']
        else:
            newColTitles += ['p' + str(oldName)]

    sample_df.columns = newColTitles

    # Change first column values from float to int

    sample_df['cpg'] = sample_df['cpg'].astype(int)

    # Move last row (Age) to be first row

    df1 = sample_df.iloc[len(sample_df) - 1:]
    df2 = sample_df.iloc[:len(sample_df) - 1]
    sample_df = pandas.concat([df1, df2], ignore_index=True)

    return sample_df

print("Reformatting dataframe")
full_DNAm_df = reformat(df)

print ("Writing to pickle file", sOutputFileName)
full_DNAm_df.to_pickle(sOutputFileName)
