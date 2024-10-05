import pandas
import sys

def findCorrelations (df):
    
    df2 = df.transpose()

    df2 = df2.drop (labels='cpg', axis=0)
    
    lCorrelations = []

    for iColumn, column in enumerate (df2):
        if (iColumn % 24000) == 0:
            print ("Processed", iColumn, "sites")
        lCorrelations.append (df2[0].corr(df2[column], method='spearman'))

    df.insert (loc=1, column='corr', value=lCorrelations)

    return df

if len (sys.argv) != 3:
    print ("Required parameters: InputFileName OutputFileName")
    exit ()

sInputFileName   = sys.argv [1]
sOutputFileName  = sys.argv [2]

print ("Reading pickle file", sInputFileName)
df = pandas.read_pickle (sInputFileName)

print ("Sorting by age")
df = df.sort_values (by=0, axis=1) # Sort columns by 0th row which contains the ages

print ("Correlating")
df = findCorrelations (df)

print ("DataFrame rows", df.shape[0], "columns", df.shape[1])

print ("Writing to pickle file", sOutputFileName)
df.to_pickle (sOutputFileName)
