import pandas
import sys

if len (sys.argv) != 3:
    print ("Required parameters: InputFileName OutputFileName")
    exit ()

sInputFileName      = sys.argv [1]
sOutputFileName     = sys.argv [2]

if not sInputFileName.endswith (".pickle"):
    print ("Input file must be a pickle file")
    exit ()

if not sOutputFileName.endswith (".pickle"):
    print ("Output file must be a pickle file")
    exit ()

print ("Reading pickle file", sInputFileName)
df = pandas.read_pickle (sInputFileName)

print(df.head())

dfPositives = df.loc[df['corr'] >= 0.60]
dfNegatives = df.loc[df['corr'] <= -0.60]

print(dfPositives.head())
print(dfPositives.shape)
print(dfNegatives.head())
print(dfNegatives.shape)

df = pandas.concat([dfPositives, dfNegatives])
print(df.head())
print(df.shape)

print("Writing to pickle file", sOutputFileName)
df.to_pickle (sOutputFileName)
