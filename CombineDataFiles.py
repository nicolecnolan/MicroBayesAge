import glob
import pandas
import sys

if len (sys.argv) < 3:
    print ("Required parameters: OutputFileName InputFileName...")
    exit ()

sOutputFileName  = sys.argv [1]
asInputFileNames = sys.argv [2:]

if not sOutputFileName.endswith (".pickle"):
    print ("Output file must be a pickle files")
    exit ()

dfList = []

for sWildCardPattern in asInputFileNames:
    for sInputFileName in glob.glob (sWildCardPattern):
        print('Reading', sInputFileName)
        dfList += [pandas.read_pickle (sInputFileName)]

print('Concatenating', len(dfList), 'dataframes')
df = pandas.concat(dfList, ignore_index=True)

print ("Writing to pickle file", sOutputFileName)
df.to_pickle (sOutputFileName)
