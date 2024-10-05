import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

if len (sys.argv) < 2:
    print ("Required parameters: InputFilename...")
    exit ()

sInputFileName = sys.argv [1]

print ("Reading pickle file", sInputFileName)
full_DNAm_df = pd.read_pickle(sInputFileName)
print(full_DNAm_df.head())
print(full_DNAm_df.shape)

ages = full_DNAm_df['Age'].to_numpy()
print(ages)

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

fig,ax = plt.subplots(1,1)

ax.hist(ages, bins, color = "gray", edgecolor = "black") 
  
ax.set_title("Histogram of Ages") 
ax.set_xlabel('Real Age') 
ax.set_ylabel('Frequency') 
  
plt.show()