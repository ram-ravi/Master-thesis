# clean generated files in '/home/nora/2020_WaterQuality/Data/input/allFiles':
# 	- round times to closest full hour (hour is the highest resolution we currently have)
#	- transform everything (from GKV mainly) to numerical values except colifast measurements

import os
import pandas as pd
import numpy as np

dataFolder = '../data/allFiles'
outDataFolder = '../data/cleanedFiles'

categoriesColifast = pd.api.types.CategoricalDtype(['<50',  '50', '100', '200', '400', '>400'], ordered=True)

########## parameters to set:
# get all files from data folder or input a specific list of files that should be preprocessed
allFiles = os.listdir(dataFolder)
#allFiles = ['precipitation_VB.csv']
#allFiles = ['precipitation_KR.csv']

###################

for fileName in allFiles:
    if not fileName.endswith('csv'):
        continue
    featName = fileName.split('.csv')[0]
    print(fileName)
    fPath = os.path.join(dataFolder, fileName)
    df = pd.read_csv(fPath, parse_dates=['timestamp'], index_col='timestamp')

    # initially done: round timestamp times to full hour
    # drawback: in rare cases two measurements were rounded to the same datetime + limits the precision of the feature selection when more fine-grained feature measurements will be available
    # df.index = df.index.round('H')


    # handle non-numeric values
    # for colifast data, filter for categories from categoriesColifast
    if featName.startswith('colifast'):
        # replace <10 -> <50; >450 -> >400
        dfRepl = df.replace(to_replace={'<10':'<50', '>450':'>400'})
        dfCat = dfRepl.astype(categoriesColifast)
        df2Store = dfCat.dropna()
    elif not pd.api.types.is_numeric_dtype(df[featName]):
        # replace <1 -> <0;
        #       <2, <10, <100, >2400, >24000 -> 0
        dfRepl = df.replace(to_replace={'<1':'0'})
        dfRepl = dfRepl.replace(to_replace=['<2', '<10', '<16', '<20', '<100', '>2400', '>24000', '>4800'], value=np.nan)
        df2Store = dfRepl.dropna()
    else:
        df2Store = df
    
    # store cleaned files
    outFile = os.path.join(outDataFolder, fileName)
    df2Store.to_csv(outFile)

