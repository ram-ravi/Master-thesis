# read data and combine it into a useful data structure

import os
import time

import numpy as np
import pandas as pd

################################## auxiliary functions #########################

# skip first lines until actual table starts with keyword 'Datum' or another keyword given in 'line'
def skip_to(fName, line,**kwargs):
    if os.stat(fName).st_size == 0:
        raise ValueError("File is empty")
    with open(fName) as f:
        pos = 0
        cur_line = f.readline()
        while not cur_line.startswith(line):
            pos = f.tell()
            cur_line = f.readline()

        f.seek(pos)
        return pd.read_csv(f, **kwargs)

# create an individual file for every feature (column)
def sepAndStoreFeatures(datFrame, outDataFolder, place, quality=False):
    for featCol in datFrame.columns:
        # possible non-feature columns in SMHI
        if (featCol=='timestamp' or featCol.startswith('Kvalitet')):
            continue
        if ('Datum' in featCol or 'dygn' in featCol):
            continue
        # possible non-feature columns in GKV
        if (featCol=='SampleID' or featCol=='Location'):
            continue
        
        #print(featCol)
        finalFeatName = str(featCol)+'_'+place
        qualFeat = 'None'
        qualFeatName = 'None'
        # get position of respective feature to store subsequent quality
        if (quality):
            featPos = np.argwhere(datFrame.columns==featCol).item()
            qualFeat = datFrame.columns[featPos+1]
            qualFeatName = 'qual_'+finalFeatName
            chosenCols = []
        # generate new data frame with only that feature
        datRed = datFrame.filter(items=['timestamp', featCol, qualFeat])
        datRed.rename(columns = {featCol:finalFeatName, qualFeat:qualFeatName}, inplace=True)
        # remove all rows  without feature measurement
        # TODO: seems to not have worked in all cases, specifically in 'colifast_GA'. Double-check why/ it could not have worked in other cases either!
        datRed = datRed[datRed[finalFeatName].notnull()]
        if len(datRed)==0:
            return
        # save data frame
        outFile = os.path.join(outDataFolder, finalFeatName) + '.csv'
        datRed.to_csv(outFile, index=False)


#################################################################################


dataFolder = '../data/raw'
dataFolderSMHI = os.path.join(dataFolder, 'SMHI')
dataFolderGKV = os.path.join(dataFolder, 'GKV')

#outDataFolder = os.path.join(dataFolder, 'input', 'allFiles')
outDataFolder = '../data/allFiles'

measurementsSMHI = ['precipitation_GBG', 'airTemp_GBG', 'clouds_VI', 'sunTime_GBG', 'wind_GBG'] # initially received SMHI data
additionalMeasSMHI_1 = ['precipitation_VB']
additionalMeasSMHI_2 = ['precipitation_KR']
dictFeatRenaming = {'Datum_Tid (UTC)':'timestamp', 'Till Datum Tid (UTC)': 'timestamp',
                    'Nederbördsmängd':'precipitation', 'Lufttemperatur': 'airTemp', 'Total molnmängd': 'clouds',
                    'Solskenstid':'sunTime', 'Vindriktning':'windDir', 'Vindhastighet':'windSpeed'}

# set meas2Read to the (list of) measurement(s) that should be converted into the input format, possibly also need to add more names to the dictFeatRenaming
meas2Read = measurementsSMHI + additionalMeasSMHI_1 + additionalMeasSMHI_2

# iter over all SMHI files
for feat in meas2Read:

    print(feat)

    # read input data (mainly SMHI) and labels (GKV) and store each feature for each location in a separate file with the same structure
    place = feat.split('_')[-1]
    fName = os.path.join(dataFolderSMHI, feat)+'.csv'
    if feat in ['precipitation_VB', 'precipitation_KR']:
        dat = skip_to(fName, 'Från Datum Tid (UTC)', sep=';', parse_dates=['Till Datum Tid (UTC)'])
    else:
        dat = skip_to(fName, 'Datum', sep=';', parse_dates=[['Datum', 'Tid (UTC)']])
    # remove unnecessary columns (starting with an unnamed column)
    lastCol = np.argwhere(dat.columns.str.startswith('Unnamed')).item(-1)
    datRed = dat.iloc[:, 0:lastCol]
    # rename columns 
    datRed.rename(columns = dictFeatRenaming, inplace=True)
    # remove all data before 2002 (labels start only in 2012, so earlier data will most likely not be necessary)
    datRed = datRed[(datRed['timestamp'] >= '2002-01-01 00:00:00')]

    # create 1 new file per feature
    sepAndStoreFeatures(datRed, outDataFolder, place, quality=True)
    