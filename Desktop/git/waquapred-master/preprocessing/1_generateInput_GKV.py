# read GKV data and store it into a useful data structure
# some files are handled individually because of their different structure

import os

import numpy as np
import pandas as pd

################################## auxiliary functions #########################

# skip first lines until actual table starts with keyword 'Datum'
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
        # possible non-feature columns in GKV
        if (featCol=='SampleID' or featCol=='Location'):
            continue
        print(featCol)
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


############################################################################### plot model fit with vertical black dotted line where the training data ends
# plot model fit with vertical black dotted line where the training data ends
# plot model fit with vertical black dotted line where the training data ends
###


dataFolder = '../data/raw'
dataFolderSMHI = os.path.join(dataFolder, 'SMHI')
dataFolderGKV = os.path.join(dataFolder, 'GKV')

#outDataFolder = os.path.join(dataFolder, 'input', 'allFiles')
outDataFolder = '../data/allFiles'

measurementsGKV = ['ColifastEcoliGotaAlv_cleaned', 'LabdataGotaAlv']
#measurementsGKV = ['ColifastEcoliGotaAlv', 'LabdataGotaAlv']
dictFeatRenaming = {'Date':'timestamp', 'Escherichia_coli_per100ml':'colifast',
                'Coliforms_/100ml': 'coliforms', 'E. coli_/100ml':'ecoli',
                'Clostridia_/100ml': 'clostridia', 'EnterococciMF_/100ml':'enterococciMF' ,
                'EnterococciMPN_/100ml': 'enterococciMPN', 'Coliphages_/100ml': 'coliphages',
                'Turbidity_FNU': 'turb','Conductivity_mS/m': 'cond', 'Temperature_C': 'waterTemp',
                'Colour_mg/lPt': 'color', 'Colour410nm_mg/lPt': 'color410'}
dictLocations = {'GÄVGAXXS':'GA', 'GÄVNOXXS': 'SN', 'GÄVSUXXS': 'SU', 'GÄVLÄXXS': 'LAE',
                'LillaEdet':'LE', 'Goeteborgsgrenen':'GG'}

# set meas2Read to the (list of) measurement(s) that should be converted into the input format, possibly also need to add more names to the dictFeatRenaming
meas2Read = measurementsGKV

# read, parse and store GKV data
for feat in measurementsGKV:

    print(feat)

    fName = os.path.join(dataFolderGKV, feat)+'.csv'
    dat = pd.read_csv(fName, parse_dates=['Date'], encoding='iso-8859-1')
    dat.rename(columns = dictFeatRenaming, inplace=True)
    dat = dat[(dat['timestamp'] >= '2002-01-01 00:00:00')]

    # separate the data by place 
    for locKey in dictLocations.keys():
        place = dictLocations.get(locKey)
        rowsForLoc = dat.Location.str.contains(locKey)    
        datPerLoc = dat[rowsForLoc]
        if (datPerLoc.size==0):
            continue
        sepAndStoreFeatures(datPerLoc, outDataFolder, place, quality=False)


# handle flowrate file format (Date, place1, place2)
feat = 'FlowrateVattenfall'
print(feat)
fName = os.path.join(dataFolderGKV, feat)+'.csv'
dat = pd.read_csv(fName, parse_dates=['Date'])
dat.rename(columns = dictFeatRenaming, inplace=True)
dat = dat[(dat['timestamp'] >= '2002-01-01 00:00:00')]

for location in dat.columns[1:]:
    placeAbb = dictLocations.get(location)
    finalFeatName = 'flowrate_' + placeAbb
    datRed = dat.filter(items=['timestamp', location])
    datRed.rename(columns = {location:finalFeatName}, inplace=True)
    outFile = os.path.join(outDataFolder, finalFeatName) + '.csv'
    datRed.to_csv(outFile, index=False)
    
