# check correlation between different features, first between colifast and e.coli (lab) using a scatter plot

import os
import itertools
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#########################################################################
# Function to Detection Outlier on one-dimentional datasets (taken from https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623)
def find_anomalies(random_data):
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 4    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    anomalies = []
    for outlier_index in range(1, len(random_data)):
        outlier = random_data[outlier_index]
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier_index)
    return anomalies


# plot functions
def plot_cat_box(feat1, feat2, df, plotFolder):
    # make sure that categorical (if only 1) is on the x-axis
    if pd.api.types.is_numeric_dtype(df[feat1]):
        x_ax, y_ax = feat2, feat1
    else:
        x_ax, y_ax = feat1, feat2
    # plot
    plt.close()
    plotFile = os.path.join(plotFolder, 'categoricals', (feat1+'_'+feat2+'_'+str(len(df))+'datapoints.png'))
    _ = plt.figure(figsize=(8, 8))
    if pd.api.types.is_numeric_dtype(df[feat1]) or pd.api.types.is_numeric_dtype(df[feat2]):
        sns.catplot(x=x_ax, y=y_ax, data=df, kind='box')
    else:
        sns.scatterplot(x=x_ax, y=y_ax, data=df)
    _ = plt.xlabel(x_ax)
    _ = plt.ylabel(y_ax)
    plt.savefig(plotFile, bbox_inches='tight')
    plt.close()


###############################################################################



dataFolderCleaned = '../data/cleanedFiles'
places = ['GBG', 'VI', 'GA', 'SN', 'SU', 'LAE', 'LE', 'GG']
features = ['precipitation', 'airTemp', 'clouds', 'sunTime', 'windDir', 'windSpeed',
            'colifast', 'coliforms', 'ecoli', 'clostridia', 'enterococciMF', 'enterococciMPN', 'coliphages',
            'turb', 'cond', 'waterTemp', 'color', 'color410']
explorationFolder = '/home/nora/2020_WaterQuality/DataExploration'
plotFolder = os.path.join(explorationFolder, 'correlationPlots')

# get all files from dataFolderCleaned and compare with all others
allFiles = os.listdir(dataFolderCleaned)
allFiles = [fileName for fileName in allFiles if fileName.endswith('.csv')]
allFiles.sort()
correlations = pd.DataFrame(np.nan, columns = allFiles)

featNames = [fileName.split('.csv')[0] for fileName in allFiles]
spCorr = pd.DataFrame(np.nan, columns = featNames, index=featNames)
np.fill_diagonal(spCorr.values, 1)

for fileName1, fileName2 in itertools.combinations(allFiles, 2):
    featName1 = fileName1.split('.csv')[0]
    print(featName1)
    fPath1 = os.path.join(dataFolderCleaned, fileName1)
    # read file
    df1 = pd.read_csv(fPath1, parse_dates=['timestamp'], index_col='timestamp')
    if len(df1)==0:
        print(fileName1+' empty ############################################################')
        continue
    if pd.api.types.is_numeric_dtype(df1[featName1]):
        outliers = find_anomalies(df1[featName1])
        df1.drop(df1.index[outliers], axis=0, inplace=True)
        uniques = np.unique(df1[featName1])
    else:
        # check number of categorical values
        uniques = set(df1[featName1])
        nValsCat = len(df1) - sum(df1[featName1].str.isnumeric())
        numbers = df1[df1[featName1].str.isnumeric()][featName1].tolist()
        unNumbers = np.unique(pd.to_numeric(numbers))
        categ = set(df1[~df1[featName1].str.isnumeric()][featName1].tolist())
        print('Numerical values ('+str(len(df1))+'):\t')
        print(unNumbers)
        print('Categorical values ('+str(nValsCat)+'):\t')
        print(categ)


    featName2 = fileName2.split('.csv')[0]
    print(featName2)
    fPath2 = os.path.join(dataFolderCleaned, fileName2)
    df2 = pd.read_csv(fPath2, parse_dates=['timestamp'], index_col='timestamp')
    if len(df2)==0:
        print(fileName2+' empty ############################################################')
        continue
    if pd.api.types.is_numeric_dtype(df2[featName2]):
        outliers = find_anomalies(df2[featName2])
        df2.drop(df2.index[outliers], axis=0, inplace=True)
    else:
        # check number of categorical values
        nValsCat = len(df2) - sum(df2[featName2].str.isnumeric())
        categ = set(df2[~df2[featName2].str.isnumeric()][featName2].tolist())
        print('########################################### Number of non-numerics: '+str(nValsCat))
        print(categ)

    # first try: fill up missing values according to the previous ones, might not be a good idea for very sparse data
    # ffill fills until the next time point, can the interval be limited?
    # df = pd.concat([df1, df2], axis=1).ffill().dropna()

    # second try: use pandas.merge_asof
    # find shorter data frame
    if (len(df1)<len(df2)):
        leftDf = df1
        rightDf = df2
        feat1, feat2 = featName1, featName2
    else:
        leftDf = df2 
        rightDf = df1
        feat1, feat2 = featName2, featName1
    tol = pd.Timedelta('12:00:00')
    df = pd.merge_asof(leftDf, rightDf, left_index=True, right_index=True, direction='nearest', tolerance=tol).dropna()

    # check if still available values can all be cast into numeric
    #print(feat1)
    if pd.api.types.is_string_dtype(df[feat1]):
        if all(df[feat1].str.isnumeric()):
            df[feat1] = pd.to_numeric(df[feat1])
        else:
            df[feat1] = pd.Categorical(df[feat1], categories = ['<50', '50', '100', '200', '400', '>400'], ordered = True)
    #print(feat2)
    if pd.api.types.is_string_dtype(df[feat2]):
        if all(df[feat2].str.isnumeric()):
            df[feat2] = pd.to_numeric(df[feat2])
        else:
            df[feat2] = pd.Categorical(df[feat2], categories = ['<50', '50', '100', '200', '400', '>400'], ordered = True)

    # calcuate spearman correlation
    # if any of the data is colifast, transform categories in numerical values from 1 to 6
    if pd.api.types.is_string_dtype(df[feat1]):
        vecForCorr1 = df[feat1].cat.codes
    else:
        vecForCorr1 = df[feat1]
    if pd.api.types.is_string_dtype(df[feat2]):
        vecForCorr2 = df[feat2].cat.codes
    else:
        vecForCorr2 = df[feat2]
    spCorr.loc[feat1, feat2] = spCorr.loc[feat2, feat1] = vecForCorr1.corr(vecForCorr2, method='spearman')

    if len(df)<20:
        continue

    if pd.api.types.is_numeric_dtype(df[feat1]) and pd.api.types.is_numeric_dtype(df[feat2]):
        plotFile = os.path.join(plotFolder, (feat1+'_'+feat2+'_'+str(len(df))+'datapoints.png'))
        _ = plt.figure(figsize=(8, 8))
        _ = plt.scatter(df[feat1].tolist(), df[feat2].tolist())
        _ = plt.xlabel(feat1)
        _ = plt.ylabel(feat2)
        plt.savefig(plotFile, bbox_inches='tight')
        plt.close()
    else:
        # use seaborn boxplots for categorical data
        plot_cat_box(feat1, feat2, df, plotFolder=plotFolder)


# store correlation matrix
outFile = os.path.join(explorationFolder, 'spearman_correlation.csv')
spCorr.to_csv(outFile, na_rep='NA')

outFile = os.path.join(explorationFolder, 'spearman_correlation.png')
f, ax = plt.subplots(figsize=(13, 12))
hm = sns.heatmap(spCorr, annot=False)
hm.set_facecolor('xkcd:periwinkle')
plt.savefig(outFile, bbox_inches='tight')

outFile = os.path.join(explorationFolder, 'spearman_correlation_clustered.png')
f, ax = plt.subplots(figsize=(13, 12))
spCorr_filledNA = spCorr.fillna(spCorr.mean().mean(), inplace=False)
sns.clustermap(spCorr_filledNA, yticklabels=True, xticklabels=True)
plt.savefig(outFile, bbox_inches='tight')

