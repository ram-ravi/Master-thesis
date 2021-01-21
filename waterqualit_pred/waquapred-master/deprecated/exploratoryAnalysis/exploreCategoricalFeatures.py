# explore features with some categorical values: should they be treated as categorical or numerical

dataFolder = '/home/nora/2020_WaterQuality/Data/input/allFiles'

allFiles = os.listdir(dataFolder)
allFiles = [fileName for fileName in allFiles if fileName.endswith('.csv')]
allFiles.sort()

the_file = open('/home/nora/2020_WaterQuality/DataExploration/overviewVariables.tsv', 'w')
the_file.write('Feature\tNo of data points\tNo of unique values\tNo of categorical\n')
for fileName in allFiles:
    featName = fileName.split('.csv')[0]
    print(featName)
    fPath = os.path.join(dataFolder, fileName)
    # read file
    df = pd.read_csv(fPath, parse_dates=['timestamp'], index_col='timestamp')
    if len(df)==0:
        print(fileName+' empty ############################################################')
        continue
    if pd.api.types.is_numeric_dtype(df[featName]):
        outliers = find_anomalies(df[featName])
        df.drop(df.index[outliers], axis=0, inplace=True)
        uniques = np.unique(df[featName])
        the_file.write('{}\t{:d}\t{:d}\t{:d}\n'.format(featName, len(df), len(uniques), 0))
        #print(str(len(df))+'\t::\t'+str(len(uniques))+'\t::\t'+str(uniques)+'\t::\t'+str(nValsCat))
    else:
        # check number of categorical values
        uniques = set(df[featName])
        nValsCat = len(df) - sum(df[featName].str.isnumeric())
        numbers = df[df[featName].str.isnumeric()][featName].tolist()
        unNumbers = np.unique(pd.to_numeric(numbers))
        categ = set(df[~df[featName].str.isnumeric()][featName].tolist())
        print('Numerical values ('+str(len(df))+'):\t')
        print(unNumbers)
        print('Categorical values ('+str(nValsCat)+'):\t')
        print(categ)
        the_file.write('{}\t{:d}\t{:d}\t{:d}\n'.format(featName, len(df), len(uniques), nValsCat))
        # how often does each value appear
        print(df[featName].value_counts()
        #print(str(len(df))+'\t::\t'+str(len(uniques))+'\t::\t'+str(uniques)+'\t::\t'+str(nValsCat))
        #print('########################################### Number of non-numerics: '+str(nValsCat)+'\tof\t'+str(len(df)))
    continue
    the_file.close()


