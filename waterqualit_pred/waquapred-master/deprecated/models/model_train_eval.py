# (i) create an input matrix from a list of features to be considered, together with their lags,
# (ii) train model,
# (iii) evaluate model performance

import os
import pandas as pd
import numpy as np
import datetime

from sklearn import model_selection
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

import matplotlib.pyplot as plt

from aux_functions_forecasting import *

######################################## FUNCTIONS ####################################################


def mkEmptyDfForCurrentFeature(featName, lagF_td, freq, index_):
    arbitraryDate = pd.Timestamp.today()
    # how many measurements do we expect for this feature and lag (the exact date=arbitraryDate does not matter)
    expNumMeas = len(pd.date_range(start=arbitraryDate-lagF_td, end=arbitraryDate, freq=freq))-1
    cols = [featName+'_'+str(i) for i in reversed(range(1, expNumMeas+1))]
    currentFeat_df = pd.DataFrame(index=index_, columns=cols)
    return currentFeat_df

def imputeMissingValues(incompleteMeas, expMeasTimes, data, freq='d', method='windowMean'):
    usedMeas = incompleteMeas.copy()
    if freq=='d':
        # identify at which dates values are missing
        missingDat_set = set(expMeasTimes.date) - set(usedMeas.index.date)
        time = usedMeas.index.time[0]
        missingDat_ = [pd.Timestamp(datetime.datetime.combine(dat, time)) for dat in list(missingDat_set)]
    # impute the mean value of the chosen window
    if method=='windowMean':
        meanVal = np.mean(usedMeas)
        for missing in missingDat_:
            usedMeas[missing] = meanVal
#    # TODO impute the last measured value
#    if method=='propagateLast':
#        for missing in missingDat_:
    return usedMeas

## create input matrix
def create_input_matrix(label, feature_lag_dict, feature_freq_dict, forecastDay=1, inputFolder='../data'):
    predDay_td = pd.Timedelta(predDay, unit='d')
    # read labels
    y = pd.read_csv(os.path.join(inputFolder, label)+'.csv', index_col='timestamp', parse_dates=True)

    allFeats_dict = {}
    # read features: in case of memory problems, storing the allFeats_dict could removed and feature matrices instead be read in 'mkFeatMat'
    for feat in feature_lag_dict.keys():
        featFile = os.path.join(inputFolder, feat)+'.csv'
        featDat = pd.read_csv(featFile, index_col='timestamp', parse_dates=True)
        allFeats_dict[feat] = featDat
        # if no frequency is given for the feature, set it to days
        if feat not in feature_freq_dict.keys():
            feature_freq_dict[feat] = 'd'

    # feature_df: feature matrix used for prediction models
    feature_df = pd.DataFrame(index=y.index)

    # for each datatype (e.g. precipitation_VB)
    for feat in allFeats_dict:
        featDat = allFeats_dict[feat]
        lagF_td = pd.Timedelta(feature_lag_dict[feat], unit='d')
        freqF = feature_freq_dict[feat]
        # create empty dataframe with the correct dimensions and column names
        currentFeat_df = mkEmptyDfForCurrentFeature(feat, lagF_td, freq=freqF, index_=feature_df.index)

        # iterate over all labels to create corresponding input
        for timeId in feature_df.index:
            # define end of lag period by subtracting number of days to predict into the future from the measurement time of y
            lagEnd = timeId-predDay_td
            lagStart = lagEnd-lagF_td
            # find closest measurement in x on the same date

            expMeasTimes = pd.date_range(start=lagEnd-lagF_td, end=lagEnd, freq=freqF)[1:]
            # if frequency = days: get last 'lag' days and associated measurements (matched on the basis of dates)
            if freqF=='d':
                # check for which dates of expected (expMeasTimes) exist measurements (x)
                idx_ = np.in1d(featDat.index.date, expMeasTimes.date)
                usedMeas = featDat[feat][idx_]
                # check for missing values and impute if less than 50% are missing
                if len(usedMeas)<len(expMeasTimes):
                    numMissing = len(expMeasTimes)-len(usedMeas)
                    print('{}:\t{} missing values'.format(timeId, numMissing))
                    if numMissing>(0.5*len(expMeasTimes)):
                        # remove this y-label from feature_df
                        feature_df.drop(timeId, inplace=True)
                        currentFeat_df.drop(timeId, inplace=True)
                    else:
                        # impute missing values
                        usedMeas = imputeMissingValues(usedMeas, expMeasTimes, x, freq=freqF, method='windowMean')
            # TODO if frequency = hours: create date_range of last 'lag' days in hours
            if (len(usedMeas>0)):
                currentFeat_df.loc[timeId] = usedMeas.values
        # append df of current feature to feature_df
        feature_df = pd.concat([feature_df, currentFeat_df], axis=1)

    # remove all y values without complete feature sets
    y_compl = y.loc[feature_df.index]

    return y_compl, feature_df

# takes fitted model; returns train and test performances and predictions (train + test)
def eval_model(fittedModel, modelAbbr, feature_df, y_df, endIndexTrain):
    label = y_df.columns[0]
    preds = fittedModel.predict(feature_df)
    preds_df = pd.DataFrame(preds, index=y_df.index, columns=[modelAbbr])

    # train and test performance
    trainPerformances = obtain_performance(y_df[label][:endIndexTrain].values, preds_df[modelAbbr][:endIndexTrain].values)
    testPerformances = obtain_performance(y_df[label][endIndexTrain:].values, preds_df[modelAbbr][endIndexTrain:].values)

    return trainPerformances, testPerformances, preds_df


# takes a model; performs hyperparameter optimization using time series (expanding window) cross validation; returns train, validation and test error
def do_CV_and_validation(model, modelAbbr, param_grid, feature_df, y_df, endIndexTrain):
    label = y_df.columns[0]
    # remove validation set
    x_cv, y_cv = feature_df[:endIndexTrain], y_df[:endIndexTrain]
    # perform expanding window cross-validation on the train set (=x_cv)
    tscv = model_selection.TimeSeriesSplit(n_splits=5)
    cv_fit = model_selection.GridSearchCV(model, param_grid, cv=tscv, scoring=['r2', 'neg_mean_absolute_error'], return_train_score=True, refit='r2')
    cv_fit.fit(x_cv,y_cv.values.ravel())
    # results: model parameter, train error, test error
    model_parameters= cv_fit.best_estimator_.get_params()
    cv_train_score = {'R2': cv_fit.cv_results_['mean_train_r2'][cv_fit.best_index_], 'MAE': -cv_fit.cv_results_['mean_train_neg_mean_absolute_error'][cv_fit.best_index_]}
    cv_test_score = {'R2': cv_fit.best_score_, 'MAE':-cv_fit.cv_results_['mean_test_neg_mean_absolute_error'][cv_fit.best_index_]}

    # get validation score
    preds = cv_fit.best_estimator_.predict(feature_df)
    preds_df = pd.DataFrame(preds, index=y_df.index, columns=[modelAbbr])

    validation_perf = obtain_performance(y_df[label][endIndexTrain:].values, preds_df[modelAbbr][endIndexTrain:].values)
    validation_perf = {'R2': validation_perf['R2'], 'MAE': validation_perf['MAE']}

    return {'optParams': model_parameters, 'cv_train': cv_train_score, 'cv_test': cv_test_score, 'val': validation_perf, 'preds': preds_df}

####################################################################

inputFolder = '../data/cleanedFiles'
outFolder = './results'

# how many days into the future should be predicted?
predDay = 1

# dictionary of all features with their lags, all lags need to be provided in days
feature_lag_dict = {'precipitation_VB':7, 'flowrate_LE':6}
label = 'ecoli_GA'

# dictionary of all features with their frequencies, if a frequency is not provided here, days are assumed
feature_freq_dict = {'precipitation_VB':'d', 'flowrate_LE':'d'}

y_compl, feature_df = create_input_matrix(label, feature_lag_dict, feature_freq_dict, forecastDay=1, inputFolder=inputFolder)

# split available data into train and test set (80-20)
endIndexTrain = round(len(y_compl)*0.8)
feat_train = feature_df[:endIndexTrain]
y_train, y_test = y_compl[:endIndexTrain], y_compl[endIndexTrain:]

# parameter grids for different algorithms
LR_grid = {'normalize': [True, False]}
Lasso_grid = {'alpha':[0.2, 0.5, 0.8, 1]}
RF_grid = {'n_estimators':[50,100, 500, 1000], 'criterion': ['mse', 'mae'], 'max_depth':[None,2,3,4],
        'min_samples_split':[0.05, 5, 10, 20], 'min_samples_leaf': [0.01,0.05,0.1], 'max_features':['auto','log2','sqrt'], 'random_state':[0]}
SVR_grid = {'C': [.5,1,10, 100, 1000],'gamma': ['auto', 'scale'],'kernel':['linear', 'poly', 'rbf']}

model_dict = {'LR':{'model':linear_model.LinearRegression(), 'params':LR_grid},
            'Lasso': {'model':linear_model.Lasso(), 'params':Lasso_grid},
            'RF': {'model':RandomForestRegressor(), 'params':RF_grid},
            'SVR': {'model':svm.SVR(), 'params':SVR_grid}}

modelXY = '_'.join(['{}_{}'.format(k,v) for (k,v) in feature_lag_dict.items()])


errorPerModel_train = pd.DataFrame(columns=modelCall.keys())
errorPerModel_test = pd.DataFrame(columns=modelCall.keys())
errorPerModel_val = pd.DataFrame(columns=modelCall.keys())

for modelName in model_dict:
    print(modelName)
    model = model_dict[modelName]['model']
    cv_eval = do_CV_and_validation(model, modelName, model_dict[modelName]['params'], feature_df, y_df, endIndexTrain)
    print('train:'+'\t'.join(['{}: {}'.format(k,round(v,3)) for (k,v) in cv_eval['cv_train'].items()]))
    print('test:'+'\t '.join(['{}: {}'.format(k,round(v, 3)) for (k,v) in cv_eval['cv_test'].items()]))
    print('val:'+'\t '.join(['{}: {}'.format(k,round(v, 3)) for (k,v) in cv_eval['val'].items()]))
    errorPerModel_train = errorPerModel_train.combine_first(pd.DataFrame.from_dict(cv_eval['cv_train'], orient='index', columns=[modelName]))
    errorPerModel_test = errorPerModel_test.combine_first(pd.DataFrame.from_dict(cv_eval['cv_test'], orient='index', columns=[modelName]))
    errorPerModel_val = errorPerModel_test.combine_first(pd.DataFrame.from_dict(cv_eval['val'], orient='index', columns=[modelName]))
    plotFile = os.path.join(outFolder, 'fit_plots', '{}_{}.png'.format(modelName, modelXY))
    plot_model([y_compl[-250:], cv_eval['preds'][-250:]], y_compl.index[endIndexTrain], plotFile)

    #TODO store optimal parameters and model predictions

    # old eval implementation based on training and test set
#    train_perf, test_perf, preds_df = eval_model(fitted_model, modelName, feature_df, y_compl, endIndexTrain)
#    print('\t'.join(['{}: {}'.format(k,round(v,3)) for (k,v) in train_perf.items()]))
#    print('\t '.join(['{}: {}'.format(k,round(v, 3)) for (k,v) in test_perf.items()]))
#    errorPerModel_train = errorPerModel_train.combine_first(pd.DataFrame.from_dict(train_perf, orient='index', columns=[modelName]))
#    errorPerModel_test = errorPerModel_test.combine_first(pd.DataFrame.from_dict(test_perf, orient='index', columns=[modelName]))
#    plotFile = os.path.join(outFolder, 'fit_plots', '{}_{}.png'.format(modelName, modelXY))
#    plot_model([y_compl[-250:], preds_df[-250:]], y_compl.index[endIndexTrain], plotFile)
# store errors
#outFile =  os.path.join(outFolder, 'eval_mats', 'evalTrain_{}_{}.csv'.format(label,modelXY))
#errorPerModel_train.to_csv(outFile)
#outFile =  os.path.join(outFolder, 'eval_mats', 'evalTest_{}_{}.csv'.format(label,modelXY))
#errorPerModel_test.to_csv(outFile)



########################################################################################################
##################### plot random forest feature importances ###########################################
########################################################################################################

importances = rf_fitted.feature_importances_
plt.figure()
plt.title('Feature importances')
plt.barh(range(feature_df.shape[1]), importances, align='center')
plt.yticks(range(feature_df.shape[1]), feature_df.columns)
plt.show()
