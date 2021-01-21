import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime

from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import TimeSeriesSplit

from aux_functions_forecasting import *


####################### Functions ########################################

# naive (constant) model propagating the previous value x days into the future
def naive_model(dfAllDates, forecastRange=1):
	pred_naive = dfAllDates[:]
	pred_naive.index = pred_naive.index.shift(forecastRange, freq='D')
	pred_naive.rename(columns = {pred_naive.columns[0]:'predictions'}, inplace=True)
	return pred_naive
##########################################################################


input_file = '../data/cleanedFiles/ecoli_GA.csv'
start_training = '02-01-2012'
end_training = '04-05-2018'
start_testing = '2018-04-06'
end_testing   = '12-30-2019'

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d")

df_original = pd.read_csv(input_file, parse_dates=['timestamp'], date_parser=dateparse)
df_original.index = pd.DatetimeIndex(df_original['timestamp']).floor('D')
df_original = df_original.drop(labels='timestamp', axis=1)
# must remove duplicate entries (when more than one sample in one day: keep the latest)
df_original = df_original.drop_duplicates(subset='timestamp', keep='last')

all_days = pd.date_range(df.index.min(), df.index.max(), freq='D')

df = df_original
# include all days in interval in dataframe, filling na's with the previous existing value
df = (df.reindex(all_days).fillna(method='ffill')) 

train = df[start_training : end_training]
test = df[start_testing : end_testing]


# Checking stationarity of time series
#df.plot()
#plt.show(block=True)
# seasonal trend! --> should use differencing

# ARIMA model parameters:
# lag observations
# times to difference the raw observations to make the time series stationary (remove trend and seasonality)
# size of moving average window
predictions = list()

for t in test.iterrows() :
	model = ARIMA(train, order=(5,1,2))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.extend(yhat)
	obs = t[1]
#    print(obs)
	train = train.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

# make df assigning predictions to dates
pred_df = pd.DataFrame(predictions, index = test.index, columns=['predictions'])
# filter for dates with real data
pred_df_filtered = pred_df.loc[df_original[start_testing : end_testing].index]

# calculate error
test_original = df_original[start_testing : end_testing]
perf_arima = obtain_performance(test_original['ecoli_GA'].values, pred_df_filtered['predictions'].values)

# plot predictions
plotFile = './arima.png'
plot_model([df_original[-250:], pred_df_filtered], pd.to_datetime(start_testing), plotFile)


#output = model_fit.forecast()
#print(output)
#print(model_fit.summary())
# plot residual errors
#residuals = DataFrame(model_fit.resid)
#residuals.plot()
#plt.show()
#residuals.plot(kind='kde')
#plt.show()
#print(residuals.describe())

## naive model with one-day forecast
forecastRange = 1
pred_naive = naive_model(df, forecastRange)
# filter for dates with real values available
pred_naive_filtered = pred_naive.loc[df_original[forecastRange:].index]

# calculate error
guess_naive = pred_naive_filtered[start_testing : end_testing]
perf_naive = obtain_performance(test_original['ecoli_GA'].values, guess_naive['predictions'].values)

# plot predictions
plotFile = './naive_1day.png'
plot_model([df_original[-250:], pred_naive_filtered[-250:]], pd.to_datetime(start_testing), plotFile)


