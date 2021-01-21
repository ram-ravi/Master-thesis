
import pathlib

import pandas as pd
import numpy as np



'''

	Methods for reading data.

'''


INPUT_DATA_PATH = pathlib.Path(__file__).parent / 'data' / 'cleanedFiles'


def read_ecoli_lab(label, log_scale=False):

	#
	# E.coli Lab Measurements
	#

	filename = label + '.csv'
	
	df = pd.read_csv(
		INPUT_DATA_PATH / filename,
		parse_dates=['timestamp']
	)
	
	df.index = pd.DatetimeIndex(df['timestamp'].dt.date)
	df = df.drop(labels='timestamp', axis=1)
	df = df[~df.index.duplicated(keep='last')] # Drop duplicates on same day, keep last.
	df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
	
	new_label = label.split('_')[0] + '_lab_' + label.split('_')[1]
	df = df.rename(columns={label: new_label})

	if log_scale:
		df[new_label] = np.log1p(df[new_label])
	
	return df

def read_ecoli_colifast(label):

	#
	# E.coli Colifast
	#
	
	colifast_map = {
		'<50': 1,
		'50': 2,
		'100': 3,
		'200': 4,
		'400': 5,
		'>400': 6
	}
	
	filename = label + '.csv'
	
	df = pd.read_csv(
		INPUT_DATA_PATH / filename,
		parse_dates=['timestamp']
	)
	
	df['ds'] = df['timestamp'].dt.date
	df = df.groupby('ds').nth([0, -1]) # Keep first and last measurement every day.
	df = df.drop(labels='timestamp', axis=1)
	df[label] = df[label].map(colifast_map).astype(float)
	df = df.rename(columns={label: 'ecoli_' + label})
	
	df_first = df.groupby('ds').nth(0)
	df_first = df_first.reindex(pd.date_range(df_first.index.min(), df_first.index.max(), freq='D'))
	
	df_second = df.groupby('ds').nth(-1)
	df_first = df_first.reindex(pd.date_range(df_first.index.min(), df_first.index.max(), freq='D'))
	
	df = df_first.join(df_second, how='outer', lsuffix='_1st', rsuffix='_2nd')
	
	return df

def read_precipitation(label, is_daily_precip=True):

	#
	# Precipitation
	#
	
	filename = label + '.csv'
	
	df = pd.read_csv(
		INPUT_DATA_PATH / filename,
		parse_dates=['timestamp']
	)
	
	if is_daily_precip:
		df.index = pd.DatetimeIndex(df['timestamp'].dt.date)
		df = df.drop(['timestamp', 'qual_' + label], axis=1)
		df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
		df[label] = df[label].shift(-1)
		
	else:
		df.index = pd.DatetimeIndex(df['timestamp'])
		df = df.drop(['timestamp', 'qual_' + label], axis=1)        
		df = df.groupby(pd.Grouper(freq = '24H', offset='6H', closed='right')).sum()
		df.index = pd.DatetimeIndex(df.index.date)
		df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
	
	return df

def read_water_temp(label):

	#
	# Water Temp
	#
	
	filename = label + '.csv'
	
	df = pd.read_csv(
		INPUT_DATA_PATH / filename,
		parse_dates=['timestamp']
	)
	
	df.index = pd.DatetimeIndex(df['timestamp'].dt.date)
	df = df.drop(labels='timestamp', axis=1)
	df = df[~df.index.duplicated(keep='last')] # Keep last measurement every day.
	df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
	
	return df


def read_flow_rate(label):

	#
	# Flow Rate
	#
	
	filename = label + '.csv'
	
	df = pd.read_csv(
		INPUT_DATA_PATH / filename,
		parse_dates=['timestamp']
	)
	
	df.index = pd.DatetimeIndex(df['timestamp'].dt.date)
	df = df.drop(labels='timestamp', axis=1)
	df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
	
	return df


def read_water_level(label):

	#
	# Water Level
	#
	
	filename = label + '.csv'
	
	df = pd.read_csv(
		INPUT_DATA_PATH / filename,
		parse_dates=['timestamp']
	)
	
	df['ds'] = df['timestamp'].dt.date
	df = df.drop(['timestamp', 'qual_' + label], axis=1)
	df = df.groupby('ds').max().reset_index()
	df.index = pd.DatetimeIndex(df['ds'])
	df = df.drop(['ds'], axis=1)
	df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
	
	return df


def read_turbidity(label):

	#
	# Turbidity
	#
	
	filename = label + '.csv'
	
	df = pd.read_csv(
		INPUT_DATA_PATH / filename,
		parse_dates=['timestamp']
	)
	
	df.index = pd.DatetimeIndex(df['timestamp'].dt.date)
	df = df.drop(labels='timestamp', axis=1)
	df = df[~df.index.duplicated(keep='last')]
	df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
	
	return df


def read_coliforms(label, log_scale=False):
	
	#
	# Coliforms
	#

	filename = label + '.csv'
	
	df = pd.read_csv(
		INPUT_DATA_PATH / filename,
		parse_dates=['timestamp']
	)
	
	df.index = pd.DatetimeIndex(df['timestamp'].dt.date)
	df = df.drop(labels='timestamp', axis=1)
	df = df[~df.index.duplicated(keep='last')]
	df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
	
	if log_scale:
		df[label] = np.log1p(df[label])
	
	return df

