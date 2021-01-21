
import os

import numpy as np
import pandas as pd

"""

	Script for processing and saving Water Level data.

"""

LOCATION = 'TOR'
LABEL_NAME = 'waterLevel_' + LOCATION

INPUT_DATA_PATH = '../../data/SMHI/water_level_TOR.csv'
OUTPUT_DATA_PATH = '../data/cleanedFiles/' + LABEL_NAME + '.csv'

if __name__ == '__main__':

	df = pd.read_csv(
		INPUT_DATA_PATH,
		sep=';',
		skiprows=25910, # Read somewhere after 1970.
		names=[
			'timestamp',
			LABEL_NAME,
			'qual_' + LABEL_NAME,
			'dummy'
		],
		#parse_dates=['timestamp']
	)

	df = df.drop('dummy', axis=1)
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df = df[df['timestamp'] >= '2012-01-01']

	df.to_csv(
		OUTPUT_DATA_PATH,
		index=False
	)