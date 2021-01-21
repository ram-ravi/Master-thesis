
import os

import numpy as np
import pandas as pd

"""

	Script for processing and saving UV-data.

	http://strang.smhi.se/extraction/index.php
	CIE UV irradiance [mW/m^2]
	Note that the values are instantaneous and refer to the full hour (UTC). Swedish local time is UTC + 1 h during winter time and UTC + 2 h during the summer.

"""


LOCATION = 'LE'
LABEL_NAME = 'uv_' + LOCATION

INPUT_DATA_PATH = '../../data/SMHI/Add/uv_daily_Lilla_Edet.txt'
OUTPUT_DATA_PATH = '../data/cleanedFiles/uv_' + LOCATION + '.csv'

if __name__ == '__main__':

	df = pd.read_csv(
		INPUT_DATA_PATH,
		sep=' ',
		skiprows=4,
		names=['year', 'month', 'day', 'hour', LABEL_NAME]
	)

	df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day']])

	df = df.drop(
		[
			'year',
			'month',
			'day',
			'hour'
		],
		axis=1
	)

	# Remove -999
	df = df[df[LABEL_NAME] >= 0.0]

	df[['timestamp', LABEL_NAME]].to_csv(
		OUTPUT_DATA_PATH,
		index=False
	)