
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

def obtain_performance(model_name, y_true, y_pred):
	
	y_true, y_pred = np.array(y_true), np.array(y_pred)

	rmse = np.sqrt((1/len(y_true))*sum((y_true-y_pred)**2))
	#mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-5, y_true))) * 100
	smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
	mae = (1/len(y_true))*sum(abs((y_true-y_pred)))
	r2 = r2_score(y_true, y_pred)
	return {
		'Model': model_name,
		'MAE':mae,
		'SMAPE': smape,
		'RMSE':rmse,
		'R2':r2
	}

def highlight_cells(s):
	
	if s.name in ['MAE', 'SMAPE', 'RMSE']:
		is_min = s == s.min()
		return ['color: green' if v else '' for v in is_min]
	elif s.name in ['R2']:
		is_max = s == s.max()
		return ['color: green' if v else '' for v in is_max]
	else:
		return ['color: black' for v in s]

