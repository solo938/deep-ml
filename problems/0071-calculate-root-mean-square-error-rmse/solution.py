
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	
	mse = np.mean((y_true -  y_pred) ** 2)
	rmse_res = np.sqrt(mse)
	return round(rmse_res,3)
