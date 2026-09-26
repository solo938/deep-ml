
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	mean = np.mean(y_true)

	ss_res = np.sum((y_true - y_pred) ** 2)
	ss_total = np.sum((y_true - mean) ** 2)

	r_squared = 1 - ss_res / ss_total

	return r_squared
