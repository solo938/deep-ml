import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)


    standardized_data = (data - mean) / std


    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    
    normalized_data = (data - data_min) / (data_max - data_min)

    
    standardized_data = np.round(standardized_data, 4)
    normalized_data = np.round(normalized_data, 4)

    return standardized_data, normalized_data