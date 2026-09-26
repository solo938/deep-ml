import numpy as np
import math
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    counts = Counter(y_train)
    sorted_class = sorted(counts.keys())

    largest_count = 0
    most_frequent = None

    for class_, count in counts.items():
        if count > largest_count:
            largest_count = count
            most_frequent = class_
    
    if strategy == "most_frequent" :
        predictions = [most_frequent] * n_test
        return  predictions

    if strategy == "constant" :
        predictions = [constant] * n_test
        return  predictions
    
    if strategy == "uniform" :
        predictions = []

        for i in range(n_test):
            index = i % len(sorted_class)
            predictions.append(sorted_class[index])
        
        return  predictions
    
    if strategy == "stratified":
        total = len(y_train)

        allocations = []
        fractional_parts = []

        for class_ in sorted_class:
            count = counts[class_]

            exact = n_test * count / total
            allocation = math.floor(exact)
            fraction = exact - allocation

            allocations.append(allocation)
            fractional_parts.append(fraction)
        
        remaining = n_test - sum(allocations)

        order = sorted(
            range(len(sorted_class)),
            key=lambda i: (-fractional_parts[i], sorted_class[i])
        )

        for i in order[:remaining]:
            allocations[i] += 1
        
        predictions = []

        for i, class_ in enumerate(sorted_class):
            predictions.extend([class_] * allocations[i])
        
        return  predictions
    



        


        



        





