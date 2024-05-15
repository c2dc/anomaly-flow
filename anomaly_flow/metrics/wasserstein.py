"""
    Module used to calculate the Wasserstein metric.
    The metric is known in computer science as the earth mover's distance.(EMD)
"""
from scipy.stats import wasserstein_distance

def calculate_wasserstein(series, comparison_series):
    """
        Function that calculates the Wasserstein metric.
    """
    return wasserstein_distance(series, comparison_series)
