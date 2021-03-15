# General utility functions

import pandas as pd
from datetime import timedelta
from sktime.performance_metrics.forecasting import mase_loss


def detect_resolution(data):
    """
    Detects the resolution of the provided data set, by determining the most common difference between
    adjacent data points
    :param data: the provided data frame
    :return: a timedelta of the likely resolution
    :raise ValueError: in case a dataset is provided that is too small
    """
    if len(data) < 2:
        raise ValueError("Insufficient data points available for forecast in provided data frame")

    # keep track of differences between adjacent data points
    res = (pd.Series(data.index[1:]) - pd.Series(data.index[:-1])).value_counts()

    # most common difference is at index 0
    return res.index[0]


def calculate_error(train, test, forecast):
    """
    Calculate the error between forecast and actual values.  Currently uses MASE from sktime package.
    :param train: pandas Series of training values
    :param test: pandas Series of testing (actual) values
    :param forecast: pandas Series of forecasted values
    :return: float indicating MASE
    """
    resolution = detect_resolution(train)
    return mase_loss(test, forecast, train, sp=int(timedelta(days=1)/resolution))
