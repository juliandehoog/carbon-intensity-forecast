from ciforecast.models._base_class import CarbonIntensityBaseForecastModel
import numpy as np
import pandas as pd


class PeriodicPersistence(CarbonIntensityBaseForecastModel):
    """ Simple model that takes (possibly weighted) average of last several periods """

    def __init__(self, period=24, num_periods=1, weights=None):
        """
        Initialise model
        :param period: <string>, period over which the persistence forecast should persist
                                    (default 24 hours)
        :param num_periods: <int>, number of previous periods to average over
        :param weights: list of <int>, how much weight should be assigned to each of
                       the preceding periods, in reverse order.  For example, 'weights': [2, 1]
                       means that the preceding period has double the impact of the second-last
                       period when calculating the periodic persistence forecast.
        :return: None
        """
        super().__init__()
        self.period = period
        self.num_periods = num_periods
        if weights is None:
            weights = [1]
        # Confirm there is one weight per period
        if len(weights) != self.num_periods:
            raise ValueError("When providing weights, there must be as many weights \
                                           in the array as there are periods")
        self.weights = weights
        self.data = None

    def fit(self, data):
        """
        Fit a model to data provided.  In this case there is no particular model, just store the data.
        :param data: pandas Series
        :return: None
        """
        super().fit(data)
        self.data = data

    def predict(self, forecast_indices):
        """
        Provide forecasts for requested forecast indices
        # TODO This does not yet use the same standard approach as sktime forecasting models for handling
        #      forecast indices
        :param forecast_indices: array of indices of forecast values to be returned
        :return: pandas series of forecast values at requested indices
        """
        super().predict(forecast_indices)
        fc_values = np.zeros(forecast_indices[-1]+1)
        # Loop until maximum forecast index reached
        for fc_ix in range(0, forecast_indices[-1]+1):
            curr_value = 0.0
            sum_weights = 0.0
            # Loop through recent periods
            for period_ix in range(1, self.num_periods + 1):
                historical_ix = -1 * (period_ix * self.period) + fc_ix % self.period
                historical_value = self.data.iloc[historical_ix]
                if historical_value == 0:   # Ignore zero values when calculating weighted sum
                    continue
                else:
                    curr_value = curr_value + historical_value * self.weights[period_ix-1]
                    sum_weights = sum_weights + self.weights[period_ix-1]
            if sum_weights == 0:
                fc_values[fc_ix] = float('NaN')
            else:
                fc_values[fc_ix] = curr_value / sum_weights
        return pd.Series(fc_values[forecast_indices])
