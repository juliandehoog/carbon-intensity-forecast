from enum import Enum


class CarbonIntensityForecastModels(str, Enum):
    """ Types of models that can be used to provide forecasts """

    # Models provided by sktime
    SEASONAL_NAIVE = "seasonal_naive"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    AUTO_ETS = "auto_ets"
    ARIMA = "arima"
    AUTO_ARIMA = "auto_arima"
    PROPHET = "prophet"

    # Custom models native to this package
    PERIODIC_PERSISTENCE = "periodic_persistence"
