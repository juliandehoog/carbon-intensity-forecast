# Standard packages
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

# sktime forecasting models
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.fbprophet import Prophet

# local forecasting models
from ciforecast.models.periodic_persistence import PeriodicPersistence

# Other local imports
from ciforecast.model_names import CarbonIntensityForecastModels
from ciforecast.util.util import detect_resolution
from ciforecast.carbon_intensity import VALID_ENERGY_MIX_COLUMNS, get_carbon_intensity_time_series


def _make_timezone_naive_utc(datetime_object):
    """ If a given datetime is timezone-aware, convert to UTC and make timezone=naive """

    if (datetime_object.tzinfo is not None) and (datetime_object.tzinfo.utcoffset(datetime_object) is not None):
        # Convert to UTC
        utc_time = datetime_object.astimezone(pytz.utc)

        # Remove the timezone attribute to make it timezone-naive
        utc_time = utc_time.replace(tzinfo=None)

        return utc_time

    else:
        return datetime_object


def _generate_timing_params(data, start=None, end=None, custom_resolution=None):
    """
    Generate some helpful parameters related to the timing of the forecast
    :param data: <pandas Series or Dataframe> input data
    :param start: <timestamp> start of desired forecast
    :param end: <timestamp> end of desired forecast
    :param custom_resolution: <datetime interval> requested custom resolution for forecast
    :return: dict as follows (where all timestamps are timezone-naive UTC-based):
        {
            'resolution': <datetime interval> resolution of forecast,
            'forecast_start': <timestamp> start of forecast that will be returned,
            'forecast_end': <timestamp> end of forecast that will be returned,
            'timestamps': <list of timestamps> all timestamps of forecast that will be returned,
            'gap_start': <timestamp> start of actual generated forecast, which may be earlier than start of
                         forecast that will be returned,
            'full_length': <int> length of full forecast including possible gap at start,
        }
    """

    # Determine resolution of forecast
    if custom_resolution is not None:
        resolution = custom_resolution
    else:
        resolution = detect_resolution(data)

    # Set forecast start and end.  Default behaviour is 24 hours from end of last data point.
    forecast_start = (data.index[-1] + resolution) if (start is None) else start
    forecast_end = (data.index[-1] + timedelta(hours=24)) if (end is None) else end

    # If either start or end is timezone aware, convert to utc and make timezone naive, for consistency
    forecast_start = _make_timezone_naive_utc(forecast_start)
    forecast_end = _make_timezone_naive_utc(forecast_end)

    # determine timestamps that will be associated with returned forecast values
    timestamps = [int(pytz.utc.localize(dt).timestamp())
                  for dt in pd.Series(pd.date_range(forecast_start, forecast_end, freq=resolution))]

    # Check whether the forecast start is same as the first step after the training data.
    # If it's not the same, then that means that there is a gap before the forecast start data.
    # We want to which of the "steps ahead" are the ones we need for the requested forecast period.
    # For example, this could be from 5 steps ahead to 29 steps ahead, from last input data interval.
    first_forecast_interval = _make_timezone_naive_utc(data.index[-1] + resolution)
    if first_forecast_interval != forecast_start:
        gap_size = int((forecast_start - first_forecast_interval) / resolution)
    else:
        gap_size = 0
    forecast_size = int((forecast_end - forecast_start) / resolution) + 1
    forecast_indices = np.arange(gap_size, gap_size + forecast_size)

    return {
        'resolution': resolution,
        'timestamps': timestamps,
        'indices': forecast_indices,
    }


def _extract_time_series(data, time_series_name='values'):
    """
    Extract specific time series from provided pandas DataFrame and return as a pandas Series
    """

    # If it's already a series, just return it
    if isinstance(data, pd.Series):
        return data

    if time_series_name not in data:
        raise ValueError("Could not locate column {} in provided data".format(time_series_name))

    return pd.Series(data[time_series_name])


def generate_forecast_for_single_time_series(series, model, start=None, end=None, resolution=None, params=None):
    """
    Generate a forecast for a single time series.
    For full parameter descriptions see `generate_forecast` below.

    :return: pandas dataframe containing timestamps (index) and one columns of forecast values
    """

    # Generate params related to forecast timing
    timing_params = _generate_timing_params(series, start, end, resolution)

    # Create forecaster.  Every forecast model has some types of parameters that must be specified
    # Some of these forecasting models are standard models from other packages (like sktime), others are unique
    # to this package.  See imports at top of this file.
    # TODO: For now many of these parameters remain hard coded.
    #       It would be nice if they could have defaults but could also be passed in params.
    seasonal_period = int(timedelta(days=1) / timing_params['resolution'])
    forecaster = None

    # Existing sktime models
    if model == CarbonIntensityForecastModels.SEASONAL_NAIVE:
        forecaster = NaiveForecaster(strategy="last", sp=seasonal_period)
    elif model == CarbonIntensityForecastModels.EXPONENTIAL_SMOOTHING:
        forecaster = ExponentialSmoothing(trend="add", seasonal="add", damped_trend=True, sp=seasonal_period)
    elif model == CarbonIntensityForecastModels.AUTO_ETS:
        forecaster = AutoETS(auto=True, sp=seasonal_period, n_jobs=-1)
    elif model == CarbonIntensityForecastModels.AUTO_ARIMA:
        forecaster = AutoARIMA(sp=seasonal_period, suppress_warnings=True)
    elif model == CarbonIntensityForecastModels.ARIMA:
        forecaster = ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, seasonal_period), suppress_warnings=True)
    elif model == CarbonIntensityForecastModels.PROPHET:
        forecaster = Prophet(
            seasonality_mode="multiplicative",
            n_changepoints=int(len(series) / seasonal_period),
            add_country_holidays={"country_name": "Germany"},
            yearly_seasonality=True,
        )

    # Custom models in this package
    elif model == CarbonIntensityForecastModels.PERIODIC_PERSISTENCE:
        forecaster = PeriodicPersistence(period=seasonal_period, num_periods=3, weights=[3, 2, 1])

    # Fit data. Note that we remove timezone info since this can cause issues with some sktime models.
    forecaster.fit(series.tz_convert(None))

    # Generate forecast
    forecast_values = forecaster.predict(timing_params['indices'])

    # Reformat forecast to return the interval requested
    return pd.Series(
        index=timing_params['timestamps'],
        data=forecast_values.values[-len(timing_params['timestamps']):]
    )


def generate_forecast_from_ci(data, model, start=None, end=None, resolution=None, params=None):
    """
    Generate a carbon intensity forecast using the carbon intensity data provided.  For full parameter descriptions
    see `generate_forecast` below.

    The argument 'params' is a dict that can contain the following:
        General
        - 'column_name': <str> name of column to use for carbon intensity data (default None)
        - TODO: add remaining param options here

    :return: pandas dataframe containing timestamps (index) and one columns of forecast values
    """
    if (params is None) or ('column_name' not in params):
        column_name = 'carbon_intensity'
    else:
        column_name = params['column_name']

    # Extract carbon intensity data
    data_ci = _extract_time_series(data, column_name)

    return generate_forecast_for_single_time_series(data_ci, model,
                                                    start=start, end=end, resolution=resolution,
                                                    params=params)


def generate_forecast_from_mix(data, model, start=None, end=None, resolution=None, params=None):
    """
    Generate a carbon intensity forecast using the energy generation mix data provided.  For parameter descriptions
    see `generate_forecast` below.

    :return: pandas dataframe containing timestamps (index) and one columns of forecast values
    """
    # Get all relevant columns
    column_names = []
    for column_name in data:
        if column_name in VALID_ENERGY_MIX_COLUMNS:
            column_names.append(column_name)

    # Set forecasting model for each component of energy mix
    fc_models = {}
    if isinstance(model, CarbonIntensityForecastModels):  # if a single model is provided as argument
        for column_name in column_names:
            fc_models[column_name] = model
    elif isinstance(model, dict):  # if individual models provided for each energy mix type
        fc_models = model
    else:
        raise ValueError("Argument `model` must be either a dict or of type CarbonIntensityForecastModels")

    # TODO Should probably check here that every valid column has an associated forecast model
    # TODO Currently this function does not allow for individual parameter settings for each forecasting model

    # Generate all forecasts
    forecasts = {}
    for column_name in fc_models:
        series = _extract_time_series(data, column_name)
        forecasts[column_name] = generate_forecast_for_single_time_series(series, fc_models[column_name],
                                                                          start=start, end=end, resolution=resolution,
                                                                          params=params)

    # Calculate carbon intensity forecast
    forecasts['carbon_intensity'] = get_carbon_intensity_time_series(pd.DataFrame(forecasts))

    # Return pandas dataframe
    return pd.DataFrame(forecasts)


def generate_forecast(data, model, start=None, end=None, resolution=None, params=None):
    """
    Generate a carbon intensity forecast based on the data provided.

    Regarding input data:
    When the data is a pandas series or a dataframe having only a single column, then it is assumed that this is
    carbon intensity time series data, and a straight carbon intensity forecast model is used.
    When the data is a pandas dataframe having multiple columns, then is it assumed that this dataset contains energy
    generation mix data and a full energy generation mix forecast is used.

    Regarding model:
    The full set of forecast models supported should be listed in `model_names.py` and can be explored by calling
    `CarbonIntensityForecastModels.list_all()`.
    When only a single model is provided (e.g. CarbonIntensityForecastModels.ARIMA), this same model is used for _all_
    time series in the data.  When a dictionary is provided, different forecast models are used for different
    components in the energy generation mix, for example:
    `model = {'coal': CarbonIntensityForecastModels.ARIMA, 'solar': CarbonIntensityForecastModels.PROPHET}`

    Regarding start, end:
    Default behaviour is to provide an hourly, 24-hour-ahead forecast.  However, if
    `start` is not None then a different starting interval is used, and if `end` is not None then a different ending
    interval is used.  If a gap results (between end of input data and start of requested forecast) then this gap is
    filled with forecasted values as needed.

    Regarding resolution:
    The default is for the same resolution as the input data to be used.  However, a custom resolution can be
    specified (e.g. `resolution=timedelta(minutes=15)`).  This is not yet tested and fully supported.

    Regarding params:
    Every forecasting model has some sort of parameters that need to be chosen.  For example, even a very simple
    seasonal naive model will need to specify what period the seasonality is.  Any such parameters can be specified
    in this argument, for example, like this:
    `params = { 'trend': 'add', 'seasonal': 'add', 'damped_trend': 'True', 'sp': 24 }`

    :param data: <pandas Series or pandas Dataframe> input data
    :param model: <CarbonIntensityForecastModel> forecasting model to use
    :param start: <timestamp> start of requested forecast, default None
    :param end: <timestamp> end of requested forecast, default None
    :param resolution: <timedelta> resolution of requested forecast, default hourly
    :param params: <dict> any relevant forecast parameters
    :return: pandas dataframe containing timestamps (index) and one columns of forecast values
    """
    if isinstance(data, pd.Series) or (isinstance(data, pd.DataFrame) and len(data.columns) == 1):
        return generate_forecast_from_ci(data, model, start, end, resolution, params)
    else:
        return generate_forecast_from_mix(data, model, start, end, resolution, params)
