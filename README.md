# carbon-intensity-forecast
Package for forecasting carbon intensity


## Install

Clone this repository to your machine, and run

```bash
make local-install
```


---

## Example Usage
Examples of how this package can be used are included in the jupyter notebook [example_usage.ipynb](example_usage.ipynb).

---

## Generating Forecasts

The core functionality of this package is to provide the general-purpose function:

```python
def generate_forecast(data, model, start=None, end=None, resolution=None, params=None)
```

#### Regarding input data:
When the data is a pandas series or a dataframe having only a single column, then it is assumed that this is
carbon intensity time series data, and a straight carbon intensity forecast model is used.  The function returns
 a pandas Series (the forecast).

When the data is a pandas dataframe having multiple columns, then is it assumed that this dataset contains energy
generation mix data and a full energy generation mix forecast is used.  The function the returns a pandas DataFrame 
(of forecasts for every component of energy mix, and carbon intensity).

#### Regarding model:
The full set of forecast models supported should be listed in `model_names.py` and can be explored by running
```python
from ciforecast import CarbonIntensityForecastModels
for model in CarbonIntensityForecastModels:
    print(model)
```
When only a single model is passed (e.g. `CarbonIntensityForecastModels.ARIMA`), this same model is used for _all_
time series in the data.  When a dictionary is passed, different forecast models are used for different
components in the energy generation mix, for example:
`model = {'coal': CarbonIntensityForecastModels.ARIMA, 'solar': CarbonIntensityForecastModels.PROPHET}`

#### Regarding start, end:
Default behaviour is to provide an hourly, 24-hour ahead forecast.  However, if
`start is not None` then a different starting interval is used, and if `end is not None` then a different ending
interval is used.  If a gap results (between end of input data and start of requested forecast) then this gap is
filled with forecasted values as needed.

#### Regarding resolution:
The default is for the same resolution as the input data to be used.  However, a custom resolution can be
specified (e.g. `resolution=timedelta(minutes=15)`).  This is not yet tested and fully supported.

#### Regarding params:
Every forecasting model has some sort of parameters that need to be chosen.  For example, even a very simple
seasonal naive model will need to specify what period the seasonality is.  Any such parameters can be specified
in this argument, for example, like this:
`params = { 'trend': 'add', 'seasonal': 'add', 'damped_trend': 'True', 'sp': 24 }`
Currently this is not yet supported and at the moment (v0.1), model parameters are hard-coded.
