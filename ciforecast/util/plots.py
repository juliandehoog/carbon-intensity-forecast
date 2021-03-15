# This file provides some basic plotting utilities for exploration of data and forecasts, e.g. when using
# jupyter notebooks.  These plots are not essential for this package and therefore plotly is not included in
# requirements.txt.

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px


def generate_standard_trace(series,
                            name=None,
                            color='blue',
                            width=2,
                            dash='solid',
                            showlegend=True):
    # Important: for every time series, remove timezone info before displaying - to make display more intuitive
    return go.Scatter(
        x=series.index.tz_localize(None),
        y=series.values,
        mode='lines',
        showlegend=showlegend,
        line=dict(
            color=color,
            width=width,
            dash=dash
        ),
        name=name,
    )


def generate_standard_layout(yaxis_title=None, height=150, width=None):
    layout = go.Layout(
        height=height,
        margin=go.layout.Margin(
            l=50,
            r=200,
            b=30,
            t=0,
            pad=0
        ),
        yaxis=dict(title=yaxis_title)
    )
    if width is None:
        return layout
    else:
        layout.update(width=width)
        return layout


def generate_connector(a, b, color='blue', dash='solid'):
    """
    Generates a small connecting line from one time series to another
    :return: plotly trace connecting the two time series
    """

    return go.Scatter(
        x=[a.index[-1].tz_localize(None), b.index[0].tz_localize(None)],
        y=[a.values[-1], b.values[0]],
        mode='lines',
        name='connector',
        showlegend=False,
        line=dict(
            color=color,
            dash=dash
        ),
    )


def plot_energy_mix(energy_mix, title=None, mode='lines'):
    fig = go.Figure()

    for production_type in energy_mix.columns:
        if production_type == 'carbon_intensity':
            continue
        fig.add_trace(go.Scatter(
            x=energy_mix.index,
            y=energy_mix[production_type].values,
            name=production_type,
            mode=mode,
        ))
    fig.update_yaxes(title='Generation (MW)')

    if title is not None:
        fig.update_layout(title=title)

    return fig


def plot_energy_mix_and_carbon_intensity(energy_mix, title=None, mode='lines'):
    fig = make_subplots(rows=2, cols=1, subplot_titles=['Energy generation mix', 'Carbon intensity'],
                        shared_xaxes=True, row_heights=[0.7, 0.3])

    for production_type in energy_mix.columns:
        if production_type == 'carbon_intensity':
            continue
        fig.add_trace(go.Scatter(
            x=energy_mix.index,
            y=energy_mix[production_type].values,
            name=production_type,
            mode=mode,
        ), row=1, col=1)
    fig.update_yaxes(title='Generation (MW)', row=1, col=1)

    fig.add_trace(go.Scatter(
        x=energy_mix.index,
        y=energy_mix['carbon_intensity'],
        name='carbon intensity',
        showlegend=False,
        mode=mode,
    ), row=2, col=1)
    fig.update_yaxes(title='Carbon intensity (g CO2e)', row=2, col=1)

    if title is not None:
        fig.update_layout(title=title)

    return fig


def plot_single_time_series_with_fc(train, forecast, actual=None,
                                    name=None, fcname=None, connect=True,
                                    show_from=None):
    """
    Plot single time series together with a forecast
    :param train: pandas series
    :param forecast: pandas series
    :param actual: (optional) pandas series
    :param name: optional string indicating name of the data being forecasted
    :param fcname: optional string indicating name of the forecast (e.g. forecast model used)
    :param connect: boolean indicating whether to draw line from end of data to start of forecast
    :param show_from: optional timestamp indicating from which interval to show
    :return: plotly Figure
    """
    if name is None:
        name = 'data'
    if fcname is None:
        fcname = 'forecast'

    # reduce amount of data to show if argument provided
    if show_from is not None:
        train = train[show_from:]

    # data
    trace_data = generate_standard_trace(train, name=name)

    # forecast
    trace_fc = generate_standard_trace(forecast, name=fcname, color='orange')

    all_traces = [trace_data, trace_fc]

    if actual is not None:
        trace_actual = generate_standard_trace(actual, name=name,
                                               dash='dot', showlegend=False)
        all_traces.append(trace_actual)

    if connect:
        # Small line connecting end of data to start of forecast
        trace_connector = generate_connector(train, forecast, color='orange')
        all_traces.append(trace_connector)

    layout = generate_standard_layout()

    return go.Figure(data=all_traces, layout=layout)


def plot_full_mix_with_fc(train, forecast, actual=None, connect=True, show_from=None):
    """
    Plot full energy mix and carbon intensity together with all forecasts
    :param train: pandas series
    :param forecast: pandas series
    :param actual: (optional) pandas series
    :param connect: boolean indicating whether to draw line from end of data to start of forecast
    :param show_from: optional timestamp indicating from which interval to show
    :return: plotly Figure
    """
    num_rows = len(forecast.columns)
    fig = make_subplots(rows=num_rows, cols=1,
                        #column_widths=[0.9, 0.3],
                        #horizontal_spacing=0.05,
                        shared_xaxes=True, shared_yaxes=True)

    # reduce amount of data to show if argument provided
    if show_from is not None:
        train = train[show_from:]

    # set colours
    colors = px.colors.qualitative.Plotly

    # Data - production types
    row_ix = 1
    for production_type in forecast.columns:
        if production_type == 'carbon_intensity':
            continue
        fig.add_trace(go.Scatter(
            x=train.index,
            y=train[production_type].values,
            name=production_type,
            mode='lines',
            showlegend=True,
            line=dict(
                color=colors[row_ix],
                width=2,
                dash='solid',
            ),
        ), row=row_ix, col=1)
        fig.update_xaxes(showticklabels=False, row=row_ix, col=1)
        row_ix = row_ix + 1

    # Extract trace colours for forecast use
    trace_colours = []

    # Data - carbon intensity
    fig.add_trace(go.Scatter(
        x=train.index,
        y=train['carbon_intensity'],
        name='carbon_intensity',
        showlegend=True,
        mode='lines',
        line=dict(
            color='blue',
            width=2,
            dash='solid',
        ),
    ), row=num_rows, col=1)
    fig.update_xaxes(showticklabels=True, row=row_ix, col=1)

    # Forecasts - production types
    row_ix = 1
    for production_type in forecast.columns:
        if production_type == 'carbon_intensity':
            continue
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast[production_type].values,
            name=production_type,
            showlegend=False,
            mode='lines',
            line=dict(
                color='black',
                width=2,
                dash='solid',
            ),
        ), row=row_ix, col=1)
        fig.update_xaxes(showticklabels=False, row=row_ix, col=1)
        row_ix = row_ix + 1

    # Forecasts - carbon intensity
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast['carbon_intensity'],
        name='forecast',
        showlegend=True,
        mode='lines',
        line=dict(
            color='black',
            width=2,
            dash='solid',
        ),
    ), row=num_rows, col=1)
    fig.update_xaxes(showticklabels=True, row=row_ix, col=1)

    if actual is not None:
        # Actuals - production types
        row_ix = 1
        for production_type in forecast.columns:
            if production_type == 'carbon_intensity':
                continue
            fig.add_trace(go.Scatter(
                x=actual.index,
                y=actual[production_type].values,
                name=production_type,
                showlegend=False,
                mode='lines',
                line=dict(
                    color=colors[row_ix],
                    width=2,
                    dash='dash',
                ),
            ), row=row_ix, col=1)
            fig.update_xaxes(showticklabels=False, row=row_ix, col=1)
            row_ix = row_ix + 1

        # Actuals - carbon intensity
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual['carbon_intensity'],
            name='actual',
            showlegend=False,
            mode='lines',
            line=dict(
                color='blue',
                width=2,
                dash='dash',
            ),
        ), row=num_rows, col=1)
        fig.update_xaxes(showticklabels=True, row=row_ix, col=1)

    layout = generate_standard_layout(height=700)
    fig.update_layout(layout)

    return fig
