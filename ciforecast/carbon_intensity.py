import pandas as pd
import math

# source:
# https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources#2014_IPCC,_Global_warming_potential_of_selected_electricity_sources
CARBON_INTENSITIES = {
    "coal": 820.0,
    "gas": 490.0,
    # separate source for oil:
    # https://en.wikipedia.org/wiki/Emission_intensity#Energy_sources_emission_intensity_per_unit_of_energy_generated
    "oil": 893.0,
    "wind": 11.5,
    "hydro": 24.0,
    "solar": 41.0,  # concentrated 27, rooftop 41, utility 48
    "nuclear": 12.0,
    "geothermal": 38.0,
    "biomass": 230.0,
    "bio": 230.0,  # - alias for biomass
}

VALID_ENERGY_MIX_COLUMNS = CARBON_INTENSITIES.keys()


def get_carbon_intensity(energy_mix: dict) -> float:
    """
    Computes the average carbon intensity for a given energy mix.
    :param energy_mix: The energy mix provided as dictionary with keys as outlined in the CARBON_INTENSITIES dictionary.
    :return: the average carbon intensity in gCO2eq/kWh
    """
    total = 0.0
    sum_energy = 0.0
    for key in energy_mix:
        if isinstance(energy_mix[key], pd.Series):
            # special handling for None values in series
            values = energy_mix[key].values
            if values is not None and len(values) == 1 and values[0] is None:
                continue  # skip None columns

        if key in CARBON_INTENSITIES and energy_mix[key] is not None and not math.isnan(energy_mix[key]):
            total += float(energy_mix[key]) * CARBON_INTENSITIES[key]
            sum_energy += float(energy_mix[key])
    if sum_energy == 0.0:  # Avoid division by zero
        return float('NaN')
    return total / sum_energy


def get_carbon_intensity_time_series(energy_mix_time_series: pd.DataFrame) -> pd.Series:
    """
    Converts a full energy mix time series into a carbon intensity time series
    Input is a pandas dataframe.
    Assumes for now that index is timestamps and column names match CARBON_INTENSITIES keys
    Returns a pandas series with index of timestamps
    """
    current_mix = {}
    carbon_intensity_values = []
    for timestamp, row in energy_mix_time_series.iterrows():
        # Create dictionary of <production_type>: <value> pairs
        for production_type in energy_mix_time_series.columns:
            current_mix[production_type] = row[production_type]
        carbon_intensity_values.append(get_carbon_intensity(current_mix))
    return pd.Series(
        index=energy_mix_time_series.index,
        data=carbon_intensity_values,
    )
