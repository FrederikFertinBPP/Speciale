""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data_scripts.data_generator_v2 import DataForecaster
from common_scripts import cache_read
from model_scripts.environment import EmissionFactorEstimator

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

forecaster = DataForecaster(from_pickle=True, cache_id="Anders")
forecaster = forecaster.unpickle()
t_start = pd.to_datetime("2024-01-01", utc=True)
t_end = t_start + relativedelta(years=+1) + pd.Timedelta(1, 'hour') # Episodic implementation

decision_horizon = 24
planning_horizon = 4 * decision_horizon

# Calculates "Carbon intensity gCO₂eq/kWh (direct)" as a linear function of price [€/MWh], system wind [MW], and system solar [MW].
cache_path_mappers = os.getcwd() + "/models/plant_models/"
mapper = cache_read(cache_path_mappers + "emission_factor.pkl")
emissions_model = EmissionFactorEstimator(mapper)

n_scenarios = 50

root_dir = "scenario_data"
date_format="%Y-%m-%d %H"
float_format='%.3f'
capture_prices = {}
capture_emissions = {}

for n in range(n_scenarios):
    scenario_dir = os.path.join(root_dir, f"Anders_scenario_{n}")
    os.makedirs(scenario_dir, exist_ok=True)
    
    solar_year = []
    wind_year = []
    price_year = []
    emissions_year = []
    time_index_year = []

    t = t_start
    while t < t_end:
        timestamp_str = t.strftime("%Y%m%d")
        hourly_index = pd.to_datetime(pd.date_range(t, t + pd.Timedelta(23, 'hour'), freq='h'), utc=True)
        time_index_year += list(hourly_index)
        solar_year += list(pd.read_csv(f"{scenario_dir}/solar_{timestamp_str}.csv")["solar"].values)
        wind_year += list(pd.read_csv(f"{scenario_dir}/wind_{timestamp_str}.csv")['wind'].values)
        price_year += list(pd.read_csv(f"{scenario_dir}/prices_{timestamp_str}.csv")['price'].values)
        # system_solar_realization, system_wind_realization = forecaster.realize_vre(start=t, end=t + pd.Timedelta(decision_horizon-1, 'h'))
        # real_prices = forecaster.realize_prices(start=t, end=t+pd.Timedelta(decision_horizon-1, 'h'))
        year_month_index = hourly_index.tz_localize(None).to_period('M')
        solar_capacities = year_month_index.map(forecaster.database.caps['solar'])
        wind_capacities = year_month_index.map(forecaster.database.caps['wind'])
        solar = np.asarray(solar_year[-24:]) * solar_capacities
        wind = np.asarray(wind_year[-24:]) * wind_capacities
        price = np.asarray(price_year[-24:])
        X = pd.DataFrame(data={"price":price, "wind":wind, "solar":solar})
        emissions = emissions_model(X) / 1000 # Convert to unit tCO2/MWh.
        emissions_year += list(emissions)
        emissions_file = os.path.join(scenario_dir, f"emissions_{timestamp_str}.csv")
        pd.DataFrame(data={"emissions":emissions}).to_csv(emissions_file, index=False, date_format=date_format, float_format=float_format)

        t += pd.Timedelta(24, 'hour')
    sim_cp = {}
    sim_cp["wind"] = np.sum(np.asarray(wind_year) * np.asarray(price_year)) / np.sum(np.asarray(wind_year))
    sim_cp["solar"] = np.sum(np.asarray(solar_year) * np.asarray(price_year)) / np.sum(np.asarray(solar_year))
    sim_cp["baseload"] = np.mean(np.asarray(price_year))
    sim_ef = {}
    sim_ef["wind"] = np.sum(np.asarray(wind_year) * np.asarray(emissions_year)) / np.sum(np.asarray(wind_year))
    sim_ef["solar"] = np.sum(np.asarray(solar_year) * np.asarray(emissions_year)) / np.sum(np.asarray(solar_year))
    sim_ef["baseload"] = np.mean(np.asarray(emissions_year))
    capture_prices[str(n)] = sim_cp
    capture_emissions[str(n)] = sim_ef
    df = pd.DataFrame(data={"price": price_year, "wind": wind_year, "solar": solar_year, "emissions": emissions_year})
    path = os.path.join(scenario_dir, f"realized_year.csv")
    df.to_csv(path, float_format=float_format)
    print(f"Wrote {n}")
    