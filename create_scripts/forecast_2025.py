""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from data_scripts.data_loader import HistoricalData
from data_scripts.data_generator_v3 import DataForecaster
from model_scripts.environment import EmissionFactorEstimator
from common_scripts.utils import cache_read

from time import time
from copy import deepcopy

TRAIN_YEARS = 2
FORECAST_HORIZON = 96
SIMULATION_HORIZON = 13*31*24
LONGTERM_SIMULATION_EVERY_N_DAYS = 7

#%% Data retrieval - There is only data from 2015 and forward for Portugal
t_s = time()
start   = pd.Timestamp('20150101', tz='UTC')
end     = pd.Timestamp('20251231', tz='UTC')
data_object_ = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')
t_e = time()
print(f"Data retrieval and preprocessing took {t_e-t_s:.2f} seconds.")

training_horizon = TRAIN_YEARS # years
latest_observation = pd.Timestamp('20241231', tz='UTC')
training_start = latest_observation - pd.DateOffset(years=training_horizon)
data_object = deepcopy(data_object_)
data_object.data = data_object.data.loc[(data_object.data.index >= training_start) & (data_object.data.index <= latest_observation)]

t_s = time()
forecaster = DataForecaster(database=data_object, r_load_tag="", other_exog_tags = ["gas_with_ets"], stochastic_price_model="GARCH", weather_years=False, verbose=False)
forecaster.build_simulation_models()
t_e = time()
print(f"Data forecaster and models built in {t_e-t_s:.2f} seconds.")

# Calculates "Carbon intensity gCO₂eq/kWh (direct)" as a linear function of price [€/MWh], system wind [MW], and system solar [MW].
cache_path_mappers = os.getcwd() + "/models/plant_models/"
emissions_mapper = cache_read(cache_path_mappers + "emission_factor.pkl")
emissions_model = EmissionFactorEstimator(emissions_mapper)

X = df[["price", "solar", "wind", "Actual Load"]]
df["emissions"] = emissions_model(X)

hours_2025 = data_object_.data.loc[data_object_.data.index.year == 2025].index
days_2025 = pd.to_datetime(pd.date_range(start=hours_2025[0], end=hours_2025[-1], freq='D'))

for day in days_2025:
    day_str = day.strftime("%Y-%m-%d")

    last_known_ts = data_object.data.index[-1]
    print("Current time: ", last_known_ts)

    start = last_known_ts + pd.Timedelta(hours=1)
    end = last_known_ts + pd.Timedelta(hours=FORECAST_HORIZON)
    hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'))

    # Own Model:
    fc_24h = forecaster.forecast(start=start, end=end, n_forecasts=1)[0]
    df_forecast_fc = fc_24h.astype(float)
    year_month_index = hourly_index.tz_localize(None).to_period('M')
    solar_capacities = year_month_index.map(forecaster.database.caps['solar'])
    wind_capacities = year_month_index.map(forecaster.database.caps['wind'])
    