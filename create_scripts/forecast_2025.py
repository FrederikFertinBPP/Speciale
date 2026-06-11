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

from time import time
from copy import deepcopy

OUTPUT_DIR = "scenario_data/Historicals"
TRAIN_YEARS = np.linspace(4.5,10,12)[6:]
train_hours = TRAIN_YEARS * 8760 - 8760
DAY_AHEAD_HOURS = 24
FORECAST_HORIZON = 96
SIMULATION_HORIZON = 8760#13*31*24
LONGTERM_SIMULATION_EVERY_N_DAYS = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)
date_format="%Y-%m-%d %H"
float_format='%.3f'

LAST_INSAMPLE_TIME = pd.Timestamp('20241231 230000', tz='UTC')
forecaster_kwargs = {"other_exog_tags": ["gas_with_ets"], "stochastic_price_model": "GARCH", "weather_years":False}

def train_forecaster(dataloader, old_forecaster=None, **kwargs):
    forecaster = DataForecaster(database=dataloader, **kwargs, verbose=True)
    try:
        forecaster.build_simulation_models(old_forecaster)
    except Exception as e:
        print("Error building forecaster models:", e)
        print("Falling back to training from scratch..")
        forecaster = DataForecaster(database=dataloader, **kwargs, verbose=True)
        forecaster.build_simulation_models()
    return forecaster

def get_data_loader(full_data_loader, current_time, train_ix=0):
    training_horizon = train_hours[train_ix] # Number of hours to include in the training set, on top of one year worth of data that is always included.
    latest_observation = current_time
    training_start = latest_observation - pd.DateOffset(years=1) + pd.Timedelta(1,'h') - pd.Timedelta(training_horizon,'h')
    data_object = deepcopy(full_data_loader)
    data_object.data = data_object.data.loc[(data_object.data.index >= training_start) & (data_object.data.index <= latest_observation)]
    return data_object

#%% Data retrieval - There is only data from 2015 and forward for Portugal
t_s = time()
start   = pd.Timestamp('20150101', tz='UTC')
end     = pd.Timestamp('20251231', tz='UTC')
data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')
t_e = time()
print(f"Data retrieval and preprocessing took {t_e-t_s:.2f} seconds.")

for train_ix in range(len(train_hours)):
    print(f"\n\n{'='*10} TRAINING FORECASTER WITH {TRAIN_YEARS[train_ix]:.2f} YEARS OF TRAINING DATA {'='*10}\n")

    dataloader = get_data_loader(data_object, LAST_INSAMPLE_TIME, train_ix=train_ix)
    forecaster = train_forecaster(dataloader, **forecaster_kwargs)

    perfect_forecasting = False
    perfect_forecasting_horizon = 24

    targets = ["price", "solar", "wind"]
    models = [f"SOTA{str(round(float(TRAIN_YEARS[train_ix]),2)).replace(".","_")}year" + (f"perfect{perfect_forecasting_horizon}hours" if perfect_forecasting else "")]

    # The online period is walked day-by-day
    df_online = data_object.data.loc[data_object.data.index.year == 2025]
    hours_2025 = df_online.index
    online_days = df_online.resample("D").first().index
    n_days = len(online_days)
    print(f"Starting online learning loop over {n_days} days...\n")

    os.makedirs(OUTPUT_DIR + f"/{models[0]}", exist_ok=True)

    for day_idx, current_day in enumerate(online_days):
        day_str = current_day.strftime("%Y-%m-%d")

        # ── 1. Forecast next 24 hours ─────────────────────────────────────────
        last_known_ts = dataloader.data.index[-1]
        print("Current time: ", last_known_ts)

        start = last_known_ts + pd.Timedelta(hours=1)
        end = last_known_ts + pd.Timedelta(hours=FORECAST_HORIZON)
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'))

        # Own Model:
        fc_24h = forecaster.forecast(start=start,end=end,n_forecasts=1)[0]
        df_forecast = fc_24h.astype(float)
        if perfect_forecasting:
            l = min(sum(df_online.index > last_known_ts), perfect_forecasting_horizon) # Get up to 'perfect_forecasting_horizon' hours of perfect information.
            hourly_index = df_forecast.index[:l]
            perfect_foresight = df_online.loc[hourly_index, df_forecast.columns]
            ym = hourly_index.tz_localize(None).to_period('M')
            solar_caps = ym.map(forecaster.database.caps[forecaster.solar_tag])
            wind_caps  = ym.map(forecaster.database.caps[forecaster.wind_tag])
            real_values = pd.DataFrame(index=hourly_index,
                            data={  forecaster.solar_tag: perfect_foresight[forecaster.solar_tag] / solar_caps.values,
                                    forecaster.wind_tag: perfect_foresight[forecaster.wind_tag] / wind_caps.values,
                                    forecaster.price_tag: perfect_foresight[forecaster.price_tag]})
            df_forecast.loc[hourly_index] = real_values[df_forecast.columns].values

        # ── 1. Simulate next 13 months ────────────────────────────────────────
        if day_idx % LONGTERM_SIMULATION_EVERY_N_DAYS == 0:
            # Own Model:
            end = last_known_ts + pd.Timedelta(hours=SIMULATION_HORIZON)
            fc_year = forecaster.simulate_period(start=start,end=end)[0]
            df_simulation = fc_year.astype(float)

        # ── Save forecasts and simulations, then repeat ───────────────────────
        for m in models:
            df_forecast.to_csv(OUTPUT_DIR + f"/{m}/forecast_" + day_str + ".csv", date_format=date_format, float_format=float_format)
            if day_idx % LONGTERM_SIMULATION_EVERY_N_DAYS == 0:
                df_simulation.to_csv(OUTPUT_DIR + f"/{m}/long-term-sim_" + day_str + ".csv", date_format=date_format, float_format=float_format)
        print("  Forecasts and simulations saved for this day.\n")

        # ── 2. Observe actuals for this day ───────────────────────────────────
        dataloader = get_data_loader(data_object, last_known_ts + pd.Timedelta(24,'h'), train_ix=train_ix)

        # ── 3. Retrain models on new history ──────────────────────────────────
        forecaster = train_forecaster(dataloader, old_forecaster = forecaster, **forecaster_kwargs)