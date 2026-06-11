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
from common_scripts.utils import set_plotting_style

from time import time
from copy import deepcopy
from scipy.stats import wasserstein_distance

OUTPUT_DIR = "scenario_data/Historicals/_cross_validation"
TRAIN_YEARS = np.asarray([2.,3.,4.,5.])
train_hours = TRAIN_YEARS * 8760 - 8760
DAY_AHEAD_HOURS = 24
FORECAST_HORIZON = 96
SIMULATION_HORIZON = 8760#13*31*24
LONGTERM_SIMULATION_EVERY_N_DAYS = 7
targets = ["price", "solar", "wind"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
date_format="%Y-%m-%d %H"
float_format='%.3f'

LAST_INSAMPLE_TIME = pd.Timestamp('20191231 230000', tz='UTC')
forecaster_kwargs = {"other_exog_tags": ["gas_with_ets"], "stochastic_price_model": "GARCH", "weather_years":False}

def train_forecaster(dataloader, old_forecaster=None, **kwargs):
    forecaster = DataForecaster(database=dataloader, **kwargs, verbose=False)
    try:
        forecaster.build_simulation_models(old_forecaster)
    except Exception as e:
        print("Error building forecaster models:", e)
        print("Falling back to training from scratch..")
        forecaster = DataForecaster(database=dataloader, **kwargs, verbose=False)
        forecaster.build_simulation_models()
    return forecaster

def get_data_loader(full_data_loader, current_time, train_ix=0):
    training_horizon = train_hours[train_ix] # Number of hours to include in the training set, on top of one year worth of data that is always included.
    latest_observation = current_time
    training_start = latest_observation - pd.DateOffset(years=1) + pd.Timedelta(1,'h') - pd.Timedelta(training_horizon,'h')
    data_obj = deepcopy(full_data_loader)
    data_obj.data = data_obj.data.loc[(data_obj.data.index >= training_start) & (data_obj.data.index <= latest_observation)]
    return data_obj

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

    model = f"SOTA{str(round(float(TRAIN_YEARS[train_ix]),2)).replace(".","_")}year"

    # The online period is walked day-by-day
    df_online = data_object.data.loc[(data_object.data.index.year >= 2020) & (data_object.data.index.year <= 2024)]
    hours_2025 = df_online.index
    online_days = df_online.resample("D").first().index
    n_days = len(online_days)
    print(f"Starting online learning loop over {n_days} days...\n")

    os.makedirs(OUTPUT_DIR + f"/{model}", exist_ok=True)

    wasserstein_mae_results = []
    wasserstein_dates = []
    forecast_rmse_results = []
    forecast_rmse_average = []
    rmse_dates = []

    for day_idx, current_day in enumerate(online_days):
        # day_str = current_day.strftime("%Y-%m-%d")
        # ── 1. Forecast next 24 hours ─────────────────────────────────────────
        last_known_ts = dataloader.data.index[-1]
        print("Current time: ", last_known_ts)
        hours_left = sum(df_online.index > last_known_ts)

        start = last_known_ts + pd.Timedelta(hours=1)
        
        if hours_left >= FORECAST_HORIZON:
            end = last_known_ts + pd.Timedelta(hours=FORECAST_HORIZON)
            hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'))

            # Own Model:
            fc_24h = forecaster.forecast(start=start,end=end,n_forecasts=1)[0]
            df_forecast = fc_24h.astype(float)
            df_true = df_online.loc[hourly_index, df_forecast.columns]
            ym = hourly_index.tz_localize(None).to_period('M')
            solar_caps = ym.map(forecaster.database.caps[forecaster.solar_tag])
            wind_caps  = ym.map(forecaster.database.caps[forecaster.wind_tag])
            real_values = pd.DataFrame(index=hourly_index,
                            data={  forecaster.solar_tag: df_true[forecaster.solar_tag] / solar_caps.values,
                                    forecaster.wind_tag: df_true[forecaster.wind_tag] / wind_caps.values,
                                    forecaster.price_tag: df_true[forecaster.price_tag]})
            # ── Save forecasts RMSE for each day of the 4 days ahead ───────────────────────
            # We only calculate RMSE for the hours where there is solar production (i.e. daytime hours)
            # to avoid skewing the results with a large number of zero solar production hours.
            rmse = {col: [np.sqrt(
                            (24/np.sum(df_online.loc[hourly_index[24*(ix):24*(ix+1)], "is_day"]) if col == "solar" else 1) * 
                            np.mean(
                                    (df_forecast[col].iloc[24*(ix):24*(ix+1)] - real_values[col].iloc[24*(ix):24*(ix+1)])**2
                            )) for ix in range(int(FORECAST_HORIZON/24))]
                        for col in df_forecast.columns}
            rmse_average = {col: np.sqrt(
                            (96/np.sum(df_online.loc[hourly_index, "is_day"]) if col == "solar" else 1) * 
                            np.mean(
                                    (df_forecast[col] - real_values[col])**2
                            ))
                        for col in df_forecast.columns}
            forecast_rmse_results.append(rmse)
            forecast_rmse_average.append(rmse_average)
            rmse_dates.append(current_day)
            if hours_left >= SIMULATION_HORIZON:
                if current_day.is_month_start:
                    end = last_known_ts + pd.Timedelta(hours=SIMULATION_HORIZON)
                    hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'))
                    df_true = df_online.loc[hourly_index, df_forecast.columns]
                    ym = hourly_index.tz_localize(None).to_period('M')
                    solar_caps = ym.map(forecaster.database.caps[forecaster.solar_tag])
                    wind_caps  = ym.map(forecaster.database.caps[forecaster.wind_tag])
                    real_values = pd.DataFrame(index=hourly_index,
                                    data={  forecaster.solar_tag: df_true[forecaster.solar_tag] / solar_caps.values,
                                            forecaster.wind_tag: df_true[forecaster.wind_tag] / wind_caps.values,
                                            forecaster.price_tag: df_true[forecaster.price_tag]})
                    fc_year = forecaster.simulate_period(start=start,end=end, n_sims=3)
                    df_simulations = [fc_year[ix].astype(float) for ix in range(len(fc_year))]
                    wasserstein_mae = {col: np.mean([wasserstein_distance(df_sim[col].values, real_values[col].values) for df_sim in df_simulations]) for col in df_forecast.columns}
                    wasserstein_mae_results.append(wasserstein_mae)
                    wasserstein_dates.append(current_day)

        # ── 2. Observe actuals for this day ───────────────────────────────────
        dataloader = get_data_loader(data_object, last_known_ts + pd.Timedelta(24,'h'), train_ix=train_ix)

        # ── 3. Retrain models on new history ──────────────────────────────────
        forecaster = train_forecaster(dataloader, old_forecaster = forecaster, **forecaster_kwargs)
    df_forecast_rmse = pd.DataFrame(forecast_rmse_average, index=rmse_dates)
    # Create multi-index:
    df_forecast_rmse_wind = pd.DataFrame([res["wind"] for res in forecast_rmse_results], index=rmse_dates)
    df_forecast_rmse_solar = pd.DataFrame([res["solar"] for res in forecast_rmse_results], index=rmse_dates)
    df_forecast_rmse_price = pd.DataFrame([res["price"] for res in forecast_rmse_results], index=rmse_dates)
    df_wasserstein_mae = pd.DataFrame(wasserstein_mae_results, index=wasserstein_dates)
    df_forecast_rmse.to_csv(OUTPUT_DIR + f"/{model}/forecast_rmse.csv", index=True, float_format=float_format)
    df_wasserstein_mae.to_csv(OUTPUT_DIR + f"/{model}/wasserstein_mae.csv", index=True, float_format=float_format)
    df_forecast_rmse_wind.to_csv(OUTPUT_DIR + f"/{model}/wind_rmse_days.csv", index=True, float_format=float_format)
    df_forecast_rmse_solar.to_csv(OUTPUT_DIR + f"/{model}/solar_rmse_days.csv", index=True, float_format=float_format)
    df_forecast_rmse_price.to_csv(OUTPUT_DIR + f"/{model}/price_rmse_days.csv", index=True, float_format=float_format)


#%% Investigate CV
import matplotlib.pyplot as plt
years = range(1,6)
filenames = ("wasserstein_mae", "forecast_rmse", "wind_rmse_days", "solar_rmse_days", "price_rmse_days")
errors = {errortype : [pd.read_csv(OUTPUT_DIR + f"/SOTA{ix}_0year/{errortype}.csv", index_col=0) for ix in years] for errortype in filenames}

for f in filenames:
    df = errors[f]
    print("\n",f)
    for year in years:
        print("\n Years: ", year)
        df_x = df[year-1]
        print(df_x.median())

for jx in range(2,5):
    f = filenames[jx]
    df_list = errors[f]
    fig, ax = plt.subplots(figsize=(12,8))
    plt.title(f)
    for ix, df in enumerate(df_list):
        df_ = df.copy()
        df_.columns = list(str(int(col)+1) for col in df_.columns)
        plt.plot(df_.mean() * (1 if jx==4 else 100), label=f"{ix+1} years",marker="o")
    plt.legend(title="Training Window")
    plt.ylabel("€/MWh" if jx == 4 else "%")
    plt.xlabel("Days-ahead")
    plt.tight_layout()
    plt.savefig(f"documentation/cv_forecaster/{f}.png")
    plt.close()

for jx in range(2):
    f = filenames[jx]
    df_list = errors[f]
    fig, axs = plt.subplots(3,1,figsize=(12,8))
    axs = axs.flatten()
    plt.title(f)
    for ix, df in enumerate(df_list):
        df_ = df.copy()
        #df_.columns = list(str(int(col)+1) for col in df_.columns)
        for a in range(3):
            ax = axs[a]
            col = df.columns[a]
            ax.plot(df_[col] * (1 if col=="price" else 100), label=f"{ix+1} years")
    plt.legend(title="Training Window")
    plt.ylabel("€/MWh" if jx == 4 else "%")
    plt.xlabel("Days-ahead")
    plt.tight_layout()
    plt.savefig(f"documentation/cv_forecaster/{f}.png")
    plt.close()

# Persistence forecasts:
hi = df_online.index
ym = hi.tz_localize(None).to_period('M')
solar_caps = ym.map(forecaster.database.caps[forecaster.solar_tag])
wind_caps  = ym.map(forecaster.database.caps[forecaster.wind_tag])
real_values = pd.DataFrame(index=hi,
                data={  forecaster.solar_tag: df_online[forecaster.solar_tag] / solar_caps.values,
                        forecaster.wind_tag: df_online[forecaster.wind_tag] / wind_caps.values,
                        forecaster.price_tag: df_online[forecaster.price_tag]})
hi = data_object.data.index
ym = hi.tz_localize(None).to_period('M')
solar_caps = ym.map(forecaster.database.caps[forecaster.solar_tag])
wind_caps  = ym.map(forecaster.database.caps[forecaster.wind_tag])
all_real_values = pd.DataFrame(index=hi,
                data={  forecaster.solar_tag: data_object.data[forecaster.solar_tag] / solar_caps.values,
                        forecaster.wind_tag: data_object.data[forecaster.wind_tag] / wind_caps.values,
                        forecaster.price_tag: data_object.data[forecaster.price_tag]})

target = "solar"
y_true = real_values[target].values
y_pred = all_real_values.loc[df_online.index - pd.Timedelta(24,'h'),target].values
e = list(y_true-y_pred)

rmse= np.sqrt(np.mean(np.square(e)) * len(y_true) / sum(y_true>0.001))
print(rmse)
for i in range(5):
    print(round(np.sqrt(np.mean(errors["solar_rmse_days"][i]["0"]**2))*100,2))