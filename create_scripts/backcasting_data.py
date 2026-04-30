import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import prophet
# import pandas as pd

# filepath = "historical_data/clean_dataframes/server-ENTSOEcountry-PT2015-01-01to2024-12-31.csv"
# data = pd.read_csv(filepath, index_col=0, parse_dates=True)
# data = data.rename(columns={"price": "y"})

"""
Prophet Online Learning Forecaster
===================================
- Batch trains on first 2 years of hourly data (price, solar, wind)
- Rolls forward day-by-day: forecasts 24h ahead, then ingests actuals
- Every 7 days: produces a year-ahead (8760h) forecast
- Outputs are saved incrementally to CSV files in ./forecast_outputs/
"""

# import os
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
from data_scripts import DataForecaster, HistoricalData
from common_scripts.utils import cache_read
from model_scripts.environment import EmissionFactorEstimator

warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = "historical_data/clean_dataframes/server-ENTSOEcountry-PT2015-01-01to2024-12-31.csv"
DATA_CAPS_PATH = "historical_data/wind_solar_capacity_PT.csv"
OUTPUT_DIR = "scenario_data/Historicals"
INITIAL_TRAIN_YEARS = 2
DAY_AHEAD_HOURS = 24
FORECAST_HORIZON = 96
SIMULATION_HORIZON = 13*31*24
LONGTERM_SIMULATION_EVERY_N_DAYS = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)
date_format="%Y-%m-%d %H"
float_format='%.3f'

# ─────────────────────────────────────────────
# PROPHET MODEL FACTORY
# ─────────────────────────────────────────────
def make_prophet(target: str) -> Prophet:
    """
    Returns a Prophet model tuned for each target variable.
    All three series share common electricity-market seasonality patterns.
    """
    common_kwargs = dict(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        uncertainty_samples=0,       # faster; set to 1000 for prediction intervals
    )
    if target == "price":
        # Prices can be spiky; slightly more flexible trend
        common_kwargs["changepoint_prior_scale"] = 0.10
    elif target in ("solar", "wind"):
        # Renewables have strong additive floor at 0
        common_kwargs["seasonality_mode"] = "additive"

    return Prophet(**common_kwargs)


def fit_prophet(df_train: pd.DataFrame, target: str, initial_model: Prophet = None) -> Prophet:
    """Fit a fresh Prophet model on df_train for the given target."""
    prophet_df = df_train[["ds", target]].rename(columns={target: "y"})
    if target == "solar":
        prophet_df["y"] = prophet_df["y"].clip(lower=0)

    m = make_prophet(target)
    if initial_model is not None:
        m.fit(prophet_df, init=initial_model.params)
    else:
        m.fit(prophet_df)
    return m


def forecast_next_hours(model: Prophet, last_ds: pd.Timestamp, n_hours: int) -> pd.DataFrame:
    """Generate an n_hours forecast starting 1 hour after last_ds."""
    future_dates = pd.date_range(
        start=last_ds + pd.Timedelta(hours=1),
        periods=n_hours,
        freq="h",
    )
    future_df = pd.DataFrame({"ds": future_dates})
    fc = model.predict(future_df)
    return fc[["ds", "yhat"]].copy()


# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────

def train_forecaster(df, old_forecaster=None):
    dataloader = HistoricalData(start=df.index[0], end=df.index[-1], load_data=False)
    dataloader.data = df.tz_localize("UTC") # Set the data for the forecaster
    dataloader.data = dataloader._create_seasonal_features(df=dataloader.data[['price', 'wind', 'solar']], prod_columns=['wind', 'solar'])
    dataloader.load_capacity_data()
    forecaster = DataForecaster(database=dataloader, verbose=False)
    try:
        forecaster.build_simulation_models(old_forecaster)
    except Exception as e:
        print("Error building forecaster models:", e)
        print("Falling back to training from scratch..")
        forecaster = DataForecaster(database=dataloader, verbose=True)
        forecaster.build_simulation_models()
    return forecaster

def fill_missing_hours(df):
    missing_hours = sorted(set(df.loc[df.isna().any(axis=1)].index))
    for hour in missing_hours:
        copied_hour = hour - pd.Timedelta(1, 'day')
        row = df.loc[df.index == copied_hour]
        row.index = [hour]
        df = pd.concat([df.loc[df.index < hour], row, df.loc[df.index > hour]])
    return df

print("Loading data...")
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df.index.name = "datetime"
df = df.sort_index()
df = fill_missing_hours(df) # Ensure continuous hourly index
# Drop timezone info (Prophet doesn't handle tz-aware datetimes well; we assume data is in UTC)
df.index = df.index.tz_localize(None)

forecaster = train_forecaster(df)

#Create backcasting time series for environment.
cache_path_mappers = os.getcwd() + "/models/plant_models/"
emissions_mapper = cache_read(cache_path_mappers + "emission_factor.pkl")
emissions_model = EmissionFactorEstimator(emissions_mapper)
ems = emissions_model(df[["price","wind","solar"]])
hourly_index = df.index
year_month_index = hourly_index.tz_localize(None).to_period('M')
solar_capacities = year_month_index.map(forecaster.database.caps['solar'])
wind_capacities = year_month_index.map(forecaster.database.caps['wind'])
df_backcasting = df.copy()
df_backcasting["wind"] /= wind_capacities
df_backcasting["solar"] /= solar_capacities
df_backcasting["emissions"] = ems
df_backcasting.to_csv("historical_data/clean_dataframes/backcasting_timeseries.csv", index=True)

# Keep only the three columns of interest
df = df[["price", "solar", "wind"]].dropna(how="all")

# Resample to ensure a clean hourly grid (forward-fill tiny gaps)
df = df.resample("h").mean().ffill(limit=3)

# Add a 'ds' column Prophet needs
df["ds"] = df.index

print(f"  Total records : {len(df):,}")
print(f"  Date range    : {df.index[0]}  →  {df.index[-1]}")

# ─────────────────────────────────────────────
# SPLIT: INITIAL TRAINING vs ONLINE PERIOD
# ─────────────────────────────────────────────
cutoff_date = df.index[0] + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
cutoff_date = pd.Timestamp(2022, 7, 24)
df_init = df[df.index < cutoff_date].copy()
df_init = df_init.iloc[-(365*24)*2-24:] # Keep last 2 years + 1 day for initial training
print(df_init.head())
print(df_init.index[-1])
df_online = df[df.index >= cutoff_date].copy()

print(f"\nInitial training period : {df_init.index[0]}  →  {df_init.index[-1]}  ({len(df_init):,} rows)")
print(f"Online learning period  : {df_online.index[0]}  →  {df_online.index[-1]}  ({len(df_online):,} rows)")

# ─────────────────────────────────────────────
# INITIAL BATCH TRAINING
# ─────────────────────────────────────────────
targets = ["price", "solar", "wind"]
models = ["persistence", "prophet", "forecaster"]

print("\nBatch training initial prophets...")
prophets = {}
for t in targets:
    print(f"  Fitting {t}...")
    prophets[t] = fit_prophet(df_init, t)
print(" Prophets Done.\n")

forecaster = train_forecaster(df_init)

if False: #documentation
    fc = forecaster.simulate_year_ahead(start=df_init.index[-1] + pd.Timedelta(hours=1), n_sims=1)[0]
    last_known_ts = df_init.index[-1]
    pr_wind = forecast_next_hours(prophets['wind'], last_known_ts, YEAR_AHEAD_HOURS)
    import matplotlib.pyplot as plt
    plt.plot(pr_wind['yhat'], alpha=0.2, label="Prophet ts")
    plt.plot(fc['wind'].values*4553, alpha=0.2, label="FC ts")
    plt.plot(pd.DataFrame(fc['wind'].values*4553).rolling(window=168*3).mean(), label="FC MA")
    plt.plot(pd.DataFrame(pr_wind['yhat'].values).rolling(window=168*3).mean(), label="Prophet MA")
    plt.plot(np.sort(fc['wind'].values*4553), label="Prophet DC")
    plt.plot(np.sort(pr_wind['yhat'].values), label="Prophet DC")
    xx = df_init['wind'].iloc[-8760:].values
    plt.plot(xx, alpha=0.2, label="2016 ts")
    plt.plot(pd.DataFrame(xx).rolling(window=3*168).mean(), label="2016 MA")
    plt.plot(np.sort(xx), label="2016 DC")
    plt.legend()
    plt.savefig("wind_forecasting_comparison.png")
    plt.close()
# forecaster.simulate_year_ahead(start=df_init.index[-1] + pd.Timedelta(hours=1))
# ─────────────────────────────────────────────
# ONLINE LEARNING LOOP
# ─────────────────────────────────────────────

# We accumulate the training history as we go
df_history = df_init.copy()

# Storage for results
cf_lists = ["wind_cf", "solar_cf"]
daily_records = {m: {t: [] for t in targets + cf_lists} for m in models}
yearly_records = {m: {t: [] for t in targets + cf_lists} for m in models}

# The online period is walked day-by-day
online_days = df_online.resample("D").first().index  # unique days in online period
n_days = len(online_days)

print(f"Starting online learning loop over {n_days} days...\n")

for day_idx, current_day in enumerate(online_days):
    day_str = current_day.strftime("%Y-%m-%d")

    # ── 1. Forecast next 24 hours ─────────────────────────────────────────
    last_known_ts = df_history.index[-1]
    print("Time: ", last_known_ts)

    start=df_history.index[-1] + pd.Timedelta(hours=1)
    end=df_history.index[-1] + pd.Timedelta(hours=FORECAST_HORIZON)
    hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'))

    # Own Model:
    fc_24h = forecaster.forecast(start=start,end=end,n_forecasts=1)[0]
    df_forecast_fc = fc_24h.astype(float)
    year_month_index = hourly_index.tz_localize(None).to_period('M')
    solar_capacities = year_month_index.map(forecaster.database.caps['solar'])
    wind_capacities = year_month_index.map(forecaster.database.caps['wind'])
    for t in targets:
        fc_24h_t = fc_24h[[t]].copy()
        fc_24h_t.columns = ['yhat']
        fc_24h_t['ds'] = pd.to_datetime(fc_24h_t.index).tz_localize(None)
        fc_24h_t = fc_24h_t[['ds', 'yhat']]
        fc_24h_t["target"] = t
        fc_24h_t["forecast_origin"] = last_known_ts
        fc_24h_t["horizon"] = "day_ahead"
        fc_24h_t.index = range(len(fc_24h_t))
        if t != 'price':
            daily_records["forecaster"][t + "_cf"].append(fc_24h_t)
            fc_24h_t['yhat'] = fc_24h_t['yhat'] * (solar_capacities.values if t == 'solar' else wind_capacities.values)
        daily_records["forecaster"][t].append(fc_24h_t)

    # Prophet and Persistence models:
    df_forecast_pr = df_forecast_fc.copy()
    df_forecast_pe = df_forecast_fc.copy()
    for t in targets:
        pr_24h = forecast_next_hours(prophets[t], last_known_ts, FORECAST_HORIZON)
        pr_24h["target"] = t
        if t != 'price':
            pr_24h['yhat'] = np.clip(pr_24h['yhat'],a_min=0,a_max=np.inf)
        pr_24h["forecast_origin"] = last_known_ts
        pr_24h["horizon"] = "day_ahead"
        daily_records["prophet"][t].append(pr_24h)
        persistence_24h = pr_24h.copy()
        persistence_24h['yhat'] = list(df_history[t].iloc[-DAY_AHEAD_HOURS:].values) * (FORECAST_HORIZON // DAY_AHEAD_HOURS)
        daily_records["persistence"][t].append(persistence_24h)
        if t != 'price':
            pr_24h['yhat'] /= (solar_capacities.values if t == 'solar' else wind_capacities.values)
            persistence_24h['yhat'] /= (solar_capacities.values if t == 'solar' else wind_capacities.values)
            daily_records["prophet"][t + "_cf"].append(pr_24h)
            daily_records["persistence"][t + "_cf"].append(persistence_24h)
        df_forecast_pr[t] = pr_24h['yhat'].values
        df_forecast_pe[t] = persistence_24h['yhat'].values

    # ── 1. Simulate next 13 months ────────────────────────────────────────
    if day_idx % LONGTERM_SIMULATION_EVERY_N_DAYS == 0:
        # Own Model:
        end=df_history.index[-1] + pd.Timedelta(hours=SIMULATION_HORIZON)
        fc_year = forecaster.simulate_period(start=start,end=end)[0]
        df_simulation_fc = fc_year

        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'))
        year_month_index = hourly_index.tz_localize(None).to_period('M')
        solar_capacities = year_month_index.map(forecaster.database.caps['solar'])
        wind_capacities = year_month_index.map(forecaster.database.caps['wind'])
        for t in targets:
            fc_year_t = fc_year[[t]].copy()
            fc_year_t.columns = ['yhat']
            fc_year_t['ds'] = pd.to_datetime(fc_year_t.index).tz_localize(None)
            fc_year_t = fc_year_t[['ds', 'yhat']]
            fc_year_t["target"] = t
            fc_year_t["forecast_origin"] = last_known_ts
            fc_year_t["horizon"] = "day_ahead"
            fc_year_t.index = range(len(fc_year_t))
            if t != 'price':
                yearly_records["forecaster"][t + "_cf"].append(fc_year_t)
                fc_year_t['yhat'] = fc_year_t['yhat'] * (solar_capacities.values if t == 'solar' else wind_capacities.values)
            yearly_records["forecaster"][t].append(fc_year_t)

        # Prophet and Persistence models:
        df_simulation_pr = df_simulation_fc.copy()
        df_simulation_pe = df_simulation_fc.copy()
        for t in targets:
            pr_year = forecast_next_hours(prophets[t], last_known_ts, SIMULATION_HORIZON)
            pr_year["target"] = t
            if t != 'price':
                pr_year['yhat'] = np.clip(pr_year['yhat'],a_min=0,a_max=np.inf)
            pr_year["forecast_origin"] = last_known_ts
            pr_year["horizon"] = "year_ahead"
            yearly_records["prophet"][t].append(pr_year)
            persistence_year = pr_year.copy()
            persistence_year['yhat'] = list(df_history[t].iloc[-8760:].values) + list(df_history[t].iloc[-(SIMULATION_HORIZON-8760):].values)
            yearly_records["persistence"][t].append(persistence_year)
            if t != 'price':
                pr_year['yhat'] /= (solar_capacities.values if t == 'solar' else wind_capacities.values)
                persistence_year['yhat'] /= (solar_capacities.values if t == 'solar' else wind_capacities.values)
                yearly_records["prophet"][t + "_cf"].append(pr_year)
                yearly_records["persistence"][t + "_cf"].append(persistence_year)
            df_simulation_pr[t] = pr_year['yhat'].values
            df_simulation_pe[t] = persistence_year['yhat'].values

    # ── 2. Observe actuals for this day ───────────────────────────────────
    day_actuals = df_online[df_online.index.date == current_day.date()]
    
    # Append actuals to rolling history
    df_history = pd.concat([df_history, day_actuals])
    df_history = df_history.iloc[DAY_AHEAD_HOURS:]

    # ── 3. Retrain models on new history ──────────────────────────────────
    for t in targets:
        previous_model = prophets[t]
        prophets[t] = fit_prophet(df_history, t, initial_model=previous_model)

    forecaster = train_forecaster(df_history, old_forecaster = forecaster)

    # ── Save forecasts and simulations, then repeat ───────────────────────
    df_forecast_fc.to_csv(OUTPUT_DIR + "/forecaster/forecast_" + day_str + ".csv", date_format=date_format, float_format=float_format)
    df_forecast_pr.to_csv(OUTPUT_DIR + "/prophet/forecast_" + day_str + ".csv", date_format=date_format, float_format=float_format)
    df_forecast_pe.to_csv(OUTPUT_DIR + "/persistence/forecast_" + day_str + ".csv", date_format=date_format, float_format=float_format)
    if day_idx % LONGTERM_SIMULATION_EVERY_N_DAYS == 0:
        df_simulation_fc.to_csv(OUTPUT_DIR + "/forecaster/long-term-sim_" + day_str + ".csv", date_format=date_format, float_format=float_format)
        df_simulation_pr.to_csv(OUTPUT_DIR + "/prophet/long-term-sim_" + day_str + ".csv", date_format=date_format, float_format=float_format)
        df_simulation_pe.to_csv(OUTPUT_DIR + "/persistence/long-term-sim_" + day_str + ".csv", date_format=date_format, float_format=float_format)
    print("  Forecasts and simulations saved for this day.\n")
    # Override to save memory
    daily_records = {m: {t: [] for t in targets + cf_lists} for m in models}
    yearly_records = {m: {t: [] for t in targets + cf_lists} for m in models}


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
# print("\nSaving forecast outputs...")

# for t in targets:
#     # ── Day-ahead forecasts ───────────────────────────────────────────────
#     if daily_records[t]:
#         daily_df = pd.concat(daily_records_prophet[t], ignore_index=True)
#         daily_df.columns = ["ds", "yhat", "target", "forecast_origin", "horizon"]
#         path = os.path.join(OUTPUT_DIR, f"day_ahead_{t}.csv")
#         daily_df.to_csv(path, index=False)
#         print(f"  Saved {path}  ({len(daily_df):,} rows)")
    
#     if daily_records_forecaster[t]:
#         daily_df = pd.concat(daily_records_forecaster[t], ignore_index=True)
#         daily_df.columns = ["ds", "yhat", "target", "forecast_origin", "horizon"]
#         path = os.path.join(OUTPUT_DIR, f"day_ahead_forecaster_{t}.csv")
#         daily_df.to_csv(path, index=False)
#         print(f"  Saved {path}  ({len(daily_df):,} rows)")

#     if daily_records_persistence[t]:
#         daily_df = pd.concat(daily_records_persistence[t], ignore_index=True)
#         daily_df.columns = ["ds", "yhat", "target", "forecast_origin", "horizon"]
#         path = os.path.join(OUTPUT_DIR, f"day_ahead_persistence_{t}.csv")
#         daily_df.to_csv(path, index=False)
#         print(f"  Saved {path}  ({len(daily_df):,} rows)")

#     # ── Year-ahead forecasts ──────────────────────────────────────────────
#     if yearly_records[t]:
#         yearly_df = pd.concat(yearly_records_prophet[t], ignore_index=True)
#         yearly_df.columns = ["ds", "yhat", "target", "forecast_origin", "horizon"]
#         path = os.path.join(OUTPUT_DIR, f"year_ahead_{t}.csv")
#         yearly_df.to_csv(path, index=False)
#         print(f"  Saved {path}  ({len(yearly_df):,} rows)")

# print("\nDone! All forecasts written to:", OUTPUT_DIR)


# ─────────────────────────────────────────────
# OPTIONAL: QUICK ACCURACY SUMMARY
# ─────────────────────────────────────────────
def mae(a, b):
    return np.mean(np.abs(a - b))

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

print("\n── Day-ahead forecast accuracy (MAE / RMSE vs actuals) ──")
for t in targets:
    if not daily_records[t]:
        continue
    fc_df = pd.concat(daily_records[t], ignore_index=True)
    fc_df.columns = ["ds", "yhat", "target", "forecast_origin", "horizon"]
    fc_df["ds"] = pd.to_datetime(fc_df["ds"])

    # Merge with actuals
    actuals = df_online[["ds", t]].copy()
    actuals["ds"] = pd.to_datetime(actuals["ds"])
    merged = fc_df.merge(actuals, on="ds", how="inner")

    if len(merged) == 0:
        continue
    
    print("  Prophet accuracy: ")
    print(f"  {t:6s}  MAE={mae(merged['yhat'], merged[t]):.3f}   RMSE={rmse(merged['yhat'], merged[t]):.3f}   (n={len(merged):,})")
    
    if not daily_records_forecaster[t]:
        continue
    fc_df = pd.concat(daily_records_forecaster[t], ignore_index=True)
    fc_df.columns = ["ds", "yhat", "target", "forecast_origin", "horizon"]
    fc_df["ds"] = pd.to_datetime(fc_df["ds"])

    merged = fc_df.merge(actuals, on="ds", how="inner")

    if len(merged) == 0:
        continue
    
    print("  Forecaster accuracy: ")
    print(f"  {t:6s}  MAE={mae(merged['yhat'], merged[t]):.3f}   RMSE={rmse(merged['yhat'], merged[t]):.3f}   (n={len(merged):,})")

    if not daily_records_persistence[t]:
        continue
    fc_df = pd.concat(daily_records_persistence[t], ignore_index=True)
    fc_df.columns = ["ds", "yhat", "target", "forecast_origin", "horizon"]
    fc_df["ds"] = pd.to_datetime(fc_df["ds"])

    merged = fc_df.merge(actuals, on="ds", how="inner")

    if len(merged) == 0:
        continue
    
    print("  Persistence accuracy: ")
    print(f"  {t:6s}  MAE={mae(merged['yhat'], merged[t]):.3f}   RMSE={rmse(merged['yhat'], merged[t]):.3f}   (n={len(merged):,})")