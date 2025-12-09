""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_scripts.data_generator_v2 import DataForecaster
import pandas as pd
from dateutil.relativedelta import relativedelta


forecaster = DataForecaster(from_pickle=True, cache_id="Anders")
forecaster = forecaster.unpickle()
t_start = pd.to_datetime("2024-01-01", utc=True)
t_end = t_start + relativedelta(years=+1) + pd.Timedelta(1, 'hour') # Episodic implementation

decision_horizon = 24
planning_horizon = 4 * decision_horizon

n_scenarios = 50

root_dir = "scenario_data"
date_format="%Y-%m-%d %H"
float_format='%.3f'

for n in range(n_scenarios):
    scenario_dir = os.path.join(root_dir, f"{forecaster.cache_id}_scenario_{n}")
    os.makedirs(scenario_dir, exist_ok=True)

    solar_year = []
    wind_year = []
    price_year = []

    t = t_start
    while t < t_end:
        system_solar_realization, system_wind_realization = forecaster.realize_vre(start=t, end=t + pd.Timedelta(decision_horizon-1, 'h'))
        solar_year += list(system_solar_realization["solar"].values)
        wind_year += list(system_wind_realization["wind"].values)

        forecasts = forecaster.forecast(start=t, end=t+pd.Timedelta(planning_horizon-1, 'h'), n_forecasts=10, simulate_prices=True) # list of DFs
        if t.is_month_start and t + pd.Timedelta(24, 'hour') < t_end:
            year_simulations = forecaster.simulate_period(start = t, end=t_end, n_sims=5)
        real_prices = forecaster.realize_prices(start=t, end=t+pd.Timedelta(decision_horizon-1, 'h'))
        price_year += list(real_prices["price"].values)

        # Define filenames based on timestamp
        timestamp_str = t.strftime("%Y%m%d")
        solar_file = os.path.join(scenario_dir, f"solar_{timestamp_str}.csv")
        wind_file = os.path.join(scenario_dir, f"wind_{timestamp_str}.csv")
        price_file = os.path.join(scenario_dir, f"prices_{timestamp_str}.csv")

        # Save realizations
        system_solar_realization.to_csv(solar_file, index=False, date_format=date_format, float_format=float_format)
        system_wind_realization.to_csv(wind_file, index=False, date_format=date_format, float_format=float_format)
        real_prices.to_csv(price_file, index=False, date_format=date_format, float_format=float_format)

        # Save forecasts (list of DataFrames)
        for i, df in enumerate(forecasts):
            forecast_file = os.path.join(scenario_dir, f"forecast_{timestamp_str}_{i}.csv")
            df.to_csv(forecast_file, index=False, date_format=date_format, float_format=float_format)

        # Save yearly simulations if generated
        if t.is_month_start:
            for i, df in enumerate(year_simulations):
                sim_file = os.path.join(scenario_dir, f"year_sim_{timestamp_str}_{i}.csv")
                df.to_csv(sim_file, index=False, date_format=date_format, float_format=float_format)

        t += pd.Timedelta(24, 'hour')
    df = pd.DataFrame(data={"price": price_year, "wind": wind_year, "solar": solar_year})
    path = os.path.join(scenario_dir, f"realized_year.csv")
    df.to_csv(path, float_format=float_format)
    print(f"Wrote {n}")