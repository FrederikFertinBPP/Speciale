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
    scenario_dir = os.path.join(root_dir, f"Anders_scenario_{n}")
    os.makedirs(scenario_dir, exist_ok=True)
    
    solar_year = []
    wind_year = []
    price_year = []

    t = t_start
    while t < t_end:
        timestamp_str = t.strftime("%Y%m%d")
        solar_year += list(pd.read_csv(f"{scenario_dir}/solar_{timestamp_str}.csv")["solar"].values)
        wind_year += list(pd.read_csv(f"{scenario_dir}/wind_{timestamp_str}.csv")['wind'].values)
    
        price_year += list(pd.read_csv(f"{scenario_dir}/prices_{timestamp_str}.csv")['price'].values)
        # system_solar_realization, system_wind_realization = forecaster.realize_vre(start=t, end=t + pd.Timedelta(decision_horizon-1, 'h'))
        # real_prices = forecaster.realize_prices(start=t, end=t+pd.Timedelta(decision_horizon-1, 'h'))

        t += pd.Timedelta(24, 'hour')
    df = pd.DataFrame(data={"price": price_year, "wind": wind_year, "solar": solar_year})
    path = os.path.join(scenario_dir, f"realized_year.csv")
    df.to_csv(path, float_format=float_format)
    print(f"Wrote {n}")
    