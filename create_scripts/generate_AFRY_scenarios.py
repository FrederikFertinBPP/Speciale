""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from data_scripts.data_generator_v2 import DataForecaster
from data_scripts.data_loader import DataLoader


""" 
    ***Integrating and evaluating the case studies on AFRY data***

    We have access to the central scenario of AFRY data.
    Received AFRY data has generated capacity factor profiles for renewables (and other generation/load) for 5 weather years (2012, 14, 15, 17, and 18) and corresponding price profiles. 
    Each weather year is used to generate time series from 2026/01/01 to 2060/12/31.

    * Leap years: *
    Since 2012 was a leap year, we here have solar and wind data for the full year. However, we do not have full data for the prices.
    We do not need this to train our forecaster. 
    The price profile for weather year 2012 is missing price data for the last day of the year for all leap years.
    
    No capacity of wind and solar is stated, but the merit order effect is still in effect in the generated data.
    Max capacity factor (cf) for onshore wind is 0.915, max capacity factor for solar is 1.01.

    
    ** Proposed approach to adopting the framework to the AFRY data: **
    Fit the forecaster on AFRY data for a given year (using all weather years), e.g. fit it on the price, solar, and onshore wind profile. 
    Then generate maybe five scenarios (with daily forecasts, daily realizations, and monthly year-ahead simulations) for each year.
    The size on disc is 35.8 MB per generated year (episode). The full disc space used will be 35.8 * 5 * 35 = 6.3 GB.

    Data generation will probably take 24 hours or so.
"""

EXCELFILE = "historical_data/AFRY/251105_afry_q32025_central_pt.xlsx"
weather_year_ids = (18,19,20,21,22)
weather_years = (2012, 2014, 2015, 2017, 2018)
dfs = []
### Load AFRY data:
def _fill_na_hours(series): # We fill in with a copy of the previous day.
    na_indices = series.loc[series.isna()].index
    for hour in na_indices:
        copied_hour = hour - pd.Timedelta(1, 'day')
        price = series[copied_hour]
        series.loc[series.index == hour] = price
    return series

for w in weather_year_ids:
    price_sheet_name = "Price_124" + str(w)
    gen_sheet_name = "Generation_124" + str(w)
    df_price = pd.read_excel(EXCELFILE, sheet_name=price_sheet_name, usecols=list(range(4)), skiprows=3)
    series_price = df_price.set_index(pd.to_datetime(df_price['datetimekey']))['baseprice']
    if w==18: # This is weather year 2012 with missing price data.
        series_price = _fill_na_hours(series_price)
    df_gen = pd.read_excel(EXCELFILE, sheet_name=gen_sheet_name, usecols=list(range(11)), skiprows=3)
    df_gen = df_gen[["datetimekey", "Onshore Wind", "Solar"]]
    df_gen.columns = ["time", "wind", "solar"]
    df_gen.loc[:,"price"] = series_price.values
    df = df_gen.set_index(pd.to_datetime(df_gen['time'], utc=True))[['price', 'wind', 'solar']]
    dfs.append(df)

### Simulate own data based on AFRY data:
root_dir = "scenario_data/AFRY"
date_format = "%Y-%m-%d %H"
float_format = '%.3f'

decision_horizon = 24
planning_horizon = 4 * decision_horizon
n_scenarios = 5

database = DataLoader()
index_ = pd.date_range(start=pd.Timestamp(str(weather_years[0])), end=pd.Timestamp(str(dfs[1].index[-1].year+1)), freq='d')
caps_index = index_.tz_localize(None).to_period('M').unique()
caps = pd.DataFrame(index = caps_index, data={'wind': np.ones(len(caps_index)), 'solar': np.ones(len(caps_index))})
database.caps = caps
for year in dfs[1].index.year.unique(): # Outer loop
    ## Set up database
    dfs_year = [dfs[ix].loc[dfs[ix].index.year==year] for ix in range(len(dfs))]
    for ix, df in enumerate(dfs_year): # We want to have unique indices for data provided to the forecaster.
        df.index = [p + relativedelta(years=+(weather_years[ix]-year)) for p in df.index]
    data = pd.concat(dfs_year)

    database.data = database._create_seasonal_features(data, prod_columns=['wind', 'solar'])
    
    ## Fit forecaster
    forecaster = DataForecaster(database, verbose=False)
    forecaster.build_simulation_models()
    forecaster.price_model.trend_model.coef_ = np.asarray([[0]]) # Delete any fit trend, as it is not meaningful within a year.

    ## Define period of interest:
    t_start = pd.to_datetime(str(year)+"-01-01", utc=True)
    t_end = t_start + relativedelta(years=+1) + pd.Timedelta(1, 'hour') # Episodic implementation
    forecaster.t_init = t_start

    for n in range(n_scenarios):
        scenario_dir = os.path.join(root_dir, f"{year}_scenario_{n}")
        os.makedirs(scenario_dir, exist_ok=True)

        t = t_start
        while t < t_end:
            system_solar_realization, system_wind_realization = forecaster.realize_vre(start=t, end=t + pd.Timedelta(decision_horizon-1, 'h'))
            forecasts = forecaster.forecast(start=t, end=t+pd.Timedelta(planning_horizon-1, 'h'), n_forecasts=10, simulate_prices=True) # list of DFs
            if t.is_month_start and t + pd.Timedelta(24, 'hour') < t_end:
                year_simulations = forecaster.simulate_period(start = t, end=t_end, n_sims=5)
            real_prices = forecaster.realize_prices(start=t, end=t+pd.Timedelta(decision_horizon-1, 'h'))

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