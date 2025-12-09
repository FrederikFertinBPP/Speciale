""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from dateutil.relativedelta import relativedelta


EXCELFILE = "historical_data/251105_afry_q32025_central_pt.xlsx"
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

for year in dfs[1].index.year.unique(): # Outer loop
    ## Set up database
    dfs_year = [dfs[ix].loc[dfs[ix].index.year==year] for ix in range(len(dfs))]
    for ix, df in enumerate(dfs_year): # We want to have unique indices for data provided to the forecaster.
        df.index = [p + relativedelta(years=+(weather_years[ix]-year)) for p in df.index]
    data = pd.concat(dfs_year)

    t_start = pd.to_datetime(str(year)+"-01-01", utc=True)
    t_end = t_start + relativedelta(years=+1) + pd.Timedelta(1, 'hour') # Episodic implementation
    t = t_start
    n_scenarios=5
    realizations = {'price': [[] for _ in range(n_scenarios)],
                    'wind': [[] for _ in range(n_scenarios)],
                    'solar': [[] for _ in range(n_scenarios)]}
    while t < t_end:
        timestamp_str = t.strftime("%Y%m%d")
        for n in range(n_scenarios):
            realizations['price'][n] += list(pd.read_csv(root_dir + f"/{year}_scenario_{n}/prices_{timestamp_str}.csv")['price'].values)
            realizations['solar'][n] += list(pd.read_csv(root_dir + f"/{year}_scenario_{n}/solar_{timestamp_str}.csv")['solar'].values)
            realizations['wind'][n] += list(pd.read_csv(root_dir + f"/{year}_scenario_{n}/wind_{timestamp_str}.csv")['wind'].values)
        t += pd.Timedelta(24, 'hour')
    
    import matplotlib.pyplot as plt
    for n in range(n_scenarios):
        plt.plot(sorted(realizations['price'][n]), color="red", label="" if n > 0 else "Simulated")
        plt.plot(sorted(dfs_year[n]['price']), color="blue", label="" if n > 0 else "AFRY projections")
    plt.legend()
    plt.xlabel("Hours")
    plt.ylabel("€/MWh")
    plt.title(f"Price duration curves {year}")
    plt.savefig(f"documentation/AFRY/simulation_validation_prices_{year}.png")
    plt.close()
    for n in range(n_scenarios):
        plt.plot(sorted(realizations['solar'][n]), color="red", label="" if n > 0 else "Simulated")
        plt.plot(sorted(dfs_year[n]['solar']), color="blue", label="" if n > 0 else "AFRY projections")
    plt.legend()
    plt.xlabel("Hours")
    plt.ylabel("Capacity\nFactor")
    plt.title(f"Solar load duration curves {year}")
    plt.savefig(f"documentation/AFRY/simulation_validation_solar_{year}.png")
    plt.close()
    for n in range(n_scenarios):
        plt.plot(sorted(realizations['wind'][n]), color="red", label="" if n > 0 else "Simulated")
        plt.plot(sorted(dfs_year[n]['wind']), color="blue", label="" if n > 0 else "AFRY projections")
    plt.legend()
    plt.xlabel("Hours")
    plt.ylabel("Capacity\nFactor")
    plt.title(f"Wind load duration curves {year}")
    plt.savefig(f"documentation/AFRY/simulation_validation_wind_{year}.png")
    plt.close()