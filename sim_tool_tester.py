from data_scripts import DataForecaster, HistoricalData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from utils import cache_write

documentation = False

#%% Data retrieval - There is only data from 2015 and forward for Portugal
start   = pd.Timestamp('20160101', tz='UTC')
end     = pd.Timestamp('20201231', tz='UTC')
data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')

#%% Forecasting setup
forecaster = DataForecaster(data_object,
                            documentation=documentation,
                            seasonal_price_regression=True,
                            cache_id='seasonal',
                            verbose=True,
                            )

# Historical crosscorrelations
wind_solar_price_corr = {}
wind_solar_price_corr['wind-solar-hist']    = np.corrcoef(forecaster.train_data.loc[forecaster.train_data.is_day,'solar'], forecaster.train_data.loc[forecaster.train_data.is_day,'wind'])[0,1]
wind_solar_price_corr['price-solar-hist']   = np.corrcoef(forecaster.train_data.loc[forecaster.train_data.is_day,'solar'], forecaster.train_data.loc[forecaster.train_data.is_day,'price'])[0,1]
wind_solar_price_corr['wind-price-hist']    = np.corrcoef(forecaster.train_data['price'], forecaster.train_data['wind'])[0,1]

forecaster.build_simulation_models(hmm=False)

#%% Simulate a full year
year = forecaster.test_data.index.year.unique()
year= year[0] if len(year) == 1 else 2020
n_sims = 50
t_s = time()
print(f"Simulating {year} {n_sims} times with {data_object.caps.to_dict()}")
simulations = forecaster.simulate(year, data_object.caps, n_sims=n_sims)
print(f"Simulated {len(simulations)} scenarios for year {year} in {time()-t_s:.2f} seconds.")
if documentation:
    forecaster.investigate_test_simulation_monthly(simulations, resource='price')
    forecaster.investigate_test_simulation_monthly(simulations, resource='wind')
    forecaster.investigate_test_simulation_monthly(simulations, resource='solar')
    forecaster.investigate_annual_duration_curves(simulations, year=year, resource='price')
    forecaster.investigate_annual_duration_curves(simulations, year=year, resource='wind')
    forecaster.investigate_annual_duration_curves(simulations, year=year, resource='solar')

#%% Some plots and statistics
if documentation:
    plot_horizon = 4*24
    start_hour = 3000
    end_hour = start_hour + plot_horizon
    fig, ax1 = plt.subplots(1, figsize=(15,12))
    ax2 = ax1.twinx()
    ax1.plot(simulations[0]['price'].index[start_hour:end_hour],simulations[0]['price'].iloc[start_hour:end_hour], color='black', label='Prices')
    ax2.plot(simulations[0]['price'].index[start_hour:end_hour],simulations[0]['solar'].iloc[start_hour:end_hour], color='red', label='Solar')
    ax2.plot(simulations[0]['price'].index[start_hour:end_hour],simulations[0]['wind'].iloc[start_hour:end_hour], color='blue', label='Wind')
    ax1.set_ylabel('€/MWh')
    ax2.set_ylabel('MW')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(handles = h1+h2, labels=l1+l2)
    plt.savefig('documentation/simulated_profiles.png')
    plt.close()

# Statistics on crosscorrelation of simulated data.
t_i = data_object._specify_time_data(simulations[0]['price'])
wind_solar_price_corr['wind-solar']     = np.mean([np.corrcoef(sim['wind'].loc[t_i.is_day, 'wind'].values, sim['solar'].loc[t_i.is_day, 'solar'].values)[0,1] for sim in simulations])
wind_solar_price_corr['price-solar']    = np.mean([np.corrcoef(sim['price'].loc[t_i.is_day, 'price'].values, sim['solar'].loc[t_i.is_day, 'solar'].values)[0,1] for sim in simulations])
wind_solar_price_corr['wind-price']     = np.mean([np.corrcoef(sim['wind']['wind'].values, sim['price']['price'].values)[0,1] for sim in simulations])
wind_solar_price_corr['wind-solar-std']     = np.std([np.corrcoef(sim['wind'].loc[t_i.is_day, 'wind'].values, sim['solar'].loc[t_i.is_day, 'solar'].values)[0,1] for sim in simulations])
wind_solar_price_corr['price-solar-std']    = np.std([np.corrcoef(sim['price'].loc[t_i.is_day, 'price'].values, sim['solar'].loc[t_i.is_day, 'solar'].values)[0,1] for sim in simulations])
wind_solar_price_corr['wind-price-std']     = np.std([np.corrcoef(sim['wind']['wind'].values, sim['price']['price'].values)[0,1] for sim in simulations])
print("Crosscorrelations (historical vs simulated):")
for k,v in wind_solar_price_corr.items():
    print(f"{k}: {v:.3f}")

#%% Save simulations
conv_sims = []
for sim in simulations:
    conv_sims.append(pd.concat([sim['solar'], sim['wind'], sim['price']], axis=1))
cache_write(conv_sims, f'simulations/{n_sims}_simulated_scenarios_list_{year}_{int(time())}.pkl', verbose=True)