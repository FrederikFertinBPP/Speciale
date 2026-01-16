""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_scripts import DataForecaster, HistoricalData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from common_scripts.utils import cache_read
from model_scripts.environment import VRESystemToAssetMapping

import seaborn as sns

sns.set_theme("notebook", font_scale=1.5, style="darkgrid")
plt.rcParams['font.size'] = 16
# set legend fontsize to 14
plt.rcParams['legend.fontsize'] = 18
# set the font weight of the legend to bold
plt.rcParams['legend.title_fontsize'] = 18
# set the font size of the x and y labels to 14
plt.rcParams['axes.labelsize'] = 18
# set the font weight of the x and y labels to bold
plt.rcParams['axes.labelweight'] = 'bold'
# set the font size of the x and y ticks to 12
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# set the font size of the title to 16
plt.rcParams['axes.titlesize'] = 18
# set the font weight of the title to bold
plt.rcParams['axes.titleweight'] = 'bold'

def investigate_annual_duration_curves(simulations, resource='price'):
    resource = 'price'
    plt.figure(figsize=(10, 6))
    train_data = forecaster.train_data[resource]
    simulated_data = [sim[resource] for sim in simulations]

    if resource == 'price':
        ylabel="[€/MWh]"
    else:
        ylabel="MW"
    
    # for ix, sim in enumerate(simulated_data):
    #     lbl = "" if ix > 0 else f"Simulations of year {year}" 
    #     plt.plot(np.sort(sim[resource]), color='blue', alpha=0.2, label=lbl)
    # Draw confidence intervals
    mtx = np.asarray([np.sort(sim.values).reshape(-1) for sim in simulated_data])
    p_low = np.percentile(mtx, 5, axis=0)
    p_high = np.percentile(mtx, 95, axis=0)
    plt.fill_between(range(len(p_low)), p_low, p_high, color='blue', alpha=0.2, label='90% CI')
    plt.plot(np.sort(np.mean(mtx, axis=0)), color='black', alpha=0.8, label='Mean of simulations')
    for yr in train_data.index.year.unique():
        plt.plot(np.sort(train_data.loc[train_data.index.year==yr]), label=yr)
    plt.xlabel("Hours")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f'documentation/{forecaster.plot_dir}annual_duration_curve_{resource}.png')
    plt.close()

documentation = True
if False:
    #%% Data retrieval - There is only data from 2015 and forward for Portugal
    start   = pd.Timestamp('20230101', tz='UTC')
    end     = pd.Timestamp('20241231', tz='UTC')
    data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')

    #%% Forecasting setup
    forecaster = DataForecaster(data_object,
                                documentation=documentation,
                                cache_id='Anders',
                                verbose=False,
                                #plot_dir="Anders_simulations"
                                )
    forecaster.build_simulation_models()
else:
    forecaster = DataForecaster(from_pickle=True, cache_id="Anders")
    forecaster = forecaster.unpickle()
    data_object = forecaster.database

# Historical crosscorrelations
wind_solar_price_corr = {}
wind_solar_price_corr['wind-solar-hist']    = np.corrcoef(forecaster.train_data.loc[forecaster.train_data.is_day,'solar'], forecaster.train_data.loc[forecaster.train_data.is_day,'wind'])[0,1]
wind_solar_price_corr['price-solar-hist']   = np.corrcoef(forecaster.train_data.loc[forecaster.train_data.is_day,'solar'], forecaster.train_data.loc[forecaster.train_data.is_day,'price'])[0,1]
wind_solar_price_corr['wind-price-hist']    = np.corrcoef(forecaster.train_data['price'], forecaster.train_data['wind'])[0,1]
x = forecaster.price_model.arima_model.arima_res_.simulate(3000)
# forecaster.build_simulation_models(hmm=False)

#%% Simulate a full year
year = forecaster.test_data.index.year.unique()
year= 2024#year[0] if len(year) == 1 else 2020
n_sims = 100
t_s = time()
forecaster.plot_dir = 'Anders_simulations/'
simulations = forecaster.simulate(year, n_sims=n_sims)
print(f"Simulated {len(simulations)} scenarios for year {year} in {time()-t_s:.2f} seconds.")
if documentation:
    forecaster.investigate_test_simulation_monthly(simulations, resource='price')
    forecaster.investigate_test_simulation_monthly(simulations, resource='wind')
    forecaster.investigate_test_simulation_monthly(simulations, resource='solar')
    forecaster.investigate_annual_duration_curves(simulations, resource='price')
    forecaster.investigate_annual_duration_curves(simulations, resource='wind')
    forecaster.investigate_annual_duration_curves(simulations, resource='solar')

#%% Some plots and statistics
if documentation:
    plot_horizon = 4*24
    start_hour = 3000
    end_hour = start_hour + plot_horizon
    fig, ax1 = plt.subplots(1, figsize=(15,10))
    ax2 = ax1.twinx()
    ax1.plot(simulations[0]['price'].index[start_hour:end_hour],simulations[0]['price'].iloc[start_hour:end_hour], color='black', label='Prices')
    ax2.plot(simulations[0]['price'].index[start_hour:end_hour],simulations[0]['solar'].iloc[start_hour:end_hour], color='red', label='Solar')
    ax2.plot(simulations[0]['price'].index[start_hour:end_hour],simulations[0]['wind'].iloc[start_hour:end_hour], color='blue', label='Wind')
    ax1.set_ylabel('€/MWh')
    ax2.set_ylabel('MW')
    ax1.set_xlim(forecaster.test_data['price'].index[start_hour], forecaster.test_data['price'].index[end_hour-1])
    ax2.set_ylim(0, 4000)
    ax1.set_ylim(0, 200)
    ax1.set_title('Simulated Profiles')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(handles = h1+h2, labels=l1+l2)
    plt.savefig(f'documentation/{forecaster.plot_dir}simulated_profiles.png', bbox_inches='tight', dpi=300)
    plt.close()
    fig, ax1 = plt.subplots(1, figsize=(15,10))
    ax2 = ax1.twinx()
    ax1.plot(forecaster.test_data['price'].index[start_hour:end_hour],forecaster.test_data['price'].iloc[start_hour:end_hour], color='black', label='Prices')
    ax2.plot(forecaster.test_data['price'].index[start_hour:end_hour],forecaster.test_data['solar'].iloc[start_hour:end_hour], color='red', label='Solar')
    ax2.plot(forecaster.test_data['price'].index[start_hour:end_hour],forecaster.test_data['wind'].iloc[start_hour:end_hour], color='blue', label='Wind')
    ax1.set_ylabel('€/MWh')
    ax2.set_ylabel('MW')
    ax1.set_xlim(forecaster.test_data['price'].index[start_hour], forecaster.test_data['price'].index[end_hour-1])
    ax2.set_ylim(0, 4000)
    ax1.set_ylim(0, 200)
    ax1.set_title('Historical Profiles')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(handles = h1+h2, labels=l1+l2)
    plt.savefig(f'documentation/{forecaster.plot_dir}historical_profiles.png', bbox_inches='tight', dpi=300)
    plt.close()

# Statistics on crosscorrelation of simulated data.
t_i = data_object._specify_time_data(pd.DataFrame(index=simulations[0]['price'].index))
wind_solar_price_corr['wind-solar']     = np.mean([np.corrcoef(sim['wind'].loc[t_i.is_day].values, sim['solar'].loc[t_i.is_day].values)[0,1] for sim in simulations])
wind_solar_price_corr['price-solar']    = np.mean([np.corrcoef(sim['price'].loc[t_i.is_day].values, sim['solar'].loc[t_i.is_day].values)[0,1] for sim in simulations])
wind_solar_price_corr['wind-price']     = np.mean([np.corrcoef(sim['wind'].values, sim['price'].values)[0,1] for sim in simulations])
wind_solar_price_corr['wind-solar-std']     = np.std([np.corrcoef(sim['wind'].loc[t_i.is_day].values, sim['solar'].loc[t_i.is_day].values)[0,1] for sim in simulations])
wind_solar_price_corr['price-solar-std']    = np.std([np.corrcoef(sim['price'].loc[t_i.is_day].values, sim['solar'].loc[t_i.is_day].values)[0,1] for sim in simulations])
wind_solar_price_corr['wind-price-std']     = np.std([np.corrcoef(sim['wind'].values, sim['price'].values)[0,1] for sim in simulations])
print("Crosscorrelations (historical vs simulated):")
for k,v in wind_solar_price_corr.items():
    print(f"{k}: {v:.3f}")

# Plot crosscorrelation distributions
if documentation:
    fig, axs = plt.subplots(3,1, figsize=(6,18), tight_layout=True)
    axs[0].hist([np.corrcoef(sim['wind'].loc[t_i.is_day].values, sim['solar'].loc[t_i.is_day].values)[0,1] for sim in simulations], bins=10, alpha=0.7, color='blue', label='Simulated')
    axs[0].axvline(wind_solar_price_corr['wind-solar-hist'], color='black', linestyle='--', label='Mean Historical')
    axs[0].axvline(wind_solar_price_corr['wind-solar'], color='red', linestyle=':', label='Mean Simulated')
    axs[0].set_title('Wind-Solar Correlation')
    axs[0].set_xlabel('Correlation')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    axs[1].hist([np.corrcoef(sim['price'].loc[t_i.is_day].values, sim['solar'].loc[t_i.is_day].values)[0,1] for sim in simulations], bins=10, alpha=0.7, color='orange', label='Simulated')
    axs[1].axvline(wind_solar_price_corr['price-solar-hist'], color='black', linestyle='--', label='Mean Historical')
    axs[1].axvline(wind_solar_price_corr['price-solar'], color='red', linestyle=':', label='Mean Simulated')
    axs[1].set_title('Price-Solar Correlation')
    axs[1].set_xlabel('Correlation')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    axs[2].hist([np.corrcoef(sim['wind'].values, sim['price'].values)[0,1] for sim in simulations], bins=10, alpha=0.7, color='green', label='Simulated')
    axs[2].axvline(wind_solar_price_corr['wind-price-hist'], color='black', linestyle='--', label='Mean Historical')
    axs[2].axvline(wind_solar_price_corr['wind-price'], color='red', linestyle=':', label='Mean Simulated')
    axs[2].set_title('Wind-Price Correlation')
    axs[2].set_xlabel('Correlation')
    axs[2].set_ylabel('Frequency')
    axs[2].legend()
    plt.savefig(f'documentation/{forecaster.plot_dir}crosscorrelation_distributions.png')
    plt.close()

#%% Save simulations
cache_path_mappers = os.getcwd() + "/models/plant_models/"
solar_mapper = cache_read(cache_path_mappers + "solar.pkl")
solar_mapper = VRESystemToAssetMapping(solar_mapper)
wind_mapper = cache_read(cache_path_mappers + "wind.pkl")
wind_mapper = VRESystemToAssetMapping(wind_mapper)
conv_sims = []
year_month_index = simulations[0]['solar'].index.tz_localize(None).to_period('M')
solar_caps = year_month_index.map(forecaster.solar_model.caps['solar'])
wind_caps = year_month_index.map(forecaster.wind_model.caps['wind'])
for ix, sim in enumerate(simulations):
    df = pd.DataFrame(index = sim['solar'].index,
                      data={'price_€_MWh':sim['price'],
                            'wind_system_MW':sim['wind'],
                            'solar_system_MW':sim['solar'],
                            'wind_system_cf':sim['wind'] / wind_caps,
                            'solar_system_cf':sim['solar'] / solar_caps,
                            'wind_asset_cf':wind_mapper(sim['wind'] / wind_caps),
                            'solar_asset_cf':solar_mapper(sim['solar']/solar_caps)})
    df.to_csv(f'simulations/csv/sim_{ix}.csv')