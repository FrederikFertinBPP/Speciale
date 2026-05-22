""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from data_scripts.data_loader import HistoricalData, get_fossil_prices
from data_scripts.data_generator_v3 import DataForecaster

import xgboost as xgb
from time import time

#%% Data retrieval - There is only data from 2015 and forward for Portugal
t_s = time()
start   = pd.Timestamp('20150101', tz='UTC')
end     = pd.Timestamp('20251231', tz='UTC')
data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')
t_e = time()
print(f"Data retrieval and preprocessing took {t_e-t_s:.2f} seconds.")

t_s = time()
# exog_model = xgb.XGBRegressor(booster="gblinear", reg_alpha=0.1, reg_lambda=0.1)
forecaster = DataForecaster(database=data_object, r_load_tag="", other_exog_tags = ["gas_with_ets"], stochastic_price_model="GARCHX")
forecaster.build_simulation_models()
t_e = time()
print(f"Data forecaster and models built in {t_e-t_s:.2f} seconds.")


start = pd.to_datetime("2025-01-01",utc=True)
sims = forecaster.simulate_year_ahead(start=start, n_sims=5)
forecaster.investigate_annual_duration_curves(sims, resource="price")
exog = [pd.DataFrame({"wind":sim["wind"]*forecaster.wind_model.caps.loc["2025-01","wind"],
                      "solar":sim["solar"]*forecaster.solar_model.caps.loc["2025-01","solar"]})
                      for sim in sims]
forecaster.investigate_annual_duration_curves(exog, resource="wind")
forecaster.investigate_annual_duration_curves(exog, resource="solar")
df_gen = data_object.data[["wind", "solar", "Residual Load"]]
df_fossil = get_fossil_prices(data_object.data.index)

X_all_features = pd.concat([data_object.data[["wind", "solar", "Residual Load", "Forecasted Load"]], df_fossil],axis=1)

y_true = data_object.data["price"]
X = pd.concat([df_gen,df_fossil[["gas_with_ets"]]],axis=1)

X_train, X_test = X.loc[X.index.year < 2024], X.loc[X.index.year >= 2024]
y_train, y_test = y_true.loc[y_true.index.year < 2024], y_true.loc[y_true.index.year >= 2024]

reg = xgb.XGBRegressor(tree_method="hist")
# Fit the model using predictor X and response y.
reg.fit(X=X_train, y=y_train)

# In-sample errors
y_pred_train = reg.predict(X_train)
error_train = y_train - y_pred_train
rmse_train = np.sqrt(np.mean(error_train**2))

for year in X_train.index.year.unique():
    error_train_year = y_train.loc[y_train.index.year==year] - y_pred_train[y_train.index.year==year]
    print(year, np.sqrt(np.mean(error_train_year**2)))

print("In-sample RMSE:", rmse_train)

# standardized_error_train = error_train / X_train["gas"]
# std_error_train = (error_train - error_train.mean()) / error_train.std()
# std_error_train_gas = (standardized_error_train - standardized_error_train.mean()) / standardized_error_train.std()

# Out-of-sample errors
y_pred_test = reg.predict(X_test)
error_test = y_test - y_pred_test
rmse_test = np.sqrt(np.mean(error_test**2))

for year in X_test.index.year.unique():
    error_test_year = y_test.loc[y_test.index.year==year] - y_pred_test[y_test.index.year==year]
    print(year, np.sqrt(np.mean(error_test_year**2)))

#%% Forecasting setup
forecaster = DataForecaster(data_object,
                            cache_id="Anders",
                            verbose=False, # Takes under 10 minutes to create, set verbose equals True to see progress, but know that verbose=True for the unpickled object.
                            cache_replace=True,
                            )
forecaster.build_simulation_models(to_pickle=True)
print("Data forecaster and models built and pickled.")


hourly_index = sims[0].index
ym         = hourly_index.tz_localize(None).to_period('M')
solar_caps = ym.map(forecaster.database.caps[forecaster.solar_tag])
wind_caps  = ym.map(forecaster.database.caps[forecaster.wind_tag])
exog = [pd.DataFrame(
    index=hourly_index,
    data={forecaster.solar_tag: sims[ix][forecaster.solar_tag] * solar_caps.values,
        forecaster.wind_tag:  sims[ix][forecaster.wind_tag]   * wind_caps.values,
        forecaster.price_tag: sims[ix][forecaster.price_tag]}) for ix in range(len(sims))]
documentation = True
import matplotlib.pyplot as plt
simulations = exog
wind_solar_price_corr = {}
wind_solar_price_corr['wind-solar-hist']    = np.corrcoef(forecaster.data.loc[(forecaster.data.is_day)&(forecaster.data.index.year==2025),'solar'], forecaster.data.loc[(forecaster.data.is_day)&(forecaster.data.index.year==2025),'wind'])[0,1]
wind_solar_price_corr['price-solar-hist']   = np.corrcoef(forecaster.data.loc[(forecaster.data.is_day)&(forecaster.data.index.year==2025),'solar'], forecaster.data.loc[(forecaster.data.is_day)&(forecaster.data.index.year==2025),'price'])[0,1]
wind_solar_price_corr['wind-price-hist']    = np.corrcoef(forecaster.data.loc[forecaster.data.index.year==2025,'price'], forecaster.data.loc[forecaster.data.index.year==2025,'wind'])[0,1]
#%% Some plots and statistics
if documentation:
    plot_horizon = 4*24
    start_hour = 0
    end_hour = start_hour + plot_horizon
    fig, ax1 = plt.subplots(1, figsize=(15,10))
    ax2 = ax1.twinx()
    ax1.plot(simulations[0].index[start_hour:end_hour],simulations[0]['price'].iloc[start_hour:end_hour], color='black', label='Prices')
    ax2.plot(simulations[0].index[start_hour:end_hour],simulations[0]['solar'].iloc[start_hour:end_hour], color='red', label='Solar')
    ax2.plot(simulations[0].index[start_hour:end_hour],simulations[0]['wind'].iloc[start_hour:end_hour], color='blue', label='Wind')
    ax1.set_ylabel('€/MWh')
    ax2.set_ylabel('MW')
    ax1.set_xlim(simulations[0].index[start_hour], simulations[0].index[end_hour-1])
    ax2.set_ylim(0, 5000)
    ax1.set_ylim(0, 200)
    ax1.set_title('Simulated Profiles')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(handles = h1+h2, labels=l1+l2)
    plt.savefig(f'documentation/{forecaster.plot_dir}simulated_profiles.png', bbox_inches='tight', dpi=300)
    plt.close()
    start_hour = simulations[0].index[start_hour]
    end_hour = simulations[0].index[end_hour]
    fig, ax1 = plt.subplots(1, figsize=(15,10))
    ax2 = ax1.twinx()
    ax1.plot(forecaster.data.loc[start_hour:end_hour].index,forecaster.data['price'].loc[start_hour:end_hour], color='black', label='Prices')
    ax2.plot(forecaster.data.loc[start_hour:end_hour].index,forecaster.data['solar'].loc[start_hour:end_hour], color='red', label='Solar')
    ax2.plot(forecaster.data.loc[start_hour:end_hour].index,forecaster.data['wind'].loc[start_hour:end_hour], color='blue', label='Wind')
    ax1.set_ylabel('€/MWh')
    ax2.set_ylabel('MW')
    ax1.set_xlim(start_hour, end_hour)
    ax2.set_ylim(0, 5000)
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