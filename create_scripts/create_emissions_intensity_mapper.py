import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import cache_write
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from common_scripts.utils import set_plotting_style
set_plotting_style()

import xgboost as xgb

def get_fossil_prices(hourly_index, price_indicator="c"):
    path = "historical_data/commodity_prices/"
    def _concat(df1, df2):
        df = pd.concat([df1,df2])
        df.index = pd.to_datetime([pd.Timestamp(t,unit="s") for t in df["t"]],utc=True)
        df = df.sort_index()
        df = df.drop_duplicates(subset="t")
        return df
    df_gas_monthly = pd.read_json(f"{path}gas_futures_monthly.json")
    df_gas_weekly = pd.read_json(f"{path}gas_futures_weekly.json")
    df_gas = _concat(df_gas_monthly, df_gas_weekly)

    df_oil_monthly = pd.read_json(f"{path}oil_brent_monthly.json")
    df_oil_weekly = pd.read_json(f"{path}oil_brent_weekly.json")
    df_oil = _concat(df_oil_monthly, df_oil_weekly)
    
    df_ets = pd.read_excel(f"{path}prices_eu_ets_all.xlsx",sheet_name="Data")
    df_ets.index = pd.to_datetime(df_ets["datetime"],utc=True)
    df_ets = df_ets.sort_index()
    df_ets = df_ets.drop_duplicates(subset="datetime")
    
    def _add_series(df,column,series,key,):
        df = df.copy()
        series_lim = series.loc[(series.index >= df.index[0]) & (series.index <= df.index[-1]),key]
        row_indexer_df = [t in series_lim.index for t in df.index]
        df.loc[row_indexer_df, column] = series_lim.values
        df.loc[df.index[0],column] = series.iloc[max(0, sum(series.index <= df.index[0])-1)][key]
        df.loc[df.index[-1],column] = series.iloc[min(sum(series.index < df.index[-1]), len(series)-1)][key]
        return df

    # hourly_index = pd.to_datetime(pd.date_range(start="2010-01-01", end="2025-12-31"),utc=True)
    df = pd.DataFrame(index=hourly_index, columns=["gas","oil","ets"], dtype=float)
    df = _add_series(df, "gas", df_gas, price_indicator)
    df = _add_series(df, "oil", df_oil, price_indicator)
    df = _add_series(df, "ets", df_ets, "price")
    df = df.interpolate()

    return df







df_ren_prices = pd.read_csv("historical_data/clean_dataframes/server-ENTSOEcountry-PT2024-01-01to2024-12-31.csv", index_col=0)
df_ren_prices.index = pd.to_datetime(df_ren_prices.index, utc=True)

df_emissions = pd.read_csv(f"historical_data/PT_2024_hourly_emissions.csv", index_col=0)
df_emissions.index = pd.to_datetime(df_emissions.index, utc=True)

df_ren_prices = df_ren_prices.loc[df_ren_prices.index.year==2024]
df_emissions = df_emissions.loc[df_ren_prices.index]

y_true = df_emissions["Carbon intensity gCO₂eq/kWh (direct)"]

# Exponentially decreasing relation with wind
fig, ax = plt.subplots(figsize=(16,12))
plt.scatter(df_ren_prices['wind'].values,y_true.values, s=10)
plt.xlabel("Wind Power (MW)")
plt.ylabel("Emissions intensity (gCO2/kWh)")
plt.savefig('documentation/co2_intensity_mapper/systemwind_vs_emissions.png')
plt.close()

fig, ax = plt.subplots(figsize=(16,12))
plt.scatter(df_ren_prices['solar'].values,y_true.values, s=10)
plt.xlabel("Solar Power (MW)")
plt.ylabel("Emissions intensity (gCO2/kWh)")
plt.savefig('documentation/co2_intensity_mapper/systemsolar_vs_emissions.png')
plt.close()

fig, ax = plt.subplots(figsize=(16,12))
plt.scatter(df_ren_prices['solar'].values*df_ren_prices['wind'].values,y_true.values, s=10)
plt.xlabel("Solar Power (MW)")
plt.ylabel("Emissions intensity (gCO2/kWh)")
plt.savefig('documentation/co2_intensity_mapper/solarXwind_vs_emissions.png')
plt.close()

# Second order relation with price
fig, ax = plt.subplots(figsize=(16,12))
plt.scatter(df_ren_prices['price'].values,y_true.values, s=10)
plt.xlabel("Electricity Price (€/MWh)")
plt.ylabel("Emissions intensity (gCO2/kWh)")
plt.savefig('documentation/co2_intensity_mapper/price_vs_emissions.png')
plt.close()

df_ren_prices['wind_sq'] = (df_ren_prices['wind'].values * df_ren_prices['wind'].values)
df_ren_prices['wind_exp'] = np.exp(df_ren_prices['wind'].values / np.max(df_ren_prices['wind'].values))
df_ren_prices['price_sq'] = (df_ren_prices['price'].values * df_ren_prices['price'].values)

model = LinearRegression()
X = df_ren_prices[["price", "wind", "solar"]]

model.fit(X=X, y=y_true)
y_pred = model.predict(X)
print(root_mean_squared_error(y_true, y_pred))

X_ols = sm.add_constant(X)
model = sm.OLS(y_true, X_ols)
results = model.fit()
print(results.summary())

fig, ax = plt.subplots(figsize=(16,12))
plt.scatter(y_true.values, y_pred,label="Model Prediction", s=10)
plt.scatter(y_true.values, y_true.values,label="True Value", color="black", s=10)
plt.xlabel("True emissions intensity (gCO2/kWh)")
plt.ylabel("Predicted emissions intensity (gCO2/kWh)")
plt.legend()
plt.savefig('documentation/co2_intensity_mapper/prediction_performance.png')
plt.close()

fig, ax = plt.subplots(figsize=(16,12))
plt.scatter(range(len(y_pred)),y_true.values-y_pred,label="Model Residuals", s=10)
plt.xlabel("Training observations")
plt.ylabel("Residuals (gCO2/kWh)")
plt.legend()
plt.savefig('documentation/co2_intensity_mapper/prediction_residuals.png')
plt.close()

fig, ax = plt.subplots(figsize=(16,12))
plt.scatter(y_true.index,y_true.values/3.6,label="Hourly emissions intensity", s=10, color='red', alpha=0.2)
plt.axhline(18, label="RFNBO requirement", color='black', lw=3, linestyle="--")
plt.axhline(np.mean(y_true.values)/3.6, label="Average emissions intensity (direct)", color='red', lw=3)
plt.xlabel("Date")
plt.ylabel("Residuals (gCO2/MJ)")
plt.legend()
plt.xlim(y_true.index[0], y_true.index[-1])
plt.tight_layout()
plt.savefig('documentation/co2_intensity_mapper/emissions_data.png')
plt.close()



cache_path = os.getcwd() + "/models/plant_models/emission_factor.pkl"
cache_write(model, cache_path, verbose=True)

print("Done")

y_true = df_ren_prices["price"]
df_fossil = get_fossil_prices(hourly_index=y_true.index)
# df_fossil = df_fossil[["gas","ets"]]
X = pd.concat([df_ren_prices[["wind", "solar"]],df_fossil],axis=1,)
reg = xgb.XGBRegressor(tree_method="hist")
# Fit the model using predictor X and response y.
reg.fit(X=X, y=y_true)
y_pred = reg.predict(X)
print(root_mean_squared_error(y_true,y_pred))

model = LinearRegression()
model.fit(X=X, y=y_true)
y_pred_lr = model.predict(X)
print(root_mean_squared_error(y_true, y_pred_lr))
# X_ols = sm.add_constant(X)
# model = sm.OLS(y_true, X_ols)
# results = model.fit()
# print(results.summary())

df_results = pd.DataFrame(data={"true":np.sort(y_true.values),"XGB":np.sort(y_pred),"LR":np.sort(y_pred_lr)})
df_results.plot()

