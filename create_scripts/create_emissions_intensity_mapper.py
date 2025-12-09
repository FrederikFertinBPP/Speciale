import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import cache_write
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme("paper", font_scale=1.5, style="darkgrid")
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

cache_path = os.getcwd() + "/models/plant_models/emission_factor.pkl"
cache_write(model, cache_path, verbose=True)

print("Done")