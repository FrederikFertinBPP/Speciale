import pandas as pd
import numpy as np
from data_scripts import HistoricalData
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from common_scripts.utils import log_transform
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from common_scripts.utils import cache_write

def get_hourly_df(resource='pv'):
    df = pd.read_csv(f"Historical Data/renewablesninja_{resource}_PT.csv", skiprows=3)
    df.index = pd.to_datetime(df['time'], utc=True)
    df = df['electricity']
    return df

df_wind = get_hourly_df('wind')
df_solar = get_hourly_df('pv')
start   = pd.Timestamp('20160101', tz='UTC')
end     = pd.Timestamp('20201231', tz='UTC')
data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')

""" Investigate correlation of renwables ninja data and system level data. """
df_wind_system = data_object.data.loc[df_wind.index, 'wind'] / data_object.caps.loc[df_wind.index.year.unique(), 'wind'].values[0]
df_solar_system = data_object.data.loc[df_solar.index, 'solar'] / data_object.caps.loc[df_solar.index.year.unique(), 'solar'].values[0]

ti_is_day = data_object.data.loc[df_solar.index, 'is_day']

corr_wind = np.corrcoef(df_wind, df_wind_system) # 0.800
corr_solar = np.corrcoef(df_solar, df_solar_system) # 0.955

""" Solar model fit """
solar_model = LinearRegression()
solar_model.fit(X=df_solar_system.loc[ti_is_day].values.reshape(-1, 1), y=df_solar.loc[ti_is_day])

dc_solar = Polynomial.fit(x=np.sort(df_solar_system.loc[ti_is_day]), y = np.sort(df_solar.loc[ti_is_day]), deg=1)


cut=range(1000,1200)
plt.plot(df_solar.index[cut], df_solar.iloc[cut].values, label="ninja_data")
plt.plot(df_solar.index[cut], df_solar_system.iloc[cut].values, label="ENTSOE_eurostat")
plt.plot(df_solar.index[cut], solar_model.predict(df_solar_system.iloc[cut].values.reshape(-1,1)), label="1st_order_model")
plt.plot(df_solar.index[cut], dc_solar(df_solar_system.iloc[cut]), label="DC_model")
plt.legend()
plt.savefig('profiles_solar_models.png')
plt.close()

plt.plot(np.sort(df_solar.loc[ti_is_day]), label="ninja_data")
plt.plot(np.sort(df_solar_system.loc[ti_is_day]), label="ENTSOE_eurostat")
plt.plot(np.sort(solar_model.predict(df_solar_system.loc[ti_is_day].values.reshape(-1,1))), label="1st_order_model")
plt.plot(np.sort(dc_solar(df_solar_system.loc[ti_is_day])), label="DC_model")
plt.legend()
plt.savefig('duration_curves_solar_models.png')
plt.close()



""" Wind model fit: """
wind_model = LinearRegression()
wind_model.fit(X=df_wind_system.values.reshape(-1, 1), y=df_wind)
wind_model_log = LinearRegression()
wind_model_log.fit(X=log_transform(df_wind_system.values.reshape(-1, 1)), y=df_wind)

wind_model_sqrt = LinearRegression(fit_intercept=False)
X = np.transpose(np.array([df_wind_system.values, np.sqrt(df_wind_system.values)]))
wind_model_sqrt.fit(X=X, y=df_wind)

wind_model_pol = Polynomial.fit(x=df_wind_system, y=df_wind, deg=3).convert()

wind_model_svr = SVR(kernel='rbf', C=100)
wind_model_svr.fit(X=df_wind_system.values.reshape(-1, 1), y=df_wind)


# Just fit the duration curve, basically a perfect fit with 3rd order polynomial:
dc_wind = Polynomial.fit(x=np.sort(df_wind_system), y = np.sort(df_wind), deg=3)

cut=range(1000,1200)
plt.plot(df_wind.index[cut], df_wind.iloc[cut].values, label="ninja_data")
plt.plot(df_wind.index[cut], df_wind_system.iloc[cut].values, label="ENTSOE_eurostat")
plt.plot(df_wind.index[cut], wind_model.predict(df_wind_system.iloc[cut].values.reshape(-1,1)), label="1st_order_model")
plt.plot(df_wind.index[cut], wind_model_log.predict(log_transform(df_wind_system.iloc[cut].values.reshape(-1,1))), label="log_model")
plt.plot(df_wind.index[cut], wind_model_pol(df_wind_system.iloc[cut]), label="5th_order_model")
plt.plot(df_wind.index[cut], dc_wind(df_wind_system.iloc[cut]), label="DC_model")
X_clip = X[cut]
plt.plot(df_wind.index[cut], wind_model_sqrt.predict(X_clip), label="sqrt_model")
plt.plot(df_wind.index[cut], wind_model_svr.predict(df_wind_system.iloc[cut].values.reshape(-1,1)), label="SVR_model")
plt.legend()
plt.savefig('profiles_wind_models.png')
plt.close()

plt.plot(np.sort(df_wind), label="ninja_data")
plt.plot(np.sort(df_wind_system), label="ENTSOE_eurostat")
plt.plot(np.sort(wind_model.predict(df_wind_system.values.reshape(-1,1))), label="1st_order_model")
# plt.plot(np.sort(wind_model_log.predict(log_transform(df_wind_system.values.reshape(-1,1)))), label="log_model")
plt.plot(np.sort(wind_model_pol(df_wind_system)), label="5th_order_model")
plt.plot(np.sort(dc_wind(df_wind_system)), label="DC_model")
plt.plot(np.sort(wind_model_sqrt.predict(X)), label="sqrt_model")
# plt.plot(np.sort(wind_model_svr.predict(df_wind_system.values.reshape(-1,1))), label="SVR_model")
plt.legend()
plt.savefig('duration_curves_wind_models.png')
plt.close()

print("Linear RMSE: ", root_mean_squared_error(df_wind, wind_model.predict(df_wind_system.values.reshape(-1,1))))
print("Log RMSE: ", root_mean_squared_error(df_wind, wind_model_log.predict(log_transform(df_wind_system.values.reshape(-1,1)))))
print("Poly(5) RMSE: ", root_mean_squared_error(df_wind, wind_model_pol(df_wind_system)))
print("Sqrt RMSE: ", root_mean_squared_error(df_wind, wind_model_sqrt.predict(X)))
print("SVR RMSE: ", root_mean_squared_error(df_wind, wind_model_svr.predict(df_wind_system.values.reshape(-1,1))))
print("DC RMSE: ", root_mean_squared_error(df_wind, dc_wind(df_wind_system)))

import os
cache_path = os.getcwd() + "/models/plant_models/wind.pkl"
cache_write(dc_wind, cache_path, verbose=True)
cache_path = os.getcwd() + "/models/plant_models/solar.pkl"
cache_write(dc_solar, cache_path, verbose=True)