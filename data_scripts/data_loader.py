import pandas as pd
import numpy as np
import os
import json
import requests
from common_scripts.utils import log_transform, delog_transform
from entsoe import EntsoePandasClient
from astral import LocationInfo
from astral.sun import sun
import matplotlib.pyplot as plt

# Token for ENTSO-E Transparency Platform: 134329c1-a120-4b33-8fa1-c96e9b46af59

def get_fossil_prices(hourly_index, price_indicator="c"):
    import pandas as pd
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

    # Get gas price without ETS price, as the model will have ETS price as a separate feature.
    # This is to avoid the model learning the relation between gas and ETS price, which is not causal and may change in the future.
    mmbtu_to_gj = 1.055056 # GJ/MMBtu
    gas_emission_factor = 0.0501 # tCO2/GJ
    df["gas_with_ets"] = df["gas"] + df["ets"] * gas_emission_factor * mmbtu_to_gj

    return df

class DataLoader:
    time_columns = ['is_weekend', 'is_winter', 'is_summer', 'is_spring', 'is_autumn', 'is_day']
    def __init__(self):
        self.data = None
        self.caps = None

    def _add_sun_times_to_df(self, df, city_name="Lisbon", country="Portugal", timezone="UTC", latitude=38.71667, longitude=-9.13333):
        """
        Adds sunrise and sunset times to a DataFrame with a DatetimeIndex.

        Parameters:
        - df: pandas DataFrame with a DatetimeIndex
        - city_name, country, timezone, latitude, longitude: location info

        Returns:
        - df: original DataFrame with 'sunrise' and 'sunset' columns added
        """
        location = LocationInfo(city_name, country, timezone, latitude, longitude)

        # Create sunrise/sunset columns
        sunrises = []
        sunsets = []

        for dt in df.index:
            s = sun(location.observer, date=dt.date(), tzinfo=dt.tzinfo)
            sunrises.append(s['sunrise'])
            sunsets.append(s['sunset'])

        df = df.assign(
            sunrise = sunrises,
            sunset = sunsets
        )
        return df

    def _specify_time_data(self, df):
        ix = df.index
        df = self._add_sun_times_to_df(df) # Get sunrise and sundown of the day for every timestamp.
        df = df.assign(
            hour_of_day = ix.hour,
            day_of_week = ix.day_of_week,
            is_weekend = [day >= 5 for day in ix.day_of_week],
            is_weekday = [day < 5 for day in ix.day_of_week],
            is_day = [(row.name > pd.Timestamp(row['sunrise'])) and (row.name < pd.Timestamp(row['sunset'])) for _, row in df.iterrows()],
            is_night = [(row.name <= pd.Timestamp(row['sunrise'])) or (row.name >= pd.Timestamp(row['sunset'])) for _, row in df.iterrows()],
            is_winter = [date.month in [12, 1, 2] for date in ix],
            is_spring = [date.month in [3, 4, 5] for date in ix],
            is_summer = [date.month in [6, 7, 8] for date in ix],
            is_autumn = [date.month in [9, 10, 11] for date in ix]
        )
        return df
    
    def _create_seasonal_features(self, df, prod_columns, drop_columns=None):
        df = self._specify_time_data(df)
        for p_col in prod_columns:
                for t_col in self.time_columns:
                    df.loc[df.index, str(p_col + '-' + t_col)] = df.loc[df.index, p_col].values * df.loc[df.index, t_col].values
        if drop_columns is not None:
            df = df.drop([drop_columns], axis=1)
        return df

class HistoricalData(DataLoader):
    # Define API endpoint and parameters
    URL = "https://api.energidataservice.dk/dataset/"
    ENTSOE_TOKEN = '134329c1-a120-4b33-8fa1-c96e9b46af59'

    def __init__(self,
                 start:pd.Timestamp,
                 end:pd.Timestamp,
                 priceArea:list     = [""],
                 limit:int          = 1000000,
                 country_code:str   = "PT",
                 server:str         = "ENTSOE",
                 load_data:bool     = True,
                 create_time_features:bool = True,
                 ):
        self.filepath = 'historical_data/clean_dataframes/' +'server-' + server + 'country-' + country_code + "_".join(priceArea) + str(start).split(' ')[0] + 'to' + str(end).split(' ')[0] + '.csv'
        self.country = country_code
        self.start, self.end = start, end
        self.server, self.limit, self.priceArea = server, limit, priceArea
        self.create_time_features = create_time_features
        if load_data:
            self.load_capacity_data()
            self.get_price_and_generation_data()
    
    def get_price_and_generation_data(self):
        # Load generation and price data
        if os.path.exists(self.filepath):
            self.data = pd.read_csv(self.filepath, index_col=0)
            self.data = self.data.set_index(pd.to_datetime(self.data.index, utc=True))
        else:
            if self.server == 'ENTSOE':
                self.data = self.get_data_from_entsoe()
                df_fossil = get_fossil_prices(self.data.index)
                self.data = pd.concat([self.data, df_fossil], axis=1)
            elif self.server == 'EnergiDataService':
                self.params = {
                    "start": str(self.start).split('+')[0],  # Start date/time in Danish time
                    "end": str(self.end).split('+')[0],    # End date/time (exclusive)
                    "filter": json.dumps({"PriceArea": self.priceArea}),  # Filter for DK1 region
                    "sort": "HourUTC asc",        # Optional: sort by time
                    "limit": self.limit,                 # Max records to retrieve
                }
                price_data = self._load_electricity_data()
                wind_data, solar_data = self.load_generation_data()
                self.data = pd.DataFrame(index=price_data.index, data = {price_data, wind_data, solar_data})
            else:
                raise(KeyError("Data server/source not known."))
            # self.data = self.data.drop(self.data.loc[self.data.isna().any(axis=1)].index)
            # self.data = self._fill_missing_generation_hours(self.data)
            self.data.to_csv(self.filepath, index=True)
        if self.create_time_features:
            self.data = self._create_seasonal_features(df=self.data, prod_columns=['wind', 'solar'])
            self.data['log_wind'] = log_transform(self.data['wind'])
            self.data['log_solar'] = log_transform(self.data['solar'])

    def load_capacity_data(self, filepath='historical_data/wind_solar_capacity_PT.csv'):
        # Taken from ENTSO-E Transparency Platform (does not match generation data):
        # directory = 'historical_data'
        # file = self.country + '_installed_capacities.csv'
        # filepath = directory + '/' + file
        # df = pd.read_csv(filepath)
        # solar_caps = df.loc[df['Production Type'] == 'Solar']
        # wind_caps = df.loc[df['Production Type'] == 'Wind Onshore']
        # s_c = solar_caps.values[0][1:]
        # w_c = wind_caps.values[0][1:]
        # years = [int(y.split(" ")[0]) for y in solar_caps.columns[1:]]
        # self.caps = pd.DataFrame(index=years,
        #                          data={'wind' : w_c.astype(float), 'solar' : s_c.astype(float)})
        df = pd.read_csv(filepath)
        df.index = [pd.Period(df['Year'].iloc[q].astype(str) + '-' + df["Month"].iloc[q].astype(str)) for q in range(len(df))]
        df.columns = ["Year", "Month", "wind", "solar"]
        self.caps = df[['wind', 'solar']]
        # df = np.transpose(pd.read_excel('historical_data/eurostat_capacities.xlsx', sheet_name='Wind', skiprows=9, skipfooter=3))
        # df.columns = df.iloc[0]
        # df = df.iloc[1:,1:]
        # df = df.set_index(df.index.astype(int))
        # years = df.index
        # w_c = df['Portugal'].values.astype(float)
        # df = np.transpose(pd.read_excel('historical_data/eurostat_capacities.xlsx', sheet_name='Solar', skiprows=9, skipfooter=3))
        # df.columns = df.iloc[0]
        # df = df.iloc[1:,1:]
        # df = df.set_index(df.index.astype(int))
        # s_c = df['Portugal'].values.astype(float)
        # self.caps = pd.DataFrame(index=years,
        #                          data={'wind' : w_c.astype(float), 'solar' : s_c.astype(float)})

    def get_data_from_entsoe(self):
        self.client = EntsoePandasClient(api_key=self.ENTSOE_TOKEN) # Object to query data through

        # Load Price Data
        df_prices   = self.client.query_day_ahead_prices(country_code=self.country,start=self.start,end=self.end+pd.Timedelta(23, 'h'))
        df_prices   = df_prices.loc[df_prices.index.minute==0]
        df_id       = self.client.query_imbalance_prices(country_code="PT",start=self.start,end=self.end+pd.Timedelta(24, 'h'))
        df_id       = df_id.loc[df_id.index.minute==0]

        # self.client.query_intraday_prices(country_code=self.country,start=self.start,end=self.end+pd.Timedelta(23, 'h'), sequence=1)
        
        # Load Capacity Data
        # df_capacity = self.client.query_installed_generation_capacity()
        # df_capacity = self.client.query_installed_generation_capacity(country_code=self.country,start=self.start,end=self.end+pd.Timedelta(24, 'h'))
        # df_capacity_per_unit = self.client.query_installed_generation_capacity_per_unit(country_code=self.country,start=self.start,end=self.end+pd.Timedelta(24, 'h'))
        # self.client.query_offered_capacity()

        # Load Load Data
        df_load = self.client.query_load_and_forecast(country_code=self.country,start=self.start,end=self.end+pd.Timedelta(24, 'h'))
        df_load = self._fill_missing_generation_hours(df_load)

        # Load Generation Data
        df_generation = self.client.query_generation(country_code=self.country, start=self.start, end=self.end+pd.Timedelta(24, 'h'))
        df_generation = df_generation.fillna(0)
        df_generation.index = pd.to_datetime(df_generation.index, utc=True)
        df_generation = self._fill_missing_generation_hours(df_generation)
        
        # Create Combined Dataframe
        df = pd.DataFrame(index = pd.to_datetime(df_prices.index, utc=True))
        df['price'] = df_prices
        df = self._fill_missing_price_hours(df)
        
        _sub_index  = 'Actual Aggregated'
        df.loc[df.index.isin(df_generation.index), 'solar'] = df_generation[('Solar',_sub_index)]
        df.loc[df.index.isin(df_generation.index), 'wind']  = df_generation[('Wind Onshore',_sub_index)] + (df_generation[('Wind Offshore',_sub_index)] if 'Wind Offshore' in df_generation.columns else 0)

        df[df_load.columns] = df_load.copy()
        df["Residual Load"] = df["Actual Load"] - df["solar"] - df["wind"]

        id_columns = [f"Imbalance {col}" for col in df_id.columns]
        df[id_columns] = df_id.copy()
        missing_imbalances = df.isna().max(axis=1)
        da_prices_corresponding = df.loc[missing_imbalances, 'price'].values
        df.loc[missing_imbalances, id_columns] = np.transpose([da_prices_corresponding]*2)
        
        return df

    def _get_response(self, url):
        # Make the request
        response = requests.get(url, params=self.params)
        data = response.json()
        # Convert to DataFrame
        df = pd.DataFrame(data.get("records", []))
        df['HourUTC'] = pd.to_datetime(df['HourUTC'])
        return df
    
    def _fill_missing_generation_hours(self, df):
        all_hours = df.index
        all_consecutive_hours = pd.date_range(start=df.index[0], end=df.index[-1], freq='h')
        missing_hours = sorted(set(all_consecutive_hours) - set(all_hours))
        for hour in missing_hours:
            copied_hour = hour - pd.Timedelta(1, 'day')
            row = df.loc[df.index == copied_hour]
            row.index = [hour]
            df = pd.concat([df.loc[df.index < hour], row, df.loc[df.index > hour]])
        return df
    
    def _fill_missing_price_hours(self, df):
        missing_hours = sorted(set(df.loc[df.isna().any(axis=1)].index))
        for hour in missing_hours:
            copied_hour = hour - pd.Timedelta(1, 'day')
            row = df.loc[df.index == copied_hour]
            row.index = [hour]
            df = pd.concat([df.loc[df.index < hour], row, df.loc[df.index > hour]])
        return df

    def _load_electricity_data(self):
        url = self.URL + "Elspotprices"
        df = self._get_response(url)
        df = df.set_index(pd.to_datetime(df.HourUTC, utc=True))
        columns = ['SpotPriceEUR']
        df = df[columns]
        columns[0] = ['price']
        df.columns = columns
        return df

    def load_generation_data(self):
        url = self.URL + "Forecasts_Hour"
        df = self._get_response(url) # Get data from API call
        df = df[['HourUTC','ForecastType','ForecastCurrent','ForecastDayAhead']]
        wind = df.loc[df['ForecastType'] == 'Offshore Wind']
        solar = df.loc[df['ForecastType'] == 'Solar']
        wind = self._fill_missing_generation_hours(wind) # Fill in hours of missing data
        solar = self._fill_missing_generation_hours(solar)
        return wind, solar

def historical_price_inspection(data_object : HistoricalData):
    # Constants
    h2_price_eur_per_mwh = 3 * 0.7 * (120 / 3.6)  # 3 €/kg example
    data = data_object.data.copy()

    """ Annual price duration curve: """
    fig, ax = plt.subplots(1, figsize=(10, 12))
    for year in data_object.data.index.year.unique():
        ax.plot(np.sort(data_object.data.loc[data_object.data.index.year==year, 'price'].values), label=f'Historical prices ({year})')
    ax.axhline(h2_price_eur_per_mwh, linestyle='dashed', label=r'Eq. H$_2$ value (3 €/kg as example)', color='red')
    ax.set_ylabel("€/MWh")
    ax.legend()
    plt.savefig('documentation/historical_annual_price_duration_curves.png')
    plt.close()

    """ Monthly price duration curves: """
    # Create subplots: 3 rows x 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharey=True)
    axes = axes.flatten()
    # fig.tight_layout(h_pad=5.0, w_pad=2.0)
    plt.tight_layout(pad=4.0, rect=[0.03, 0.03, 0.97, 0.95])
    # Loop through each month
    for i, month in enumerate(range(1, 13)):
        monthly_intersects = []
        for year in data_object.data.index.year.unique():
            monthly_data = data.loc[(data.index.month == month) & (data.index.year == year)]['price']
            sorted_prices = np.sort(monthly_data.values)
            ax = axes[i]
            ax.plot(sorted_prices, label=f'{year}')
        ax.axhline(h2_price_eur_per_mwh, linestyle='dashed', color='red', label=r'Eq. H$_2$ value')
        ax.set_title(f'Month {month}')
    ax.legend(loc='best')
    plt.savefig('documentation/historical_monthly_price_duration_curves.png')
    plt.close()

    """ Daily price duration curves: """
    # Filter for last 10 days
    jan_data = data.iloc[-24*10:]

    # Create subplots: 2 rows x 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharey=True)
    axes = axes.flatten()

    # Loop through each day
    for i, day in enumerate(jan_data.index.day.unique()):
        day_data = jan_data[jan_data.index.day == day]['price']
        sorted_prices = np.sort(day_data.values)  # Descending order

        ax = axes[i]
        ax.plot(sorted_prices)
        ax.axhline(h2_price_eur_per_mwh, linestyle='dashed', color='red', label=r'H$_2$ value')

        ax.set_xlim(0, len(sorted_prices)-1)
        ax.set_ylim(-100, 500)
        ax.set_title(f'{day_data.index[0].date()}')
        ax.legend()
    plt.savefig('documentation/historical_daily_price_duration_curves.png')
    plt.close()

#%% Main example
#     #%% Load historical data
#     start   = pd.Timestamp('20150101', tz='UTC')
#     end     = pd.Timestamp('20221231', tz='UTC')
#     data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')
#     historical_price_inspection(data_object)