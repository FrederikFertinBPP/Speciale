import pandas as pd
import numpy as np
import os
import json
import requests
from utils import cache_exists, cache_read, cache_write, log_transform, delog_transform
from entsoe import EntsoePandasClient
from astral import LocationInfo
from astral.sun import sun
import matplotlib.pyplot as plt

# Token for ENTSO-E Transparency Platform: ea0b03ee-267d-4a52-b1c6-15ed7d94b79a

class HistoricalData:
    # Define API endpoint and parameters
    URL = "https://api.energidataservice.dk/dataset/"
    ENTSOE_TOKEN = 'ea0b03ee-267d-4a52-b1c6-15ed7d94b79a'

    def __init__(self,
                 start      :pd.Timestamp,
                 end        :pd.Timestamp,
                 priceArea  :list           = [""],
                 limit      :int            = 1000000,
                 country_code    :str       = "PT",
                 server     :str            = "ENTSOE",
                 ):
        self.filepath = 'Historical Data/clean_dataframes/' +'server-' + server + 'country-' + country_code + "_".join(priceArea) + str(start).split(' ')[0] + 'to' + str(end).split(' ')[0] + '.csv'
        self.country = country_code
        self.start, self.end = start, end
        self.server, self.limit, self.priceArea = server, limit, priceArea
        self.time_columns = ['is_weekend', 'is_winter', 'is_summer', 'is_spring', 'is_autumn', 'is_day']
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
            # self.data = self._fill_missing_hours(self.data)
            self.data.to_csv(self.filepath, index=True)
        self.data = self._create_seasonal_features(df=self.data[['price', 'wind', 'solar']], prod_columns=['wind', 'solar'])
        self.data['log_wind'] = log_transform(self.data['wind'])
        self.data['log_solar'] = log_transform(self.data['solar'])

    def load_capacity_data(self):
        # Taken from ENTSO-E Transparency Platform (does not match generation data):
        # directory = 'Historical Data'
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
        df = np.transpose(pd.read_excel('Historical Data/eurostat_capacities.xlsx', sheet_name='Wind', skiprows=9, skipfooter=3))
        df.columns = df.iloc[0]
        df = df.iloc[1:,1:]
        df = df.set_index(df.index.astype(int))
        years = df.index
        w_c = df['Portugal'].values.astype(float)
        df = np.transpose(pd.read_excel('Historical Data/eurostat_capacities.xlsx', sheet_name='Solar', skiprows=9, skipfooter=3))
        df.columns = df.iloc[0]
        df = df.iloc[1:,1:]
        df = df.set_index(df.index.astype(int))
        s_c = df['Portugal'].values.astype(float)
        self.caps = pd.DataFrame(index=years,
                                 data={'wind' : w_c.astype(float), 'solar' : s_c.astype(float)})

    def get_data_from_entsoe(self):
        self.client = EntsoePandasClient(api_key=self.ENTSOE_TOKEN) # Object to query data through

        df_prices   = self.client.query_day_ahead_prices(country_code=self.country,start=self.start,end=self.end-pd.Timedelta(1, 'h'))
        df = pd.DataFrame(index = pd.to_datetime(df_prices.index, utc=True))
        df['price'] = df_prices

        df_generation = self.client.query_generation(country_code=self.country, start=self.start, end=self.end)
        df_generation = df_generation.fillna(0)
        df_generation.index = pd.to_datetime(df_generation.index, utc=True)
        df_generation = self._fill_missing_hours(df_generation)
        _sub_index  = 'Actual Aggregated'
        df.loc[df.index.isin(df_generation.index), 'solar'] = df_generation[('Solar',_sub_index)]
        df.loc[df.index.isin(df_generation.index), 'wind']  = df_generation[('Wind Onshore',_sub_index)] + (df_generation[('Wind Offshore',_sub_index)] if 'Wind Offshore' in df_generation.columns else 0)
        return df

    def _get_response(self, url):
        # Make the request
        response = requests.get(url, params=self.params)
        data = response.json()
        # Convert to DataFrame
        df = pd.DataFrame(data.get("records", []))
        df['HourUTC'] = pd.to_datetime(df['HourUTC'])
        return df
    
    def _fill_missing_hours(self, df):
        all_hours = df.index
        all_consecutive_hours = pd.date_range(start=df.index[0], end=df.index[-1], freq='h')
        missing_hours = sorted(set(all_consecutive_hours) - set(all_hours))
        for hour in missing_hours:
            copied_hour = hour - pd.Timedelta(1, 'day')
            row = df.loc[df.index == copied_hour]
            row.index = [hour]
            df = pd.concat([df.loc[df.index < hour], row, df.loc[df.index > hour]])
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
        wind = self._fill_missing_hours(wind) # Fill in hours of missing data
        solar = self._fill_missing_hours(solar)
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

if __name__ == "__main__":
    #%% Load historical data
    start   = pd.Timestamp('20150101', tz='UTC')
    end     = pd.Timestamp('20221231', tz='UTC')
    data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')
    historical_price_inspection(data_object)