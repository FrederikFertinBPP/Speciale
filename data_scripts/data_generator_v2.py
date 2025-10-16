#%% Initialization
from data_scripts.data_loader import HistoricalData
from common_scripts.utils import cache_exists, cache_read, cache_write, log_transform, delog_transform, laplace_rnd
from common_scripts.RFP_initialization import RenewableFuelPlant

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
import pmdarima as pm

import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from numba import jit
# from joblib import Parallel, delayed


#%% Common functions
def plot_acf(ts):
    if type(ts) == pd.DataFrame:
        ts_tag = ts.columns[0]
    elif type(ts) == pd.Series:
        ts_tag = ts.name
    else:
        raise("Could not resolve time series name", NameError)
    # Plot residuals with self and main regressors
    fig, axes = plt.subplots(1, 3, figsize=(15,8), sharey=True)
    axes = axes.flatten()
    fig.tight_layout(pad=4.0, rect=[0.03, 0.03, 0.97, 0.95])
    pm.plot_acf(ts, ax=axes[0], lags=48, show=False)
    pm.plot_pacf(ts, ax=axes[1], lags=48, show=False)
    axes[2] = pm.autocorr_plot(ts, show=False)
    plt.savefig(f'documentation/correlation_{ts_tag}.png')
    plt.close()

@jit(nopython=True)
def weighted_choice(p):
    cumsum = np.cumsum(p)
    r = np.random.random()
    for i in range(len(cumsum)):
        if r < cumsum[i]:
            return i
    return len(p) - 1  # fallback

@jit(nopython=True)
def simulate_stochastic_wind_process(horizon, ma_lag, observations, differences, domains,
                                 p5_obs, p98_obs, pos_dist, neg_dist,
                                 sigma_laplace, pol_model_pos, pol_model_neg, pol_model_mode):
    sim_cf = np.zeros(horizon + ma_lag)
    deltas = np.zeros(horizon + ma_lag)
    directions = np.zeros(horizon + ma_lag)

    sim_cf[:ma_lag] = observations
    deltas[:ma_lag] = differences
    directions[:ma_lag] = (differences > 0).astype(np.int32) - (differences <= 0).astype(np.int32)

    previous_diff_direction = directions[ma_lag - 1]

    for t in range(ma_lag, horizon + ma_lag):
        if (directions[t] == 0 or
            (sim_cf[t - 1] <= p5_obs and directions[t] == -1) or
            (sim_cf[t - 1] >= p98_obs and directions[t] == 1)):

            current_domain = max(0, np.sum(domains <= sim_cf[t - 1]) - 1) # np.argwhere(domains <= sim_cf[t - 1])[-1][0]
            p = neg_dist[:, current_domain] if previous_diff_direction == 1 else pos_dist[:, current_domain]
            interval_length = weighted_choice(p) # np.random.choice(np.arange(len(p)), p=p)
            directions[t:min(t + interval_length, horizon)] = -previous_diff_direction
            previous_diff_direction *= -1

        ma = np.mean(sim_cf[t - ma_lag:t])
        direction = directions[t - ma_lag]
        mean = pol_model_pos @ np.asarray([1,ma,ma**2]) if direction == 1 else pol_model_neg  @ np.asarray([1,ma,ma**2])
        # mode = pol_model_mode @ np.asarray([1,ma]) # unused
        delta = direction * np.random.exponential(max(sigma_laplace / 2, mean))
        delta = max(delta, 0) if direction == 1 else min(delta, 0)

        deltas[t] = delta
        sim_cf[t] = sim_cf[t - 1] + delta

    return sim_cf, deltas


#%% Classes
class SimulationTool:
    tool_type = ''
    ylabel    = ''
    _tag      = ''

    def __init__(self, forecaster):
        self.forecaster     = forecaster
        self.documentation  = self.forecaster.documentation
        self.cache_id       = self.forecaster.cache_id
        self.cache_replace  = self.forecaster.cache_replace
        self.verbose        = self.forecaster.verbose
        self.auto_arima     = self.forecaster.auto_arima
        self.arima_model    = None
        self.recent_data    = {}
    
    def plot_impact_of_deseason(self, data, model, residuals, labels=['Data', 'Model', 'Residuals'], name="xx"):
        fig, ax = plt.subplots(1, figsize=(12,10))
        ax.scatter(data.index, data,        color='red',    s=2, alpha=0.4, label=labels[0])
        ax.scatter(data.index, model,       color='blue',   s=2, alpha=0.4, label=labels[1])
        ax.scatter(data.index, residuals,   color='green',  s=2, alpha=0.4, label=labels[2])
        ax.set_ylabel(self.ylabel)
        ax.legend()
        plt.savefig(f'documentation/{self.forecaster.plot_dir}{name}.png')
        plt.close()
    
    def _fit_arima_model(self, time_series, order=(5,0,1), seasonal_order=(0, 0, 0, 0), name = ""):
        # Fit ARIMA model to the remaining stochastic process
        s_order = "" if sum(seasonal_order) == 0 else f"_s{seasonal_order}"
        cache_path = os.getcwd() + "/models/ts_models/" + self.tool_type + "/" + name + str(order) + s_order + str(self.cache_id) + ".pkl"
        if self.cache_id is not None and not(self.cache_replace) and cache_exists(cache_path):
            arima_model = cache_read(cache_path)
        else:
            if self.verbose: print(self.tool_type + name + ' model initialization'); t_start = time()
            if self.auto_arima:
                arima_model = pm.auto_arima(y=time_series, d=0, seasonal=True, m=24, error_action='ignore', suppress_warnings=True)
            else:
                arima_model = pm.ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True, )
            arima_model.fit(time_series)
            if self.verbose: print(self.tool_type + f' model fitted in {time() - t_start} seconds.')
            if self.cache_id is not None: cache_write(arima_model, cache_path, verbose=self.verbose)
        # self.last_state['endog_arima'] = time_series.iloc[-24:]
        return arima_model

    def _arima_simulate(self, horizon = 8760, realize = False):
        if self.arima_model is not None:
            sim = self.arima_model.arima_res_.simulate(nsimulations=horizon, anchor='end').values
            if realize:
                self.arima_model.arima_res_ = self.arima_model.arima_res_.append(sim) # Consider using extend if this is too slow.
            return sim
        else:
            raise("Cannot simulate arima process before fitting the model.")
    
    def _arima_forecast(self, horizon = 24):
        if self.arima_model is not None:
            forecast = self.arima_model.arima_res_.forecast(steps=horizon).values
            # forecast = self.arima_model.arima_res_.get_forecast(steps=horizon) # Should be used if we want confidence intervals as well.
            return forecast
        else:
            raise("Cannot forecast arima process before fitting the model.")

    def fit(self):
        pass

    def simulate(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def realize(self, start, end, refit = False, *args, **kwargs):
        """ Should simulate and then append existing time series model with new data. """
        pass


class PriceSimulationTool(SimulationTool):
    tool_type = 'price'
    ylabel    = '€/MWh'

    def __init__(self, forecaster):
        super().__init__(forecaster)
        self.log_prices = self.forecaster.log_prices
        self.price_tag  = self.forecaster.price_tag
        self._tag       = self.price_tag
        self.seasonal_price_regression = self.forecaster.seasonal_price_regression
        self.day_night_price_regression = self.forecaster.day_night_price_regression
        self.log_vre    = self.forecaster.log_vre
        self.y_train = self.forecaster.train_data[[self.price_tag]]
        self.y_test  = self.forecaster.test_data[[self.price_tag]]

    def WSS_price_regression(self, df): # Wind-Solar-Season price regression
        """ Create features """
        wind_tag = self.forecaster.wind_tag
        solar_tag = self.forecaster.solar_tag
        feature_tags = [wind_tag, solar_tag]
        if self.seasonal_price_regression: # Include solar and wind data as regressors that have varying weights dependent on season, weekend, and day/night.
            feature_tags = [wind_tag + '-is_summer', solar_tag + '-is_summer', wind_tag + '-is_winter', solar_tag + '-is_winter',
                             wind_tag + '-is_spring', solar_tag + '-is_spring', wind_tag + '-is_autumn', solar_tag + '-is_autumn',]
        elif self.day_night_price_regression: # Include solar and wind data as regressors that have varying weights dependent on day/night.
            feature_tags = [wind_tag + '-is_day', solar_tag + '-is_day', wind_tag + '-is_night', solar_tag + '-is_night']
        if self.log_vre:
            feature_tags += ['log_' + wind_tag, 'log_' + solar_tag]
        self.X_train = self.train_data[feature_tags]
        self.X_test  = self.forecaster.test_data[feature_tags]
        self.feature_tags = feature_tags

        """ Fit linear model "price(t) = w1 * wind(t) + w2 * solar(t) + ... + eps(t)" """
        wss_model = LinearRegression()
        # wss_model = SVR()
        wss_model.fit(self.X_train, df)
        merit_order_effect = wss_model.predict(self.X_train)
        residuals = df - merit_order_effect
        if self.verbose: print("Mean squared error of WSS fit (train): ", mean_squared_error(df, merit_order_effect))
        if self.verbose: print("Mean squared error of WSS fit (validation): ", mean_squared_error(self.y_test, wss_model.predict(self.X_test)))
        if self.documentation:
            self.plot_impact_of_deseason(df,
                                        merit_order_effect - wss_model.intercept_,
                                        residuals + wss_model.intercept_, name="removing_merit_order_effect",
                                        labels=['Historical Prices', 'Merit Order Effect', 'Residuals'])
            self.plot_impact_of_deseason(df,
                                        merit_order_effect,
                                        residuals, name="removing_merit_order_effect_and_bias",
                                        labels=['Historical Prices', 'Model', 'Residuals'])
        return residuals, wss_model

    def _del_trend(self, df):
        trend_model = LinearRegression()
        hours       = pd.DataFrame({'timestamp': [h.timestamp() - self.forecaster.t_zero for h in self.train_data.index]}, index=self.train_data.index) # Convert to Unix time
        weights     = [0.9999**x for x in range(len(hours))][::-1] # Weighted linear regression
        trend_model.fit(hours, df, sample_weight=weights)
        impact      = trend_model.predict(hours)
        residuals   = df - impact
        if self.documentation: self.plot_impact_of_deseason(df, impact, residuals, name=self.tool_type + "removing_trend_effect")
        return residuals, trend_model

    def _del_annual_cycle(self, df):
        month_index     = df.index.month
        avg_months      = df.groupby(month_index).mean()
        monthly_avgs    = month_index.map(avg_months[df.columns[0]])
        impact          = monthly_avgs.values.reshape(len(df),1)
        residuals       = df - impact
        if self.documentation: self.plot_impact_of_deseason(df, impact, residuals, name=self.tool_type + "removing_seasonal_effect")
        return residuals, avg_months
    
    def _del_weekday_and_weekend_pattern(self, df):
        time_info = self.train_data
        weekend_data = df.loc[time_info.is_weekend, self.price_tag]
        weekday_data = df.loc[time_info.is_weekday, self.price_tag]
        avg_weekend_hours = weekend_data.groupby(weekend_data.index.hour).mean()
        avg_weekday_hours = weekday_data.groupby(weekday_data.index.hour).mean()
        weekend_impact    = weekend_data.index.hour.map(avg_weekend_hours)
        weekday_impact    = weekday_data.index.hour.map(avg_weekday_hours)
        df.loc[time_info.is_weekend, self.price_tag] -= weekend_impact
        df.loc[time_info.is_weekday, self.price_tag] -= weekday_impact
        return df, avg_weekday_hours, avg_weekend_hours

    def fit(self):
        self.train_data = self.forecaster.train_data.copy() # Historical price data
        residuals = self.train_data[[self.price_tag]]

        residuals, self.wss_model = self.WSS_price_regression(residuals)

        residuals, self.trend_model = self._del_trend(residuals)
        
        self.max_historical, self.min_historical = np.max(residuals), np.min(residuals)

        residuals, self.monthly_avg = self._del_annual_cycle(residuals)
        residuals, self.weekday_avg, self.weekend_avg = self._del_weekday_and_weekend_pattern(residuals)
        
        residuals                 = self._calculate_rs_probabilities(residuals)
        self.arima_model          = self._fit_arima_model(residuals, order=(2,0,1), seasonal_order=(1, 0, 0, 24))    
        if self.documentation:
            self.investigate_heteroskedasticity(residuals)
            plot_acf(residuals)
        
    def investigate_heteroskedasticity(self, residuals):
        # Even though we do find that the data is heteroskedastic, log-transforming does not change this at all. So it is ignored.
        from statsmodels.stats.diagnostic import het_breuschpagan
        self.bp_test = het_breuschpagan(residuals, sm.add_constant(self.train_data[[self.forecaster.wind_tag, self.forecaster.solar_tag]]))
        self.heteroskedastic = self.bp_test[3] < 0.05 # Reject the null-hypothesis of homoskedasticity at a 5% significane threshold.
        # Plot residuals with self and main regressors
        fig, axes = plt.subplots(1, 3, figsize=(15,8), sharey=True)
        axes = axes.flatten()
        fig.tight_layout(pad=4.0, rect=[0.03, 0.03, 0.97, 0.95])
        ylabel = 'Residuals' + ('' if self.log_prices else ' [€/MWh]')
        axes[0].set_ylabel(ylabel)
        axes[0].scatter(self.train_data.index,    residuals, s=1)
        axes[0].set_xticks(axes[0].get_xticks(), labels=axes[0].get_xticklabels(), rotation=45)
        axes[1].scatter(self.train_data[self.forecaster.wind_tag],  residuals, s=1)
        axes[2].scatter(self.train_data[self.forecaster.solar_tag], residuals, s=1)
        plt.savefig(f'documentation/{self.forecaster.plot_dir}heteroskedastic_visual.png')
        plt.close()

    def _calculate_rs_probabilities(self, prices):
        ### We define extreme observations as observations outside 2 standard deviations of the mean (of 0)
        self.n_regimes = 3
        residual_mu = np.mean(prices)
        residual_std = np.std(prices.values)
        high_price_regime = (prices > 2 * residual_std + residual_mu).astype(int)
        low_price_regime = (prices < -2 * residual_std + residual_mu).astype(int)
        regimes = high_price_regime - low_price_regime + 1 # Regimes 0: Low, 1: Normal, 2: High

        # Build transition matrix:
        self.rs_prob_matrix = np.zeros((self.n_regimes, self.n_regimes)) # row is from regime, columns are to regimes.
        from_regime = 1
        for to_regime in regimes.values:
            self.rs_prob_matrix[from_regime,to_regime] += 1
            from_regime = to_regime
        self.rs_prob_matrix *= 1/len(regimes)
        self.price_regime_probabilities = self.rs_prob_matrix.sum(axis=0) # Probability of being in each regime.
        self.rs_prob_matrix = np.transpose(np.transpose(self.rs_prob_matrix) / self.rs_prob_matrix.sum(axis=1)) # Probabilities in each row sum to 1.

        ### Save info on high and low regimes before removing it from the residual price dataframe.
        high_prices    = prices.iloc[:,0][high_price_regime.astype(bool).iloc[:,0]]
        low_prices     = prices.iloc[:,0][low_price_regime.astype(bool).iloc[:,0]]
        normal_prices  = prices.iloc[:,0][((1-low_price_regime).astype(bool).iloc[:,0] & (1-high_price_regime).astype(bool).iloc[:,0])]
        if self.documentation:
            bin_width = 10
            _bins = lambda d : np.arange(min(d), max(d) + bin_width, bin_width)
            fig, ax = plt.subplots(1, figsize=(12,8))
            ax.hist(high_prices, label='High Regime', color='darkblue', bins=_bins(high_prices))
            ax.hist(low_prices, label='Low Regime', color='lightblue', bins=_bins(low_prices))
            ax.hist(normal_prices, label='Standard Regime', color='blue', bins=_bins(normal_prices))
            ax.set_xlabel('€/MWh')
            ax.legend()
            ax.set_title('Price regimes of residuals')
            ax.set_xlim(-3 * residual_std + residual_mu, 3 * residual_std + residual_mu)
            ax.set_xlim(np.min(prices), np.max(prices))
            plt.savefig(f'documentation/{self.forecaster.plot_dir}regime_histogram.png')
            plt.close()

        self.high_prices = high_prices # The outliers are not normal distributed at all. Sampled from a one-sided laplace distribution later
        self.high_base   = 2 * residual_std + residual_mu
        self.high_std    = sum(abs(self.high_prices - self.high_base)) / len(high_prices)
        self.low_prices  = low_prices # The outliers are not normal distributed at all. Sampled from a one-sided laplace distribution later
        self.low_base    = -2 * residual_std + residual_mu
        self.low_std     = sum(abs(self.low_prices - self.low_base)) / len(low_prices)

        # Remove extreme residuals from time series and replace with mean
        residuals = prices * (1-high_price_regime)*(1-low_price_regime) + residual_mu * ((high_price_regime) | (low_price_regime))
        self.recent_data['regime'] = regimes[self.price_tag].values[-1]
        return residuals

    def _simulate_price_regimes(self, horizon=8760, realize = False):
        # We simulate from latest observed price regime
        realized_price_regimes  = [np.random.choice(self.n_regimes, p=self.rs_prob_matrix[self.recent_data['regime']])]
        def _sample_extreme_price(r):
            if r == 1: # Normal price regime
                return 0
            elif r == 0: # Low price regime
                return laplace_rnd(self.low_base, self.low_std, np.random.uniform(-0.5,0))
            elif r == 2: # High price regime 
                return laplace_rnd(self.high_base, self.high_std, np.random.uniform(0,0.5))
        for h in range(1,horizon):
            realized_price_regimes.append(np.random.choice(self.n_regimes, p=self.rs_prob_matrix[realized_price_regimes[h-1]]))
        extreme_prices = [_sample_extreme_price(r) for r in realized_price_regimes]
        normal_regime_list = np.asarray(realized_price_regimes) == 1
        # normal_regime_list is True when in normal regime, False when in extreme regime.
        if realize:
            self.recent_data['regime'] = realized_price_regimes[-1]
        return extreme_prices, normal_regime_list

    def simulate(self, vre_profiles:pd.DataFrame):
        # Simulate spot market electricity prices for the entire horizon
        df = vre_profiles.copy()
        time_info = self.forecaster.database._specify_time_data(pd.DataFrame(index=pd.to_datetime(df.index, utc=True)))
        return self._simulate(df, time_info)

    def forecast(self, vre_profiles:pd.DataFrame):
        df = vre_profiles.copy()
        time_info = self.forecaster.database._specify_time_data(pd.DataFrame(index=pd.to_datetime(df.index, utc=True)))
        return self._simulate(df, time_info, forecasting=True)

    def realize(self, vre_profiles:pd.DataFrame):
        df = vre_profiles.copy()
        time_info = self.forecaster.database._specify_time_data(pd.DataFrame(index=pd.to_datetime(df.index, utc=True)))
        return self._simulate(df, time_info, realize=True)

    def _simulate(self, df, time_info, realize = False, forecasting=False):
        df = df.copy()
        horizon = len(df)
        # Simulate stochastic process:
        if forecasting:
            df[self.price_tag] = self._arima_forecast(horizon=horizon)
        else:
            df['stoch_price_residuals'] = self._arima_simulate(horizon=horizon, realize=realize)
            # Simulate price regime transitions following Markov Transition Process:
            df['extreme_price_impact'], normal_regime_array = self._simulate_price_regimes(horizon=horizon, realize=realize)
            # Obtain initial residual prices
            df[self.price_tag] = df['stoch_price_residuals'] * normal_regime_array + df['extreme_price_impact']

        # Add daily patterns:
        df.loc[time_info.is_weekend, self.price_tag] += df.loc[time_info.is_weekend].index.hour.map(self.weekend_avg)
        df.loc[time_info.is_weekday, self.price_tag] += df.loc[time_info.is_weekday].index.hour.map(self.weekday_avg)
        df[self.price_tag] += df.index.month.map(self.monthly_avg[self.price_tag])

        df[self.price_tag] = np.clip(df[self.price_tag], self.min_historical, self.max_historical)
        
        # Add trend effect on prices:
        u_hours = pd.DataFrame(index=df.index, data={'timestamp': [h.timestamp() - self.forecaster.t_zero for h in df.index]}) # Convert to Unix time
        df[self.price_tag] += self.trend_model.predict(u_hours)[:,0]

        _tags = [self.forecaster.wind_tag, self.forecaster.solar_tag]
        if self.seasonal_price_regression or self.day_night_price_regression: # Include solar and wind data as regressors that have varying weights dependent on season, weekend, and day/night.
            X = self.forecaster.database._create_seasonal_features(df.copy(), prod_columns = _tags)
        else:
            X = df[_tags].copy()
        if self.forecaster.log_vre:
            X['log_' + self.forecaster.wind_tag] = log_transform(df.loc[:, self.forecaster.wind_tag])
            X['log_' + self.forecaster.solar_tag] = log_transform(df.loc[:, self.forecaster.solar_tag])
        X = X[self.feature_tags]

        # Add merit order effect on prices:
        df[self.price_tag] += self.wss_model.predict(X)[:,0]

        return df[[self.price_tag]]


class RenewablesSimulationTool(SimulationTool):
    tool_type = 'renewable'
    ylabel    = 'MW'

    def __init__(self, forecaster, caps, vre_tag, weather_years = True):
        super().__init__(forecaster)
        self.caps = caps
        self.vre_tag = vre_tag
        self.generate_weather_years = weather_years
    
    def _del_capacity_trend(self, df_):
        df = df_.copy()
        year_index          = df.index.year
        yearly_caps         = year_index.map(self.caps[self.vre_tag])
        df.loc[:,self.vre_tag]    = df[self.vre_tag] / yearly_caps
        if df[self.vre_tag].max() > 1.0:
            print(f"Warning: {self.vre_tag} capacity factor larger than 1.0 detected in training data.")
            df = df_.copy() # Revert to original data
            yearly_max = df.groupby(year_index).max()
            yearly_caps = year_index.map(yearly_max[self.vre_tag])
            df.loc[:,self.vre_tag]    = df[self.vre_tag] / yearly_caps
        return df

    def _simulate_cf(self, hourly_index:pd.DatetimeIndex, realize = False, *args):
        pass

    def realize(self, hourly_index: pd.DatetimeIndex):
        """ Realize a solar capacity factor time series between start and end timestamps, based on the fitted ARIMA model.
        Update the ARIMA model with the realized values, so that future simulations are conditioned on these values. """
        sim_cf = self._simulate_cf(hourly_index, realize=True)
        return sim_cf


class SolarSimulationTool(RenewablesSimulationTool):
    tool_type = 'solar'

    def fit(self):
        data = self.forecaster.train_data[[self.vre_tag]].copy()
        # Step 1: Remove effect from added capacities of each year.
        df = self._del_capacity_trend(data) # Produces capacity utilization factor for all timestamps

        # Prep 1: Determine average hourly mean profile and max mean hourly value for each month.
        self.hourly_monthly_mean_profiles = df.groupby([df.index.month, df.index.hour]).mean()
        self.hourly_monthly_max_profiles = df.groupby([df.index.month, df.index.hour]).max()
        self.hourly_monthly_min_profiles = df.groupby([df.index.month, df.index.hour]).min()
        self.hourly_monthly_std_profiles = df.groupby([df.index.month, df.index.hour]).std()

        # Prep 2: Determine average hourly std and average mean hourly std for each month.
        self.monthly_mean_max = self.hourly_monthly_mean_profiles.groupby(level=0).max()
        self.monthly_std_means = self.hourly_monthly_std_profiles.groupby(level=0).mean()

        if self.documentation:
            fig, axs = plt.subplots(4,3, figsize=(15,10), sharex=True, sharey=True)
            axs = axs.flatten()
            for month in range(1,13):
                ax = axs[month-1]
                for day in range(1,28):
                    month_mean_values = self.hourly_monthly_mean_profiles.loc[(month), self.vre_tag].values
                    actual_values = df.loc[(df.index.day == day) & (df.index.month == month), self.vre_tag].values[:24]
                    ax.scatter(np.arange(1,25), actual_values - month_mean_values, s=1)
            plt.savefig(f'documentation/{self.forecaster.plot_dir}monthly_variation_from_daily_mean.png')
            plt.close()

        # Step 2: Establish a time series of daily maximum values
        daily_max       = df.groupby(df.index.date).max()
        daily_max.index = pd.to_datetime(daily_max.index)
        daily_month_ix  = daily_max.index.month
        self.historical_monthly_max = daily_max.groupby(daily_month_ix).max()
        
        # Step 3: Divide daily max by max mean hourly value for each month.
        impact = daily_month_ix.map(self.monthly_mean_max[self.vre_tag])
        daily_max[self.vre_tag] = daily_max[self.vre_tag] / impact # Now a distribution centered around 1.
        self.mu_daily_max = daily_max.mean().values[0]
        daily_max[self.vre_tag] -= self.mu_daily_max # Make max process centered around 0.

        # Prep 3: Obtain standard deviation of max values for each month.
        self.monthly_std_of_max = daily_max.groupby(daily_month_ix).std()

        # Step 4: Normalize variation for each month. Winter months have a lot larger variation in daily peaks.
        daily_std = daily_month_ix.map(self.monthly_std_of_max[self.vre_tag])
        daily_max[self.vre_tag] = daily_max[self.vre_tag] / daily_std

        self.residuals = daily_max

        if self.documentation: plot_acf(self.residuals)
        
        # Step 2: Subtract daily profiles
        hour_residuals = df.copy()
        for month in hour_residuals.index.month.unique():
            daily_mean_profile = self.hourly_monthly_mean_profiles.loc[month, self.vre_tag] # Daily profiles
            daily_std_profile = self.hourly_monthly_std_profiles.loc[month, self.vre_tag] # Daily profiles
            hour_residuals.loc[hour_residuals.index.month==month, self.vre_tag] -= hour_residuals.loc[hour_residuals.index.month==month].index.hour.map(daily_mean_profile)
            hour_residuals.loc[(hour_residuals.index.month==month) & (self.forecaster.train_data.is_day), self.vre_tag] /= hour_residuals.loc[(hour_residuals.index.month==month) & (self.forecaster.train_data.is_day)].index.hour.map(daily_std_profile)
        
        self.hourly_arima_model = self._fit_arima_model(hour_residuals, order=(1,0,1), seasonal_order=(1,0,0,24), name="hour")

        # Step 5: Fit arima model to daily max residual data.
        self.arima_model = self._fit_arima_model(self.residuals, order=(2,0,0))

    def simulate(self, capacity, year=2023):
        # Create hourly index for leap year
        hourly_index = pd.to_datetime(pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h'), utc=True)
        day_index    = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='d')
        n_hours = len(hourly_index)
        n_days  = int(n_hours/24)

        # Reverse Step 5: Simulate daily maximum values of solar production.
        sim_daily_max = pd.DataFrame(index=day_index, data={self.vre_tag : self._arima_simulate(n_days).values})

        # Rev. Step 4: Multiply simulated values with monthly std of maximums.
        impact = sim_daily_max.index.month.map(self.monthly_std_of_max[self.vre_tag])
        sim_daily_max[self.vre_tag] = sim_daily_max[self.vre_tag] * impact

        # Rev. Step 3: Multiply daily max by max mean hourly value for each month.
        impact = sim_daily_max.index.month.map(self.monthly_mean_max[self.vre_tag])
        sim_daily_max[self.vre_tag] += self.mu_daily_max

        # Rev. Step 2: Go from daily max to hourly profiles
        hourly_variation = self.hourly_arima_model.arima_res_.simulate(nsimulations=n_hours).values
        profile = pd.DataFrame(index=hourly_index, data={self.vre_tag : hourly_variation})
        for month in range(1,13):
            hourly_means = self.hourly_monthly_mean_profiles.loc[month]
            hourly_stds  = self.hourly_monthly_std_profiles.loc[month]
            hourly_min  = self.hourly_monthly_min_profiles.loc[month]
            hourly_max  = self.hourly_monthly_max_profiles.loc[month]
            ix = profile.index.month == month
            hours_of_month = profile.index[ix]
            day_profile = [np.clip(hourly_means.loc[timestamp.hour,self.vre_tag] * sim_daily_max.loc[pd.to_datetime(timestamp.date()),self.vre_tag]
                                   + profile.loc[timestamp, self.vre_tag] * hourly_stds.loc[timestamp.hour,self.vre_tag],
                                  hourly_min.loc[timestamp.hour,self.vre_tag], hourly_max.loc[timestamp.hour,self.vre_tag]) for timestamp in hours_of_month]
            profile.loc[ix, self.vre_tag] = day_profile
        
        if self.documentation:
            data = self.forecaster.train_data[[self.vre_tag]]
            df_hist = self._del_capacity_trend(data)
            plt.plot(np.sort(profile[self.vre_tag]), color='red')
            for year in df_hist.index.year.unique():
                plt.plot(np.sort(df_hist.loc[df_hist.index.year==year, self.vre_tag]), color='blue')
            plt.savefig(f'documentation/{self.forecaster.plot_dir}solar_load_duration_curves.png')
            plt.close()

        profile[self.vre_tag] *= capacity

        return profile

    def _simulate_cf(self, hourly_index: pd.DatetimeIndex, realize=False, forecasting = False):
        day_index    = pd.date_range(hourly_index[0], hourly_index[-1], freq='d')

        # Reverse Step 5: Simulate daily maximum values of solar production.
        if forecasting:
            sim_daily_max = pd.DataFrame(index=day_index, data={self.vre_tag : self._arima_forecast(len(day_index))})
        else:
            sim_daily_max = pd.DataFrame(index=day_index, data={self.vre_tag : self._arima_simulate(len(day_index), realize=realize)})

        # Rev. Step 4: Multiply simulated values with monthly std of maximums.
        impact = sim_daily_max.index.month.map(self.monthly_std_of_max[self.vre_tag])
        sim_daily_max[self.vre_tag] = sim_daily_max[self.vre_tag] * impact

        # Rev. Step 3: Multiply daily max by max mean hourly value for each month.
        impact = sim_daily_max.index.month.map(self.monthly_mean_max[self.vre_tag])
        sim_daily_max[self.vre_tag] += self.mu_daily_max

        # Rev. Step 2: Go from daily max to hourly profiles
        n_hours = len(hourly_index)
        if forecasting:
            sim_hourly_variation = self.hourly_arima_model.arima_res_.forecast(steps=n_hours).values
        else:
            sim_hourly_variation = self.hourly_arima_model.arima_res_.simulate(nsimulations=n_hours).values
            if realize:
                self.hourly_arima_model.arima_res_ = self.hourly_arima_model.arima_res_.append(sim_hourly_variation)

        profile = pd.DataFrame(index=hourly_index, data={self.vre_tag : sim_hourly_variation})
        for month in profile.index.month.unique():
            # Apply monthly distinct daily profiles to the simulated daily maximum values. 
            hourly_means = self.hourly_monthly_mean_profiles.loc[month]
            hourly_stds  = self.hourly_monthly_std_profiles.loc[month]
            hourly_min  = self.hourly_monthly_min_profiles.loc[month]
            hourly_max  = self.hourly_monthly_max_profiles.loc[month]
            ix = profile.index.month == month
            hours_of_month = profile.index[ix]
            day_profile = [np.clip(hourly_means.loc[timestamp.hour,self.vre_tag] * sim_daily_max.loc[pd.to_datetime(timestamp.date(), utc=True),self.vre_tag]
                                   + profile.loc[timestamp, self.vre_tag] * hourly_stds.loc[timestamp.hour,self.vre_tag],
                                  hourly_min.loc[timestamp.hour,self.vre_tag], hourly_max.loc[timestamp.hour,self.vre_tag]) for timestamp in hours_of_month]
            profile.loc[ix, self.vre_tag] = day_profile
        
        if self.documentation:
            data = self.forecaster.train_data[[self.vre_tag]]
            df_hist = self._del_capacity_trend(data)
            plt.plot(np.sort(profile[self.vre_tag]), color='red')
            for year in df_hist.index.year.unique():
                plt.plot(np.sort(df_hist.loc[df_hist.index.year==year, self.vre_tag]), color='blue')
            plt.savefig(f'documentation/{self.forecaster.plot_dir}solar_load_duration_curves.png')
            plt.close()
        return profile

    def forecast(self, hourly_index: pd.DatetimeIndex):
        return self._simulate_cf(hourly_index=hourly_index, forecasting=True)


class WindSimulationTool(RenewablesSimulationTool):
    tool_type = 'wind'

    def solar_wind_regression(self, wind, solar):
        model = LinearRegression()
        # wss_model = SVR()
        model.fit(X=solar, y=wind)
        solar_effect = model.predict(solar)
        residuals = wind - solar_effect
        if self.verbose: print("Mean squared error of WSS fit (train): ", mean_squared_error(wind, solar_effect))
        # if self.verbose: print("Mean squared error of WSS fit (validation): ", mean_squared_error(self.forecaster.test_data['wind'], model.predict(self.forecaster.test_data['solar'])))
        # if self.documentation:
        #     self.plot_impact_of_deseason(df,
        #                                 merit_order_effect - self.wss_model.intercept_,
        #                                 residuals + self.wss_model.intercept_, name="removing_merit_order_effect",
        #                                 labels=['Historical Prices', 'Merit Order Effect', 'Residuals'])
        #     self.plot_impact_of_deseason(df,
        #                                 merit_order_effect,
        #                                 residuals, name="removing_merit_order_effect_and_bias",
        #                                 labels=['Historical Prices', 'Model', 'Residuals'])
        return residuals, model

    def _deseasonalise(self, df):
        # Prep 1: Identify seasonal effect (monthly quantiles and means)
        month_year_ix = df.index.tz_localize(None).to_period('M')
        monthly_groups = df.groupby(month_year_ix)
        monthly_p90s = monthly_groups.quantile(0.9)
        monthly_p10s = monthly_groups.quantile(0.1)
        monthly_means= monthly_groups.mean()
        month_ix     = monthly_means.index
        annual_p90s  = monthly_p90s.groupby(month_ix.year).mean()
        annual_p10s  = monthly_p10s.groupby(month_ix.year).mean()

        # Prep 2: Define stretch and move factors for each month in the historical data to deseasonalise data.
        self.monthly_stretch_factors = month_ix.year.map((annual_p90s - annual_p10s)[self.vre_tag]) / (monthly_p90s - monthly_p10s)[self.vre_tag]
        self.monthly_move_factors    = - monthly_p10s[self.vre_tag] * self.monthly_stretch_factors.values + month_ix.year.map(annual_p10s[self.vre_tag])
        self.weather_years           = self.monthly_move_factors.index.year.unique()
        
        # Step 1: Deseasonalise capacity factors:
        deseasoned_cf = df[self.vre_tag] * month_year_ix.map(self.monthly_stretch_factors) + month_year_ix.map(self.monthly_move_factors)
        
        # Prep 3: Define average daily profile:
        daily_profiles = deseasoned_cf.groupby([deseasoned_cf.index.year, deseasoned_cf.index.hour]).mean()
        mean_cfs = deseasoned_cf.groupby(deseasoned_cf.index.year).mean()
        # Step 2: Subtract daily profile
        residuals = pd.DataFrame(index=deseasoned_cf.index, data={self.vre_tag : deseasoned_cf})
        for year in df.index.year.unique():
            dp = daily_profiles.loc[year]
            mean_cf = mean_cfs.loc[year]
            residuals.loc[residuals.index.year==year, self.vre_tag] -= residuals.loc[df.index.year==year].index.hour.map(dp)

        # Variables, which are used to seasonalise based on average seasons:
        self.daily_profile        = daily_profiles.groupby(level=1).mean()
        self.avg_monthly_stretch  = self.monthly_stretch_factors.groupby(self.monthly_stretch_factors.index.month).mean()
        self.avg_monthly_move     = self.monthly_move_factors.groupby(self.monthly_move_factors.index.month).mean()
        self.std_monthly_stretch  = self.monthly_stretch_factors.groupby(self.monthly_stretch_factors.index.month).std()
        self.std_monthly_move     = self.monthly_move_factors.groupby(self.monthly_move_factors.index.month).std()
        self.avg_monthly_p90s     = monthly_p90s.groupby(month_ix.month).mean()
        self.avg_monthly_p10s     = monthly_p10s.groupby(month_ix.month).mean()
        
        return residuals

    def fit(self):
        # Step 1: Remove effect from added capacities of each year.
        data = self.forecaster.train_data[[self.vre_tag]].copy()
        capacity_factors = self._del_capacity_trend(data)
        data_solar = self.forecaster.train_data[['solar']].copy()
        capacity_factors_solar = self.forecaster.solar_model._del_capacity_trend(data_solar)
        self.min_historical_production = np.min(capacity_factors)
        self.max_historical_production = np.max(capacity_factors)

        # Remove effect of solar on wind profile
        residuals, self.solar_reg_model = self.solar_wind_regression(capacity_factors, capacity_factors_solar)

        df = self._deseasonalise(residuals)

        # Step X:
        if self.documentation: plot_acf(df)

        prev_df = df.shift(1).fillna(df.iloc[0][self.vre_tag])
        diff = df - prev_df
        self.mu_laplace = np.median(diff) # Laplace distribution parameter
        self.sigma_laplace = sum(abs(diff[self.vre_tag] - self.mu_laplace)) / len(diff) # Laplace distribution parameter

        max_lag = 168 # Maximum lag considered is a full week.
        self.ma_lag = self._get_significant_ma_lag(df, max_lag) # 11 is the most significant lag (ofc dependent on historical data).

        # Polyfit of 2nd order for positive differences, 1st order for negative and mode differences:
        self.pol_model_pos, self.pol_model_neg, self.pol_model_mode = self._get_exponential_models(df, self.ma_lag)

        self.pos_int_length_distributions, self.neg_int_length_distributions = self._calculate_interval_probabilities(df)

        self.p5_deseason_observation = np.quantile(df, 0.05)
        self.p98_deseason_observation = np.quantile(df, 0.98)

        self.recent_data['observations']= df.iloc[-self.ma_lag:][self.vre_tag].values
        self.recent_data['differences'] = diff.iloc[-self.ma_lag:][self.vre_tag].values
        
    def _get_significant_ma_lag(self, df, maxlag):
        """
        Identifies the lag for which the dependency between the moving average
        (level) of the capacity utilisation and the following rate of change
        is most significant.

        Parameters:
        -----------
        CapacityUtilisation : array-like
            Hourly values of observed capacity utilisation of total installed wind energy feed-in capacity.
        maxlag : int
            Maximum moving average lag to test.

        Returns:
        --------
        MovavgLag : int
            Optimal moving average lag with the most significant dependency.
        """
        prev_df = df.shift(1).fillna(df.iloc[0][self.vre_tag])
        diff = df - prev_df

        r_list = np.zeros(maxlag)

        for lag in range(1, maxlag + 1):
            # Compute simple moving average
            ma = df.rolling(window=lag).mean()

            # Align x and y for correlation
            x = ma[lag:]
            y = diff[lag:]

            if len(x) > 1:
                r = np.corrcoef(x[self.vre_tag].values, y[self.vre_tag].values)[0, 1]
                r_list[lag - 1] = r
            else:
                r_list[lag - 1] = 0  # Not enough data to compute correlation

        if self.documentation:
            plt.plot(r_list)
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.title("Correlation between moving average and first order difference")
            plt.savefig(f'documentation/{self.forecaster.plot_dir}wind_corr_ma_n_diff.png')
            plt.close()
        # Find lag with maximum absolute correlation
        MovavgLag = np.argmax(np.abs(r_list)) + 1
        return MovavgLag

    def _get_exponential_models(self, df, MovavgLag):
        """
        Estimate positive/negative mean deviations and mode of capacity utilisation changes
        grouped by utilisation level intervals.

        Parameters:
        -----------
        CapacityUtilisation : array-like
            Hourly capacity utilisation values.
        MovavgLag : int
            Optimal lag for moving average.

        Returns:
        --------
        pfParaPosMean : ndarray
            Coefficients of 2nd-degree polynomial fit for positive mean deviations.
        pfParaNegMean : ndarray
            Coefficients of 2nd-degree polynomial fit for negative mean deviations.
        pfParaMode : ndarray
            Coefficients of 1st-degree polynomial fit for mode values.
        """
        intervals = np.arange(min(df[self.vre_tag]), max(df[self.vre_tag]), (max(df[self.vre_tag])-min(df[self.vre_tag]))/20)
        centres = np.zeros(len(intervals)-1)
        centres[0] = min(df[self.vre_tag])
        centres[-1] = max(df[self.vre_tag])
        for l in range(1, len(centres) - 1):
            centres[l] = (intervals[l+1] + intervals[l]) / 2

        # Compute moving average and differences
        ma = df.rolling(window=MovavgLag).mean()
        prev_df = df.shift(1).fillna(df.iloc[0][self.vre_tag])
        diff = df - prev_df

        # Cut series to align
        ma = ma[MovavgLag-1:]
        diff = diff[MovavgLag-1:]

        # Preallocate
        PosParaList = np.zeros(len(centres))
        NegParaList = np.zeros(len(centres))
        ModeParaList= np.zeros(len(centres))

        for i, center in enumerate(centres):
            lb = intervals[i]   # Lower bound interval
            ub = intervals[i+1] # Upper bound interval
            ma_interval   = ma.loc[(ma[self.vre_tag] > lb) & (ma[self.vre_tag] <= ub)]
            diff_interval = diff.loc[ma_interval.index]
            if len(diff_interval) < 2:
                ModeValue = diff_interval.iloc[0][self.vre_tag] if len(diff_interval) == 1 else 0
                posMean = 0
                negMean = 0
            else:
                # Histogram for mode estimation
                edges = np.arange(min(diff_interval[self.vre_tag]), max(diff_interval[self.vre_tag]) + 0.001, 0.001)
                counts, _ = np.histogram(diff_interval, bins=np.append(edges, np.inf))
                sorted_counts = np.sort(counts)[::-1]
                max_count = sorted_counts[0]
                mode_bins = np.where(counts == max_count)[0]
                ModeValue = edges[int(np.mean(mode_bins))] + 0.0005

                # Separate into positive and negative deviations
                neg_diff = diff_interval.loc[diff_interval[self.vre_tag] < ModeValue] - ModeValue
                pos_diff = diff_interval.loc[diff_interval[self.vre_tag] > ModeValue] - ModeValue

                posMean = np.mean(pos_diff) if len(pos_diff) > 0 else 0
                negMean = -np.mean(neg_diff) if len(neg_diff) > 0 else 0

            PosParaList[i] = posMean
            NegParaList[i] = negMean
            ModeParaList[i]= ModeValue

        # Polynomial fits
        from numpy.polynomial import Polynomial

        # Easy alternative version (will be used):
        pol_model_mode_easy = Polynomial.fit(x=ma[self.vre_tag], y=diff[self.vre_tag], deg=1).convert()

        # Negative differences:
        diff_neg = diff.loc[pol_model_mode_easy(ma[self.vre_tag]) > diff[self.vre_tag]]
        ma_neg   = ma.loc[diff_neg.index]
        diff_neg.loc[:,self.vre_tag] -= pol_model_mode_easy(ma_neg[self.vre_tag])
        pol_model_neg_easy = Polynomial.fit(x=ma_neg[self.vre_tag], y=-diff_neg[self.vre_tag], deg=2).convert()

        # Positive differences:
        diff_pos = diff.loc[pol_model_mode_easy(ma[self.vre_tag]) < diff[self.vre_tag]]
        ma_pos   = ma.loc[diff_pos.index]
        diff_pos.loc[:,self.vre_tag] -= pol_model_mode_easy(ma_pos[self.vre_tag])
        pol_model_pos_easy = Polynomial.fit(x=ma_pos[self.vre_tag], y=diff_pos[self.vre_tag], deg=2).convert()

        pol_model_pos = Polynomial.fit(x=centres, y=PosParaList, deg=2).convert()
        pol_model_neg = Polynomial.fit(x=centres, y=NegParaList, deg=1).convert()
        pol_model_mode= Polynomial.fit(x=centres, y=ModeParaList, deg=1).convert()

        if self.documentation:
            plt.scatter(ma_pos[self.vre_tag], diff_pos[self.vre_tag], color='black', s=1, alpha=0.2, label="Observations")
            plt.plot(centres, PosParaList, label="Bracket Centroids")
            plt.plot(centres, pol_model_pos(centres), color= "green", label = "Bracket Fit - weighted fit")
            plt.plot(centres, pol_model_pos_easy(centres), color='red', label = "Normal Fit")
            plt.axhline(self.sigma_laplace, color='orange', linestyle='--', label='Mean Absolute Deviation')
            plt.xlim(intervals[0],intervals[-1])
            plt.ylim(0,2*max(PosParaList))
            plt.legend()
            plt.savefig(f'documentation/{self.forecaster.plot_dir}exp_model_fit_positives.png')
            plt.close()
            plt.scatter(ma_neg[self.vre_tag], -diff_neg[self.vre_tag], color='black', s=1, alpha=0.2, label="Observations")
            plt.plot(centres, NegParaList, label="Bracket Centroids")
            plt.plot(centres, pol_model_neg(centres), color= "green", label = "Bracket Fit - weighted fit")
            plt.plot(centres, pol_model_neg_easy(centres), color='red', label = "Normal Fit")
            plt.axhline(self.sigma_laplace, color='orange', linestyle='--', label='Mean Absolute Deviation')
            plt.xlim(intervals[0],intervals[-1])
            plt.ylim(0,2*max(NegParaList))
            plt.legend()
            plt.savefig(f'documentation/{self.forecaster.plot_dir}exp_model_fit_negatives.png')
            plt.close()
            plt.scatter(ma[self.vre_tag], diff[self.vre_tag], color='black', s=1, alpha=0.2, label="Observations")
            plt.plot(centres, ModeParaList, label="Bracket Centroids")
            plt.plot(centres, pol_model_mode(centres), color= "green", label = "Bracket Fit - weighted fit")
            plt.plot(centres, pol_model_mode_easy(centres), color='red', label = "Normal Fit")
            plt.xlim(intervals[0],intervals[-1])
            plt.ylim(min(ModeParaList),2*max(ModeParaList))
            plt.legend()
            plt.savefig(f'documentation/{self.forecaster.plot_dir}exp_model_fit_modevalues.png')
            plt.close()

        return pol_model_pos_easy, pol_model_neg_easy, pol_model_mode_easy
    
    def _calculate_interval_probabilities(self, df):
        """
        Creates a list of random interval lengths containing exclusively
        positive or negative values, distributed like the corresponding
        intervals in the input data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the time series of hourly changes in capacity utilisation.

        Returns:
        --------
        intLengthDist : np.ndarray
            Probability distribution of interval lengths.
        minMaxIntLength : int
            Minimum of the max interval lengths found in positive and negative distributions.
        """
        # Compute first order differencing series
        prev_df = df.shift(1).fillna(df.iloc[0][self.vre_tag])
        diff = df - prev_df
        # We assume that the absolute maximum streak we can get is 2 days of continuously decreasing or increasing wind
        # Maximum observed streak is 32 hours of continuosly increasing wind (an outlier from 24).
        max_len = 24*3
        self.domains = np.quantile(df, [0,0.05,0.25,0.5,0.65,0.8,0.95]) # Create equally sized domains, 4 domains
        self.domains[0] = -np.inf
        posIntLengthDist = np.zeros((max_len, len(self.domains), 3))  # columns: length, count, probability
        negIntLengthDist = np.zeros((max_len, len(self.domains), 3))


        posCount = 0
        negCount = 0
        domain = np.argwhere(df[self.vre_tag].iloc[0]>=self.domains)[-1]

        for t in range(1,len(df)):
            delta = diff[self.vre_tag].iloc[t]
            if delta >= 0:
                posCount += 1
                indicator = 1
            else:
                negCount += 1
                indicator = -1

            if posCount > 0 and negCount > 0: # Only note down when a "streak" ends.
                if indicator < 0: # Last interval was negative, but we come from a positive "streak".
                    posIntLengthDist[posCount, domain, 0] = posCount
                    posIntLengthDist[posCount, domain, 1] += 1
                    posCount = 0
                else:
                    negIntLengthDist[negCount, domain, 0] = negCount
                    negIntLengthDist[negCount, domain, 1] += 1
                    negCount = 0
                domain = np.argwhere(df[self.vre_tag].iloc[t-1]>=self.domains)[-1]
        # Compute domain dependent probabilities
        pos_totals = np.sum(posIntLengthDist[:,:,1], axis=0)
        neg_totals = np.sum(negIntLengthDist[:,:,1], axis=0)
        
        posIntLengthDist[:, :, 2] = np.divide(posIntLengthDist[:,:, 1], pos_totals)
        negIntLengthDist[:, :, 2] = np.divide(negIntLengthDist[:,:, 1], neg_totals)

        minMaxIntLength = int(max(list(posIntLengthDist[:,:, 0].flatten()) + list(negIntLengthDist[:,:, 0].flatten())))
        posIntLengthDists = posIntLengthDist[:minMaxIntLength + 1 , :, 2]
        negIntLengthDists = negIntLengthDist[:minMaxIntLength + 1 , :, 2]

        if self.documentation:
            plt.plot(np.sum(posIntLengthDist[:minMaxIntLength, :, 2], axis=1)/len(self.domains), label="Increasing wind periods (all)", color='blue')
            plt.plot(np.sum(negIntLengthDist[:minMaxIntLength, :, 2], axis=1)/len(self.domains), label="Decreasing wind periods (all)", color='red')
            plt.legend()
            plt.savefig(f'documentation/{self.forecaster.plot_dir}difference_streak_probability.png')
            plt.close()
            
            fig, axs = plt.subplots(2, 1, sharex=True)
            for dom in range(len(self.domains)):
                axs[0].plot(posIntLengthDists[:, dom], label=f"In domain ({dom})")
                axs[1].plot(negIntLengthDists[:, dom], label=f"In domain ({dom})")
            axs[0].legend(title="Increasing wind period probabilities", loc='upper right', fontsize='small', ncols=2)
            axs[1].legend(title="Decreasing wind period probabilities", loc='upper right', fontsize='small', ncols=2)
            axs[1].set_xlabel("Interval Length (hours)")
            plt.savefig(f'documentation/{self.forecaster.plot_dir}difference_streak_probability_domains.png')
            plt.close()

        return posIntLengthDists, negIntLengthDists

    def _stochastic_process_simulation(self, horizon, forecasting=False):
        # Create hourly index for year
        sim_cf       = np.zeros(horizon + self.ma_lag) # Simulated capacity factors - includes latest observations.
        deltas       = np.zeros(horizon + self.ma_lag)
        directions   = np.zeros(horizon + self.ma_lag)
        # We start the simulation from the last observation in the training data. Not important for a year sim, but maybe for a day.
        sim_cf[:self.ma_lag] = self.recent_data['observations']
        deltas[:self.ma_lag] = self.recent_data['differences']
        directions[:self.ma_lag] = np.sign(self.recent_data['differences']).astype(int)

        previous_diff_direction = directions[self.ma_lag-1] # We start the simulation from the last observation in the training data.
        # Simulate the rest of the time series:
        for t in range(self.ma_lag, horizon+self.ma_lag):
            if (directions[t] == 0 or
                (sim_cf[t-1] <= self.p5_deseason_observation and directions[t] == -1) or 
                (sim_cf[t-1] >= self.p98_deseason_observation and directions[t] == 1)):
                current_domain = np.argwhere(self.domains <= sim_cf[t-1])[-1][0]
                if previous_diff_direction == 1:
                    p = self.neg_int_length_distributions[:,current_domain]
                else:
                    p = self.pos_int_length_distributions[:,current_domain]
                interval_length = np.random.choice(range(len(p)), p=p)
                directions[t:min(t+interval_length, horizon)] = -previous_diff_direction
                previous_diff_direction *= -1
            moving_average = np.mean(sim_cf[t-self.ma_lag:t])
            direction      = directions[t-self.ma_lag]
            if direction == 1:
                mean = self.pol_model_pos(moving_average)
            else:
                mean = self.pol_model_neg(moving_average)
            mode  = self.pol_model_mode(moving_average) # Unused, could be a place of future investigation of improving the shape of the wind curve.
            delta = direction * np.random.exponential(max(self.sigma_laplace/2, mean)) #+ mode
            if direction == 1:
                delta = max(delta, 0)
            else:
                delta = min(delta, 0)
            deltas[t] = delta
            sim_cf[t] = sim_cf[t-1] + delta
        return sim_cf, deltas

    def simulate(self, capacity, solar_cf_profile, year=2023, save_last_obs=False):
        # Create hourly index for year
        hourly_index = pd.to_datetime(pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h'), utc=True)
        horizon      = len(hourly_index)
        sim_cf       = np.zeros(horizon) # Simulated capacity factors
        diff_direction = np.zeros(horizon)
        # First simulate self.mov_avg_lag laplace distributed wind values for the differences before we can bootstrap.
        sim_diff = np.random.laplace(self.mu_laplace, self.sigma_laplace, size=self.ma_lag)
        # We start the simulation from the last observation in the training data. Not important for a year sim, but maybe for a day.
        sim_cf[0]= self.last_observation + sim_diff[0]
        for t in range(1, self.ma_lag):
            sim_cf[t] = sim_cf[t-1] + sim_diff[t]
        diff_direction[:self.ma_lag] = np.sign(sim_diff).astype(int)
        previous_diff_direction = diff_direction[self.ma_lag-1] # We start the simulation from the last observation in the training data.
        # Simulate the rest of the time series:
        for t in range(self.ma_lag, horizon):
            if (diff_direction[t] == 0 or
                (sim_cf[t-1] <= self.p5_deseason_observation and diff_direction[t] == -1) or 
                (sim_cf[t-1] >= self.p98_deseason_observation and diff_direction[t] == 1)):
                current_domain = np.argwhere(self.domains <= sim_cf[t-1])[-1][0]
                if previous_diff_direction == 1:
                    p = self.neg_int_length_distributions[:,current_domain]
                else:
                    p = self.pos_int_length_distributions[:,current_domain]
                interval_length = np.random.choice(range(len(p)), p=p)
                diff_direction[t:min(t+interval_length, horizon)] = -previous_diff_direction
                previous_diff_direction *= -1
            moving_average = np.mean(sim_cf[t-self.ma_lag:t])
            direction      = diff_direction[t-self.ma_lag]
            if direction == 1:
                mean = self.pol_model_pos(moving_average)
            else:
                mean = self.pol_model_neg(moving_average)
            mode  = self.pol_model_mode(moving_average)
            delta = direction * np.random.exponential(max(self.sigma_laplace/2, mean)) #+ mode
            if direction == 1:
                delta = max(delta, 0)
            else:
                delta = min(delta, 0)
            sim_cf[t] = sim_cf[t-1] + delta
        
        if save_last_obs:
            self.last_observed_diff = delta
            self.last_observation   = sim_cf[-1]

        profile = pd.DataFrame(index=hourly_index, data={self.vre_tag: sim_cf})
        # Add daily profile and subtract mean
        profile[self.vre_tag] += profile.index.hour.map(self.daily_profile) #- np.mean(sim_cf)#self.mean_cf
        # Monthly Seasonalisation using p10 and p90 quantiles to stretch and move the simulated data.
        use_dogans_reseasoning = True
        if self.generate_weather_years:
            # weather_year = np.random.choice(self.weather_years)
            # monthly_move_factor = self.monthly_move_factors.loc[self.monthly_move_factors.index.year==weather_year]
            # monthly_stretch_factor = self.monthly_stretch_factors.loc[self.monthly_stretch_factors.index.year==weather_year]
            # monthly_move_factor.index = monthly_move_factor.index.month
            # monthly_stretch_factor.index = monthly_stretch_factor.index.month
            monthly_stretch_factor = np.clip(np.random.normal(loc = self.avg_monthly_stretch, scale=self.std_monthly_stretch),
                                          self.avg_monthly_stretch - 2 * self.std_monthly_stretch,
                                          self.avg_monthly_stretch + 2 * self.std_monthly_stretch)
            monthly_move_factor = np.clip(np.random.normal(loc = self.avg_monthly_move, scale=self.std_monthly_move),
                                          self.avg_monthly_move - 2 * self.std_monthly_move,
                                          self.avg_monthly_move + 2 * self.std_monthly_move)
            profile[self.vre_tag] = (profile[self.vre_tag] - profile.index.month.map(monthly_move_factor)) / profile.index.month.map(monthly_stretch_factor)
        elif use_dogans_reseasoning:
            p10_sim, p90_sim = np.quantile(profile[self.vre_tag], [0.1, 0.9])
            stretch_factor   = profile.index.month.map((self.avg_monthly_p90s - self.avg_monthly_p10s)[self.vre_tag]) / (p90_sim - p10_sim)
            move_factor      = -p10_sim * stretch_factor + profile.index.month.map(self.avg_monthly_p10s[self.vre_tag])
            profile[self.vre_tag] = profile[self.vre_tag] * stretch_factor + move_factor
        else:
            profile[self.vre_tag] = (profile[self.vre_tag] - profile.index.month.map(self.avg_monthly_move)) / profile.index.month.map(self.avg_monthly_stretch)
        
        profile[self.vre_tag] += self.solar_reg_model.predict(solar_cf_profile)[:,0]

        # Clip to historical observations with exponential noise around max.
        profile[self.vre_tag] += (profile[self.vre_tag] > self.max_historical_production) * ((self.max_historical_production-profile[self.vre_tag]) - np.random.exponential(abs(self.pol_model_neg(self.max_historical_production)),size=len(profile)))
        profile[self.vre_tag] += (profile[self.vre_tag] < self.min_historical_production) * ((self.min_historical_production-profile[self.vre_tag]) + np.random.exponential(abs(self.pol_model_pos(self.min_historical_production)),size=len(profile)))

        if self.documentation:
            data = self.forecaster.train_data[[self.vre_tag]]
            df_hist = self._del_capacity_trend(data)
            plt.plot(np.sort(profile[self.vre_tag]), color='red')
            for year in df_hist.index.year.unique():
                plt.plot(np.sort(df_hist.loc[df_hist.index.year==year, self.vre_tag]), color='blue')
            plt.savefig(f'documentation/{self.forecaster.plot_dir}wind_load_duration_curves.png')
            plt.close()
        
        profile *= capacity

        return profile
    
    def _simulate_cf(self, hourly_index:pd.DatetimeIndex, solar_cf_profile:pd.DataFrame=None, realize=False, forecasting = False):
        sim_cf, deltas = self._stochastic_process_simulation(horizon=len(hourly_index), forecasting=forecasting)
        # jit implementation: (not significantly faster)
        # sim_cf, deltas = simulate_stochastic_wind_process(horizon=len(hourly_index), ma_lag=self.ma_lag, observations=np.asarray(self.recent_data['observations']), differences=np.asarray(self.recent_data['differences']), domains=self.domains,
        #                          p5_obs=self.p5_deseason_observation, p98_obs=self.p98_deseason_observation, pos_dist=self.pos_int_length_distributions, neg_dist=self.neg_int_length_distributions,
        #                          sigma_laplace=np.float32(self.sigma_laplace), pol_model_pos=self.pol_model_pos.coef, pol_model_neg=self.pol_model_neg.coef, pol_model_mode=self.pol_model_mode.coef)
        
        if realize:
            self.recent_data['observations'] = sim_cf[-self.ma_lag:]
            self.recent_data['differences']  = deltas[-self.ma_lag:]

        profile = pd.DataFrame(index=hourly_index, data={self.vre_tag: sim_cf[self.ma_lag:]})
        # Add daily profile and subtract mean
        profile[self.vre_tag] += profile.index.hour.map(self.daily_profile) #- np.mean(sim_cf)#self.mean_cf
        # Monthly Seasonalisation using p10 and p90 quantiles to stretch and move the simulated data.
        use_dogans_reseasoning = False
        if self.generate_weather_years:
            # weather_year = np.random.choice(self.weather_years)
            # monthly_move_factor = self.monthly_move_factors.loc[self.monthly_move_factors.index.year==weather_year]
            # monthly_stretch_factor = self.monthly_stretch_factors.loc[self.monthly_stretch_factors.index.year==weather_year]
            # monthly_move_factor.index = monthly_move_factor.index.month
            # monthly_stretch_factor.index = monthly_stretch_factor.index.month
            monthly_stretch_factor = np.clip(np.random.normal(loc = self.avg_monthly_stretch, scale=self.std_monthly_stretch),
                                          self.avg_monthly_stretch - 2 * self.std_monthly_stretch,
                                          self.avg_monthly_stretch + 2 * self.std_monthly_stretch)
            monthly_move_factor = np.clip(np.random.normal(loc = self.avg_monthly_move, scale=self.std_monthly_move),
                                          self.avg_monthly_move - 2 * self.std_monthly_move,
                                          self.avg_monthly_move + 2 * self.std_monthly_move)
            profile[self.vre_tag] = (profile[self.vre_tag] - profile.index.month.map(monthly_move_factor)) / profile.index.month.map(monthly_stretch_factor)
        elif use_dogans_reseasoning:
            p10_sim, p90_sim = np.quantile(profile[self.vre_tag], [0.1, 0.9])
            stretch_factor   = profile.index.month.map((self.avg_monthly_p90s - self.avg_monthly_p10s)[self.vre_tag]) / (p90_sim - p10_sim)
            move_factor      = -p10_sim * stretch_factor + profile.index.month.map(self.avg_monthly_p10s[self.vre_tag])
            profile[self.vre_tag] = profile[self.vre_tag] * stretch_factor + move_factor
        else:
            profile[self.vre_tag] = (profile[self.vre_tag] - profile.index.month.map(self.avg_monthly_move)) / profile.index.month.map(self.avg_monthly_stretch)
        
        profile[self.vre_tag] += self.solar_reg_model.predict(solar_cf_profile)[:,0]

        # Clip to historical observations with exponential noise around max.
        profile[self.vre_tag] += (profile[self.vre_tag] > self.max_historical_production) * ((self.max_historical_production-profile[self.vre_tag]) - np.random.exponential(abs(self.pol_model_neg(self.max_historical_production)),size=len(profile)))
        profile[self.vre_tag] += (profile[self.vre_tag] < self.min_historical_production) * ((self.min_historical_production-profile[self.vre_tag]) + np.random.exponential(abs(self.pol_model_pos(self.min_historical_production)),size=len(profile)))

        if self.documentation:
            data = self.forecaster.train_data[[self.vre_tag]]
            df_hist = self._del_capacity_trend(data)
            plt.plot(np.sort(profile[self.vre_tag]), color='red')
            for year in df_hist.index.year.unique():
                plt.plot(np.sort(df_hist.loc[df_hist.index.year==year, self.vre_tag]), color='blue')
            plt.savefig(f'documentation/{self.forecaster.plot_dir}wind_load_duration_curves.png')
            plt.close()

        return profile

    def forecast(self, hourly_index:pd.DatetimeIndex, solar_cf_profile:pd.DataFrame=None):
        """ Forecasting wind does not change anything from just simulating it. """
        return self._simulate_cf(hourly_index=hourly_index, solar_cf_profile=solar_cf_profile, forecasting=True)

    def realize(self, hourly_index: pd.DatetimeIndex, solar_profile: pd.DataFrame):
        """ Realize a solar capacity factor time series between start and end timestamps, based on the fitted ARIMA model.
        Update the ARIMA model with the realized values, so that future simulations are conditioned on these values. """
        return self._simulate_cf(hourly_index, solar_profile, realize=True)


def simulate_year_ahead_single_run(hourly_index, solar_capacities, wind_capacities,
                                   solar_model, wind_model, price_model,
                                   solar_tag, wind_tag, price_tag, tags, seed=None):
    # np.random.seed(seed)
    solar_simulation_cf = solar_model._simulate_cf(hourly_index)
    wind_simulation_cf  = wind_model._simulate_cf(hourly_index, solar_cf_profile=solar_simulation_cf)
    
    vre_profiles = pd.DataFrame(index=hourly_index,
                                data={solar_tag: solar_simulation_cf[solar_tag] * solar_capacities.values,
                                      wind_tag: wind_simulation_cf[wind_tag] * wind_capacities.values})
    
    price_simulation = price_model.simulate(vre_profiles)
    
    df = vre_profiles.copy()
    df[price_tag] = price_simulation[price_tag]
    df[solar_tag] = solar_simulation_cf[solar_tag]
    df[wind_tag]  = wind_simulation_cf[wind_tag]
    
    return df[tags]


class DataForecaster:
    _tags = RenewableFuelPlant.uncertainties

    def __init__(self,
                 database:HistoricalData  = None,
                 price_tag = 'price',
                 wind_tag = 'wind',
                 solar_tag = 'solar',
                 log_prices = False,
                 log_vre = False,
                 documentation = False,
                 seasonal_price_regression = False,
                 day_night_price_regression = False,
                 weather_years = True,
                 verbose = True,
                 auto_arima = False,
                 plot_dir = "",
                 cache_id = None,
                 cache_replace = False,
                 from_pickle = False,
                 ):
        """ Initialize and set up train and test data. """
        if from_pickle:
            cache_path = os.getcwd() + "/models/ts_models/forecaster/" + str(cache_id) + ".pkl"
            if cache_exists(cache_path):
                self.unpickled = cache_read(cache_path)
            else:
                raise FileNotFoundError("No cached model found with the given cache_id.")
        else:
            if database is None:
                raise ValueError("A HistoricalData instance must be provided if we are not unpickling.")
            self.database       = database
            self.data           = database.data
            self.price_tag, self.wind_tag, self.solar_tag = price_tag, wind_tag, solar_tag
            self.log_prices     = log_prices
            self.log_vre        = log_vre
            self.seasonal_price_regression = seasonal_price_regression
            self.day_night_price_regression = day_night_price_regression
            self.weather_years  = weather_years
            self.verbose        = verbose
            self.auto_arima     = auto_arima
            self.plot_dir       = plot_dir + "/" if plot_dir != "" else plot_dir
            self.documentation  = documentation if plot_dir == "" else True
            dn = os.path.dirname(os.getcwd() + "/documentation/" + self.plot_dir)
            if not os.path.exists(dn):
                os.mkdir(dn)
            self.cache_id, self.cache_replace = cache_id, cache_replace
            self.create_train_test_data()
    
    def unpickle(self):
        if hasattr(self, 'unpickled'):
            return self.unpickled
        else:
            raise AttributeError("No unpickled object found. Please initialize with from_pickle=True.")

    def build_simulation_models(self, to_pickle=False):
        # Build time series models:
        self.solar_model = SolarSimulationTool(self, caps=self.database.caps, vre_tag=self.solar_tag, weather_years=False)
        self.solar_model.fit()
        self.wind_model  = WindSimulationTool(self, caps=self.database.caps, vre_tag=self.wind_tag, weather_years=self.weather_years)
        self.wind_model.fit()
        self.price_model = PriceSimulationTool(self)
        self.price_model.fit()
        self.solar_realization_cf, self.solar_realization_cf = None, None
        if to_pickle:
            cache_path = os.getcwd() + "/models/ts_models/forecaster/" + str(self.cache_id) + ".pkl"
            if self.cache_id is not None: cache_write(self, cache_path, verbose=self.verbose)

    def create_train_test_data(self):
        ## Train and test split and y (prices) and X (renewables).
        self.train_data, self.test_data = pm.model_selection.train_test_split(self.data, test_size=8760) # A full year of test data, should be at least two years of data.
        # self.y_train, self.y_test = self.train_data[[self.price_tag]], self.test_data[[self.price_tag]]        
        # self.X_train, self.X_test = self.train_data[feature_tags], self.test_data[feature_tags]
        self.t_zero = self.train_data.index[0].timestamp() # To be used when fitting trend and later on when reapplying trend.
        self.t_init =  self.train_data.index[-1] + pd.Timedelta(1, 'hour')

    def simulate(self, year, caps, n_sims=1):
        sims = []
        for sim in tqdm(range(n_sims), disable=not(self.verbose)):
            solar_simulation = self.solar_model.simulate(capacity = caps.loc[year, self.solar_tag], year=year)
            solar_simulation_cf = self.solar_model._del_capacity_trend(solar_simulation)
            wind_simulation  = self.wind_model.simulate(capacity  = caps.loc[year, self.wind_tag], solar_cf_profile=solar_simulation_cf,  year=year)
            # wind_simulation = self.simulate_wind(caps.loc[year, self.wind_tag], horizon=len(solar_simulation))
            price_simulation = self.price_model.simulate(wind_simulation, solar_simulation, year=year)
            sims.append({self.wind_tag : wind_simulation, self.solar_tag : solar_simulation, self.price_tag : price_simulation})
        return sims

    def _simulate_year_ahead_single_run(self, hourly_index, solar_capacities, wind_capacities, seed=None):
        # np.random.seed(seed)
        solar_simulation_cf = self.solar_model._simulate_cf(hourly_index)
        wind_simulation_cf  = self.wind_model._simulate_cf(hourly_index, solar_cf_profile=solar_simulation_cf)
        vre_profiles = pd.DataFrame(index=hourly_index,
                                    data={self.solar_tag : solar_simulation_cf[self.solar_tag] * solar_capacities.values, 
                                            self.wind_tag : wind_simulation_cf[self.wind_tag] * wind_capacities.values})
        price_simulation = self.price_model.simulate(vre_profiles)
        df = vre_profiles.copy()
        df[self.price_tag] = price_simulation[self.price_tag]
        df[self.solar_tag] = solar_simulation_cf[self.solar_tag]
        df[self.wind_tag]  = wind_simulation_cf[self.wind_tag]
        return df[self._tags]

    def simulate_year_ahead(self, start:pd.Timestamp, n_sims=3, deterministic:bool = False):
        end = start + relativedelta(years=+1) - pd.Timedelta(1, 'hour')
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'), utc=True)
        solar_capacities = hourly_index.year.map(self.database.caps[self.solar_tag])
        wind_capacities = hourly_index.year.map(self.database.caps[self.wind_tag])
        t_s = time()
        sims = [self._simulate_year_ahead_single_run(hourly_index, solar_capacities, wind_capacities) for _ in range(n_sims)]
        print(f"Simulated {n_sims} year aheads in {time() - t_s} seconds.")
        # sims = Parallel(n_jobs=-1)(
        #     delayed(simulate_year_ahead_single_run)(
        #         hourly_index,
        #         solar_capacities,
        #         wind_capacities,
        #         self.solar_model,
        #         self.wind_model,
        #         self.price_model,
        #         self.solar_tag,
        #         self.wind_tag,
        #         self.price_tag,
        #         self._tags,
        #         seed
        #     ) for seed in range(n_sims)
        # )
        if deterministic: # We average the simulated futures instead of returning all of them. Will ruin the crosscorrelation.
            wind_forecast  = np.mean(np.asarray([df[self.wind_tag].values for df in sims]), axis=1)
            solar_forecast = np.mean(np.asarray([df[self.solar_tag].values for df in sims]), axis=1)
            price_forecast = np.mean(np.asarray([df[self.price_tag].values for df in sims]), axis=1)
            sims = pd.DataFrame(index=hourly_index, data = {self.price_tag : price_forecast, 
                                                                 self.solar_tag: solar_forecast, 
                                                                 self.wind_tag: wind_forecast})
            sims = sims[self._tags]
        return sims

    def realize_vre(self, start:pd.Timestamp, end:pd.Timestamp):
        """ Simulates (and then assumes that this is the future that is realized).
        self.solar_model and self.wind_model is updated with the realized data, which is appended to the existing model data. """
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'), utc=True)
        self.solar_realization_cf = self.solar_model.realize(hourly_index=hourly_index)
        self.wind_realization_cf  = self.wind_model.realize(hourly_index=hourly_index, solar_profile=self.solar_realization_cf)
        return self.solar_realization_cf.copy(), self.wind_realization_cf.copy()

    def realize_prices(self, start:pd.Timestamp, end:pd.Timestamp):
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'), utc=True)
        solar_capacities = hourly_index.year.map(self.database.caps[self.solar_tag])
        wind_capacities = hourly_index.year.map(self.database.caps[self.wind_tag])
        if (self.solar_realization_cf is None) or (self.wind_realization_cf is None):
            raise (SyntaxError, "Need to realize VRE before we can realize prices.")
        else:
            assert (hourly_index == self.solar_realization_cf.index).all()
            vre_profiles = pd.DataFrame(index=hourly_index,
                                        data={self.solar_tag : self.solar_realization_cf[self.solar_tag] * solar_capacities.values, 
                                              self.wind_tag  : self.wind_realization_cf[self.wind_tag] * wind_capacities.values})
            prices = self.price_model.realize(vre_profiles)
            return prices

    def forecast(self, start:pd.Timestamp, end:pd.Timestamp, n_forecasts:int = 10, deterministic:bool = False):
        """ Call to forecast the VRE and prices from the current time stamp of the price models.
        The VRE is likely determined for the first time period.
        If we only ask for one forecast, then call forecast, otherwise simulate possible outcomes. """
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'), utc=True)
        solar_capacities = hourly_index.year.map(self.database.caps[self.solar_tag])
        wind_capacities = hourly_index.year.map(self.database.caps[self.wind_tag])
        df_structure = pd.DataFrame(index=hourly_index, columns=[self.price_tag, self.solar_tag, self.wind_tag])
        if (self.solar_realization_cf is not None) and (self.wind_realization_cf is not None):
            vre_forecast_index = pd.to_datetime(pd.date_range(self.solar_realization_cf.index[-1] + pd.Timedelta(1, 'hour'), end, freq='h'), utc=True)
        else:
            vre_forecast_index = hourly_index
        forecasts = []
        for _ in range(n_forecasts):
            df    = df_structure.copy()
            solar_forecast_cf = self.solar_model.forecast(hourly_index=vre_forecast_index)
            wind_forecast_cf  = self.wind_model.forecast(hourly_index=vre_forecast_index, solar_cf_profile=solar_forecast_cf)
            solar_production  = pd.concat([self.solar_realization_cf[self.solar_tag], solar_forecast_cf[self.solar_tag]]) * solar_capacities.values
            wind_production   = pd.concat([self.wind_realization_cf[self.wind_tag], wind_forecast_cf[self.wind_tag]]) * wind_capacities.values
            price_forecast    = self.price_model.forecast(pd.DataFrame(index=hourly_index, data = {self.solar_tag: solar_production, self.wind_tag: wind_production}))
            df.loc[hourly_index, self.solar_tag] = pd.concat([self.solar_realization_cf[self.solar_tag], solar_forecast_cf[self.solar_tag]])
            df.loc[hourly_index, self.wind_tag]  = pd.concat([self.wind_realization_cf[self.wind_tag], wind_forecast_cf[self.wind_tag]])
            df.loc[hourly_index,       self.price_tag] = price_forecast[self.price_tag]
            forecasts.append(df)
        if deterministic: # We average the simulated futures instead of returning all of them. Will likely ruin the crosscorrelation.
            wind_forecast  = np.mean(np.asarray([df[self.wind_tag].values for df in forecasts]), axis=1)
            solar_forecast = np.mean(np.asarray([df[self.solar_tag].values for df in forecasts]), axis=1)
            price_forecast = np.mean(np.asarray([df[self.price_tag].values for df in forecasts]), axis=1)
            forecasts = pd.DataFrame(index=hourly_index, data = {self.price_tag : price_forecast, 
                                                                 self.solar_tag: solar_forecast, 
                                                                 self.wind_tag: wind_forecast})
        return forecasts

    def investigate_test_simulation_monthly(self, simulations, resource='price'):
        real_data = self.test_data[resource]
        simulated_data = [sim[resource] for sim in simulations]
        year = simulated_data[0].index.year[0]
        if resource == 'price':
            cap = 1
            ylabel="[€/MWh]"
        else:
            cap = self.database.caps.loc[year, resource]
            ylabel="MW"
        fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharey=True)
        axes = axes.flatten()
        # fig.tight_layout(h_pad=5.0, w_pad=2.0)
        plt.tight_layout(pad=4.0, rect=[0.03, 0.03, 0.97, 0.95])

        # Loop through each month
        for i, month in enumerate(range(1, 13)):
            monthly_data = real_data[real_data.index.month == month]
            sorted_ = np.sort(monthly_data.values)
            mean_monthly = np.mean(monthly_data.values)/cap
            sim_means = [np.mean(sim[sim.index.month == month].values)/cap for sim in simulated_data]
            ax = axes[i]
            for sim in simulated_data:
                d = sim[resource]
                m_d = d[d.index.month == month]
                # ax.plot(np.sort(m_d.values), color='blue', alpha=0.4)
            # Draw confidence intervals
            mtx = np.asarray([np.sort(sim.loc[sim.index.month == month, resource].values).reshape(-1) for sim in simulated_data])
            p_low = np.percentile(mtx, 5, axis=0)
            p_high = np.percentile(mtx, 95, axis=0)
            ax.fill_between(range(len(p_low)), p_low, p_high, color='blue', alpha=0.2, label='90% CI')
            ax.plot(sorted_, label=f'Realized', color='black')
            ax.set_xlim(0, len(sorted_))
            if not(resource == 'price'): ax.set_ylim(0, max(sorted_) * 1.3)
            ax.set_title(f'Month {month}')
            txt = "Mean Price" if resource == 'price' else "Mean Capacity Factor"
            ax.annotate(f'{txt}\nRealized: {mean_monthly:.2f}\nSim: {np.mean(sim_means):.2f}', xy=(0.25, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
            ax.legend()
            ax.set_xlabel("Hours")
            ax.set_ylabel(ylabel)
        plt.savefig(f'documentation/{self.plot_dir}monthly_duration_curve_{resource}.png')
        plt.close()

    def investigate_annual_duration_curves(self, simulations, year, resource='price'):
        plt.figure(figsize=(10, 6))
        train_data = self.train_data[resource]
        simulated_data = [sim[resource] for sim in simulations]
        year = simulated_data[0].index.year[0]
        if resource == 'price':
            ylabel="[€/MWh]"
            cap = 1
        else:
            cap = self.database.caps.loc[year, resource]
            ylabel="MW"
        
        # for ix, sim in enumerate(simulated_data):
        #     lbl = "" if ix > 0 else f"Simulations of year {year}" 
        #     plt.plot(np.sort(sim[resource]), color='blue', alpha=0.2, label=lbl)
        # Draw confidence intervals
        mtx = np.asarray([np.sort(sim[resource].values).reshape(-1) for sim in simulated_data])
        p_low = np.percentile(mtx, 5, axis=0)
        p_high = np.percentile(mtx, 95, axis=0)
        plt.fill_between(range(len(p_low)), p_low, p_high, color='blue', alpha=0.2, label='90% CI')
        plt.plot(np.sort(np.mean(mtx, axis=0)), color='blue', alpha=0.8, label='Mean of simulations')
        for yr in train_data.index.year.unique():
            plt.plot(np.sort(train_data.loc[train_data.index.year==yr]), label=yr, alpha=0.8)

        test_data = self.test_data[resource]
        plt.plot(np.sort(test_data), label=test_data.index.year.unique(), color='black', alpha=0.8)
        plt.xlabel("Hours")
        plt.ylabel(ylabel)

        txt = "Mean Price" if resource == 'price' else "Mean Production [MW]"
        plt.annotate(f'{txt}:\nTraining set: {np.mean(train_data):.2f}\nValidation set: {np.mean(test_data):.2f}\nSim: {np.mean(simulated_data):.2f}', xy=(0.25, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

        plt.legend()
        plt.savefig(f'documentation/{self.plot_dir}annual_duration_curve_{resource}.png')
        plt.close()

