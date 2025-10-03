#%% Initialization
from data_scripts.data_loader import HistoricalData
from utils import cache_exists, cache_read, cache_write, trigo_fit, log_transform, delog_transform, laplace_rnd

#from sklearn.pipeline import Pipeline
#from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from hmmlearn import hmm

from scipy.optimize import curve_fit
from scipy.special import expit, logit

import statsmodels.api as sm
import pmdarima as pm

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

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
    
    def plot_impact_of_deseason(self, data, model, residuals, labels=['Data', 'Model', 'Residuals'], name="xx"):
        fig, ax = plt.subplots(1, figsize=(12,10))
        ax.scatter(data.index, data,        color='red',    s=2, alpha=0.4, label=labels[0])
        ax.scatter(data.index, model,       color='blue',   s=2, alpha=0.4, label=labels[1])
        ax.scatter(data.index, residuals,   color='green',  s=2, alpha=0.4, label=labels[2])
        ax.set_ylabel(self.ylabel)
        ax.legend()
        plt.savefig('documentation/' + name + 'png')
        plt.close()
    
    def _fit_arima_model(self, time_series, order=(5,0,1), seasonal_order=(0, 0, 0, 0), name = ""):
        # Fit ARIMA model to the remaining stochastic process
        cache_path = os.getcwd() + "/models/ts_models/" + self.tool_type + "/" + name + str(self.cache_id) + ".pkl"
        if self.cache_id is not None and not(self.cache_replace) and cache_exists(cache_path):
            arima_model = cache_read(cache_path)
        else:
            if self.verbose: print(self.tool_type + name + ' model initialization'); t_start = time()
            if self.auto_arima:
                arima_model = pm.auto_arima(y=time_series, d=0, seasonal=True, m=24, error_action='ignore', suppress_warnings=True)
            else:
                arima_model = pm.ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True, error_action='ignore',)
            arima_model.fit(time_series)
            if self.verbose: print(self.tool_type + f' model fitted in {time() - t_start} seconds.')
            if self.cache_id is not None: cache_write(arima_model, cache_path, verbose=self.verbose)
        return arima_model

    def _arima_simulate(self, horizon = 8760):
        if self.arima_model is not None:
            return self.arima_model.arima_res_.simulate(nsimulations=horizon)
        else:
            raise("Cannot simulate arima process before fitting the model.")

    def fit(self):
        pass

    def simulate(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

class PriceSimulationTool(SimulationTool):
    tool_type = 'price'
    ylabel    = '€/MWh'

    def __init__(self, forecaster, hmm = True):
        super().__init__(forecaster)
        self.log_prices = self.forecaster.log_prices
        self.price_tag  = self.forecaster.price_tag
        self._tag       = self.price_tag
        self.hmm        = hmm
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

    def _del_weekly_cycle(self, df):
        day_index       = self.train_data.day_of_week
        avg_days        = df.groupby(day_index).mean()
        daily_avgs      = day_index.map(avg_days[self.price_tag])
        impact          = daily_avgs.values.reshape(len(df),1)
        residuals       = df - impact
        if self.documentation: self.plot_impact_of_deseason(df, impact, residuals, name=self.tool_type + "removing_weekday_effect")
        return residuals, avg_days

    def _del_daily_cycle(self, df):
        hour_index      = df.index.hour
        avg_hours       = df.groupby(hour_index).mean()
        hourly_avgs     = hour_index.map(avg_hours[self.price_tag])
        impact          = hourly_avgs.values.reshape(len(df),1)
        residuals       = df - impact
        if self.documentation: self.plot_impact_of_deseason(df, impact, residuals, name=self.tool_type + "removing_hourly_effect")
        return residuals, avg_hours

    def _deseasonalise(self, df):
        """
        Deseasonalizes time series using trigonometric fitting and logit transformation.

        Parameters:
        - feed_in: array-like, raw generation data
        - capacity: array-like, installed capacity

        Returns:
        - deseasonalised_y: residuals after seasonal fit
        - y: logit-transformed normalized data
        - beta_season_fit: fitted parameters
        """
        def _logit_n_seasonal_fit(df):
            y = df
            # Moving average (equivalent to movavg(Y,1,1,0) in MATLAB)
            moving = y.rolling(window=4, center=True, min_periods=1).mean().values.reshape(-1)

            # Initial guess for parameters
            beta0 = [1.5, 1/365, 1, 1/365, 1, 1, 1]

            # Fit the trigonometric model
            t = np.arange(len(moving))
            beta_opt, _ = curve_fit(f=trigo_fit, xdata=t, ydata=moving, p0=beta0)

            # Seasonal fit
            seasonal_effect = trigo_fit(t, *beta_opt)

            y["residuals"] = y[self.price_tag] - seasonal_effect
            y["seasonal_fit"] = seasonal_effect

            return y

        residuals = []
        fits = []
        # Create a separate seasonal fit for every year
        for year in df.index.year.unique():
            y = _logit_n_seasonal_fit(df.loc[df.index.year == year])
            residuals += list(y['residuals'].values)
            fits += list(y['seasonal_fit'].values)

        df_residuals = pd.DataFrame(index=df.index, data={self.price_tag: residuals})
        df_fits = pd.DataFrame(index=df.index, data={self.price_tag: fits})
        hour_of_year_index = df.index.hour + 24 * (df.index.day_of_year - 1)
        yearly_cycle = df_fits.groupby(hour_of_year_index).mean()

        if self.documentation:
            for year in df.index.year.unique():
                plt.plot(np.arange(len(df_fits.loc[df.index.year==year])),df_fits.loc[df.index.year==year], label = 'Fit')
            plt.plot(np.arange(len(yearly_cycle)), yearly_cycle, label = 'Average')
            plt.legend()
            plt.savefig(f'documentation/seasonal_fits_{self.price_tag}.png')
            plt.close()

        return df_residuals, yearly_cycle

    def _remove_autoregressive_pattern(self, df):
        """ Not used, but good to investigate cross-correlation and autocorrelation"""
        from statsmodels.tsa.stattools import ccf
        ar1 = ccf(df, df)[1]
        prev_df1 = df.shift(1).fillna(df.iloc[0][self.price_tag])
        diff1 = df - prev_df1 * ar1
        ar24 = ccf(diff1, diff1)[24]
        prev_df24 = diff1.shift(24)
        prev_df24.loc[prev_df24.index[0:24], self.price_tag] = diff1.iloc[0:24][self.price_tag]
        diff24 = diff1 - prev_df24 * ar24 # Formula: r_p(t) = p(t) - p(t-24) * ar(24) -> p(t) = r_p(t) + p(t-24) * ar(24)
        return diff24, ar1, ar24

    def _fit_hmm(self, residuals, n_states=50, ):
        wind = self.train_data[[self.forecaster.wind_tag]]
        solar = self.train_data[[self.forecaster.solar_tag]]
        X = residuals.copy()
        X.loc[:,'lag1'] = residuals.shift(1).fillna(residuals.iloc[0] * 0.9)
        X.loc[:,'lag24'] = residuals.shift(24)
        X.loc[X.index[0:24],'lag24'] = X.iloc[0:24][self.price_tag] * 0.3
        X.loc[:,wind.columns] = wind   # The state is (price(t-1), price(t-24), wind(t), solar(t))
        X.loc[:,solar.columns] = solar # So we get a sarimax transition: price(t) = fn(price(t-1), price(t-24), wind(t), solar(t))
        X = X[['lag1', 'lag24', 'wind', 'solar']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Use scaler.partial_fit for incremental updates when seeing new observations.

        cache_path = os.getcwd() + "/models/ts_models/hmm_models/" + str(self.cache_id) + ".pkl"
        if self.cache_id is not None and not(self.cache_replace) and cache_exists(cache_path):
            hmm_model = cache_read(cache_path)
        else:
            if self.verbose: print(f"Fitting HMM model on {len(X_scaled)} observations...")
            t_start = time()
            hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type = "diag", n_iter = 50) # 210 seconds with 50 states and 50 iter and diag.
            hmm_model.fit(X_scaled)
            if self.verbose: print(f"HMM model fit done in {time()-t_start} seconds")
            if self.cache_id is not None: cache_write(hmm_model, cache_path, verbose=self.verbose)
        
        return hmm_model, scaler
        # Fit four models, one for each quarter of the year
        # for q in range(1,5):
        #     X_ = X_scaled.loc[X_scaled.index.quarter == q]
        #     model = hmm.GaussianHMM(n_components=n_states, covariance_type = "diag", n_iter = 50)
        #     model.fit(X_)

    def fit_old(self):
        data = self.forecaster.y_train # Historical price data
        ### Fit linear model "price(t) = w1 * wind(t) + w2 * solar(t) + eps(t)"
        self.residual_prices, self.wss_model = self.WSS_price_regression(data) # Creates self.residual_prices and self.wss_model

        if self.log_prices: # Possible log-transform of the price residuals. Never done - should be removed.
            self.residual_prices = log_transform(self.residual_prices)
        
        ### Remove trends, cycles, and patterns from the price residuals.
        self.residual_prices, self.price_trend_model    = self._del_trend(self.residual_prices)
        self.residual_prices, self.monthly_avg_prices   = self._del_annual_cycle(self.residual_prices)
        self.residual_prices, self.daily_avg_prices     = self._del_weekly_cycle(self.residual_prices)
        self.residual_prices, self.hourly_avg_prices    = self._del_daily_cycle(self.residual_prices)
        # self.residual_prices, self.annual_trig_cycle    = self._deseasonalise(self.residual_prices)

        ## Calculate regime-switching probabilities and remove extreme residuals from residuals
        # Creates self.rs_prob_matrix, which shows the probabilities of going from one regime to another 
        # Creates second-order laplace moments of extreme price regimes.
        self.residual_prices    = self._calculate_rs_probabilities(self.residual_prices)

        if self.documentation:
            self.investigate_heteroskedasticity()
            plot_acf(self.residual_prices)
        
        self.arima_model        = self._fit_arima_model(self.residual_prices, order=(5,0,1), seasonal_order=(0, 0, 0, 0))

    def fit(self):
        self.train_data = self.forecaster.train_data.copy() # Historical price data
        # r stands for residuals
        residuals = self.train_data[[self.price_tag]]
        if not(self.hmm):
            residuals, self.wss_model = self.WSS_price_regression(residuals)
        residuals, self.trend_model = self._del_trend(residuals)
        self.max_historical, self.min_historical = np.max(residuals), np.min(residuals)
        residuals, self.monthly_avg = self._del_annual_cycle(residuals)
        residuals, self.weekday_avg, self.weekend_avg = self._del_weekday_and_weekend_pattern(residuals)
        # r4, self.annual_trig_cycle = self._deseasonalise(r3)
        if self.hmm:
            # Remove lag AR(1) term from residuals and then AR(24) from residuals
            # residuals, self.ar1_corr, self.ar24_corr = self._remove_autoregressive_pattern(residuals)
            # We do this directly in the HMM to include it in the simulation dependency
            self.hmm_model, self.scaler = self._fit_hmm(residuals, n_states = 50)
        else:
            residuals                 = self._calculate_rs_probabilities(residuals)
            self.arima_model          = self._fit_arima_model(residuals, order=(1,0,0), seasonal_order=(1, 0, 0, 24))    
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
        plt.savefig('documentation/heteroskedastic_visual.png')
        plt.close()

    def _calculate_rs_probabilities(self, prices):
        ### We define extreme observations as observations outside 2 standard deviations of the mean (of 0)
        self.n_regimes = 3
        residual_mu = np.mean(prices)
        residual_std = np.std(prices.values)
        high_price_regime = (prices > 2 * residual_std).astype(int)
        low_price_regime = (prices < -2 * residual_std).astype(int)
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
            ax.set_xlim(-100,300)
            plt.savefig('documentation/regime_histogram.png')
            plt.close()

        self.high_prices                = high_prices # The outliers are not normal distributed at all. Sampled from a one-sided laplace distribution later
        self.high_base, self.high_std   = 2 * residual_std, np.std(self.high_prices)
        self.low_prices                 = low_prices # The outliers are not normal distributed at all. Sampled from a one-sided laplace distribution later
        self.low_base,  self.low_std    = -2 * residual_std, np.std(self.low_prices)

        # Remove extreme residuals from time series and replace with mean
        residuals = prices * (1-high_price_regime)*(1-low_price_regime) + residual_mu * ((high_price_regime) | (low_price_regime))
        return residuals

    def _generate_price_regimes(self, horizon=8760):
        realized_price_regimes  = [np.random.choice(self.n_regimes, p=self.price_regime_probabilities)]
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
        return extreme_prices

    def simulate(self, wind_profile, solar_profile, year=2023):
        # Simulate spot market electricity prices for the entire horizon
        df = pd.DataFrame(index = solar_profile.index,
                                        data = {self.forecaster.wind_tag  : wind_profile[self.forecaster.wind_tag].values,
                                                self.forecaster.solar_tag : solar_profile[self.forecaster.solar_tag].values})
        time_info = self.forecaster.database._specify_time_data(pd.DataFrame(index=pd.to_datetime(df.index, utc=True)))
        horizon = len(df)
        if self.hmm:
            return self._simulate_hmm(df, time_info, horizon)
        else:
            return self._simulate(df, time_info, horizon)

    def _simulate_hmm(self, df, time_info, horizon):
        # Simulate spot market electricity prices for the entire horizon
        df = df.copy()
        sim_prices = np.zeros(horizon)

        ### HMM simulation ###
        sim_vre_scaled = self.scaler.transform(df)[:,2:] # Scale and drop placeholder price lists
        # Initial time steps to get lags: Assume solar profile is similar to the first day
        sim_prev_24_hours = [self.hmm_model.means_[self.hmm_model.startprob_.argmax()]] # Initial price lags set as the most probable starts.]
        sim_prev_24_hours[0][-1] = sim_vre_scaled[0,1]
        currstate = np.argmin(np.sum((self.hmm_model.means_ - sim_prev_24_hours[0])**2 , axis=1))
        for h in range(1,24):
            next_state, next_state_ix = self.hmm_model.sample(n_samples = 1, currstate = currstate)
            next_state[0][-1] = sim_vre_scaled[h,1]
            sim_prev_24_hours.append(next_state[0])
            currstate = np.argmin(np.sum((self.hmm_model.means_ - next_state[0])**2 , axis=1))

        # Actual simulation time steps:
        for h in range(horizon):
            lag1 = sim_prices[h-1] if h>0 else sim_prev_24_hours[-1][0]
            lag24 = sim_prices[h-24] if h>23 else sim_prev_24_hours[h][1]
            curr_state = np.array([lag1, lag24] + list(sim_vre_scaled[h]))
            currstate  = np.argmin(np.sum((self.hmm_model.means_ - curr_state)**2, axis=1))
            next_state, _ = self.hmm_model.sample(n_samples = 1, currstate = currstate) # price[h] = fn(price[h-1], wind[h], solar[h])
            sim_prices[h] = next_state[0][0]
        sim_prices = self.scaler.inverse_transform(np.concatenate([sim_prices.reshape(horizon, 1), sim_prices.reshape(horizon, 1), sim_vre_scaled], axis=1))[:,0] # Remove scaling
        df[self.price_tag] = sim_prices
        # Add daily patterns:
        df.loc[time_info.is_weekend, self.price_tag] += df.loc[time_info.is_weekend].index.hour.map(self.weekend_avg)
        df.loc[time_info.is_weekday, self.price_tag] += df.loc[time_info.is_weekday].index.hour.map(self.weekday_avg)
        df[self.price_tag] += df.index.month.map(self.monthly_avg[self.price_tag])

        df[self.price_tag] = np.clip(df[self.price_tag], self.min_historical, self.max_historical) # GaussianHMM can produce outliers as variance is not dependent on state?
        
        # Add trend effect on prices:
        u_hours = pd.DataFrame(index=df.index, data={'timestamp': [h.timestamp() - self.forecaster.t_zero for h in df.index]}) # Convert to Unix time
        df[self.price_tag] += self.trend_model.predict(u_hours)[:,0]

        return df[[self.price_tag]]

    def _simulate(self, df, time_info, horizon):
        df = df.copy()
        # Simulate stochastic process:
        df['stoch_price_residuals']  = self._arima_simulate(horizon=horizon).values
        # Simulate price regime transitions following Markov Transition Process:
        df['extreme_price_impact']   = self._generate_price_regimes(horizon=horizon)
        # Obtain initial residual prices
        df[self.price_tag] = df['stoch_price_residuals'] + df['extreme_price_impact']

        # TODO: Does not currently work with seasonal_price_regression = True
        _tags = [self.forecaster.wind_tag, self.forecaster.solar_tag]
        if self.seasonal_price_regression or self.day_night_price_regression: # Include solar and wind data as regressors that have varying weights dependent on season, weekend, and day/night.
            X = self.forecaster.database._create_seasonal_features(df.copy(), prod_columns = _tags)
        else:
            X = df[_tags].copy()
        if self.forecaster.log_vre:
            X['log_' + self.forecaster.wind_tag] = log_transform(df.loc[:, self.forecaster.wind_tag])
            X['log_' + self.forecaster.solar_tag] = log_transform(df.loc[:, self.forecaster.solar_tag])
        X = X[self.feature_tags]

        # Add daily patterns:
        df.loc[time_info.is_weekend, self.price_tag] += df.loc[time_info.is_weekend].index.hour.map(self.weekend_avg)
        df.loc[time_info.is_weekday, self.price_tag] += df.loc[time_info.is_weekday].index.hour.map(self.weekday_avg)
        df[self.price_tag] += df.index.month.map(self.monthly_avg[self.price_tag])

        df[self.price_tag] = np.clip(df[self.price_tag], self.min_historical, self.max_historical)
        
        # Add trend effect on prices:
        u_hours = pd.DataFrame(index=df.index, data={'timestamp': [h.timestamp() - self.forecaster.t_zero for h in df.index]}) # Convert to Unix time
        df[self.price_tag] += self.trend_model.predict(u_hours)[:,0]

        # Add merit order effect on prices:
        df[self.price_tag] += self.wss_model.predict(X)[:,0]

        return df[[self.price_tag]]

    def simulate_old(self, wind_profile, solar_profile, year=2023):
        # Forecast spot market electricity prices for the entire horizon
        df_simulation = pd.DataFrame(index = solar_profile.index,
                                        data = {self.forecaster.wind_tag:wind_profile[self.forecaster.wind_tag].values,
                                                self.forecaster.solar_tag:solar_profile[self.forecaster.solar_tag].values})
        horizon = len(df_simulation)

        df_simulation['extreme_price_impact']   = self._generate_price_regimes(horizon=horizon) # Array of price regimes following Markov Transition Process
        df_simulation['stoch_price_residuals']  = self._arima_simulate(horizon=horizon)

        df_simulation[self.price_tag] = df_simulation['stoch_price_residuals'] + df_simulation['extreme_price_impact']
        df_simulation[self.price_tag] += df_simulation.index.hour.map(self.hourly_avg_prices[self.price_tag])
        df_simulation[self.price_tag] += df_simulation.index.day_of_week.map(self.daily_avg_prices[self.price_tag])
        df_simulation[self.price_tag] += df_simulation.index.month.map(self.monthly_avg_prices[self.price_tag])
        u_hours = pd.DataFrame({'timestamp': [h.timestamp() - self.forecaster.t_zero for h in df_simulation.index]}, index=df_simulation.index) # Convert to Unix time
        df_simulation[self.price_tag] += self.price_trend_model.predict(u_hours)[:,0]
        if self.seasonal_price_regression: # Include solar and wind data as regressors that have varying weights dependent on season, weekend, and day/night.
            X = self.forecaster.database._create_seasonal_features(df_simulation[[self.forecaster.wind_tag, self.forecaster.solar_tag]],
                                                                   prod_columns=[self.forecaster.wind_tag, self.forecaster.solar_tag])
        else:
            X = df_simulation[[self.forecaster.wind_tag, self.forecaster.solar_tag]]
        df_simulation[self.price_tag] += self.wss_model.predict(X)[:,0]

        return df_simulation[self.price_tag]

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

class SolarSimulationTool(RenewablesSimulationTool):
    tool_type = 'solar'

    def fit_old_version(self):
        # Step 1: Remove effect from added capacities of each year.
        data = self.forecaster.train_data[[self.vre_tag]]
        df = self._del_capacity_trend(data) # Produces capacity utilization factor for all timestamps

        # Step 2: Identify hourly profiles for each mon
        # df, self.max_months = self._del_annual_cycle(df)
        self.hourly_monthly_mean_profiles = df.groupby([df.index.month, df.index.hour]).mean()
        self.hourly_monthly_std_profiles = df.groupby([df.index.month, df.index.hour]).std()
        
        # for month in range(1,13):
        #     hourly_profiles = self.hourly_monthly_mean_profiles.loc[month]
        #     index = df.index.month == month
        #     hours_of_month = df.index[index]
        #     df.loc[index, self.vre_tag] = hours_of_month.hour.map(hourly_profiles[self.vre_tag])
        #     df.loc[index, self.vre_tag] -= hours_of_month.tz_localize(None).to_period('d').to_timestamp().map(self.hourly_monthly_mean_profiles[self.vre_tag])
        # This approach does not work^^ Unsure how to do it

        # Step 3: Establish a time series of daily maximum values
        daily_max = df.groupby(df.index.date).max()
        daily_max.index = pd.to_datetime(daily_max.index)

        self.residuals, self.yearly_cycle = self.deseasonalise_data(daily_max)
        
        if self.documentation:
            for f in seasonal_fits:
                plt.plot(self.yearly_cycle.index,f, label = 'Fit')
            plt.plot(self.yearly_cycle, label = 'Average')
            plt.legend()
            plt.savefig(f'documentation/seasonal_fits_{self.vre_tag}.png')
            plt.close()

        # Step X:
        if self.documentation: plot_acf(self.residuals)

        self.arima_model = self._fit_arima_model(self.residuals, order=(2,0,0))

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
            fig, axs = plt.subplots(4,3)
            axs = axs.flatten()
            for month in range(1,13):
                ax = axs[month-1]
                for day in range(1,28):
                    month_mean_values = self.hourly_monthly_mean_profiles.loc[(month), self.vre_tag].values
                    actual_values = df.loc[(df.index.day == day) & (df.index.month == month), self.vre_tag].values[:24]
                    ax.scatter(np.arange(1,25), actual_values - month_mean_values)
            plt.savefig('documentation/monthly_variation_from_daily_mean.png')
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

        # Step 5: Fit arima model to residual data.
        self.arima_model = self._fit_arima_model(self.residuals, order=(2,0,0))

    def deseasonalise_data(self, df):
        """
        Deseasonalizes time series using trigonometric fitting and logit transformation.

        Parameters:
        - feed_in: array-like, raw generation data
        - capacity: array-like, installed capacity

        Returns:
        - deseasonalised_y: residuals after seasonal fit
        - y: logit-transformed normalized data
        - beta_season_fit: fitted parameters
        """
        def _logit_n_seasonal_fit(df):
            # Normalize and apply logit transformation
            y = logit(np.clip(df, 1e-2, 1 - 1e-2))  # avoid infs

            # Moving average (equivalent to movavg(Y,1,1,0) in MATLAB)
            moving = y.rolling(window=3, center=True, min_periods=1).mean().values.reshape(-1)

            # Initial guess for parameters
            beta0 = [1.5, 1/365, 1, 1/365, 1, 1, 1]

            # Fit the trigonometric model
            t = np.arange(len(moving))
            beta_opt, _ = curve_fit(f=trigo_fit, xdata=t, ydata=moving, p0=beta0)

            # Seasonal fit
            seasonal_effect = trigo_fit(t, *beta_opt)

            y["residuals"] = y[self.vre_tag] - seasonal_effect
            y["seasonal_fit"] = seasonal_effect

            return y

        residuals = []
        fits = []
        # Create a separate seasonal fit for every year
        for year in df.index.year.unique():
            y = _logit_n_seasonal_fit(df.loc[df.index.year == year])
            residuals += list(y['residuals'].values)
            fits += list(y['seasonal_fit'].values)

        df_residuals = pd.DataFrame(index=df.index, data={self.vre_tag: residuals})
        df_fits = pd.DataFrame(index=df.index, data={self.vre_tag: fits})

        yearly_cycle = df_fits.groupby(df.index.day_of_year).mean()

        if self.documentation:
            for year in df.index.year.unique():
                plt.plot(yearly_cycle.index,df_fits.loc[df.index.year==year], label = 'Fit')
            plt.plot(yearly_cycle, label = 'Average')
            plt.legend()
            plt.savefig(f'documentation/seasonal_fits_{self.vre_tag}.png')
            plt.close()

        return df_residuals, yearly_cycle

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
        profile[self.vre_tag] *= capacity

        return profile

    def simulate_old_version(self, capacity, year=2023):
        # Create hourly index for leap year
        hourly_index = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')
        day_index    = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='d')
        n_hours = len(hourly_index)
        n_days  = int(n_hours/24)
        df_ = self.yearly_cycle
        if n_days == 366: # Leap year - duplicate the 28th of February
            df_ = pd.concat([df_.loc[:59], df_.loc[59:59], df_.loc[60:]])
            df_ = df_.sort_index().reset_index(drop=True)
        daily_max_cycle = pd.DataFrame(index=day_index, data={self.vre_tag: df_[self.vre_tag].values})

        # Simulate daily maximum values of solar production
        simulated_daily_residuals = self._arima_simulate(n_days)
        simulated_daily_max_logit = daily_max_cycle[self.vre_tag] + np.array(simulated_daily_residuals)
        simulated_daily_max       = expit(simulated_daily_max_logit) # Inverse of logit operation
        simulated_daily_max       = pd.DataFrame(index=day_index, data={self.vre_tag : simulated_daily_max})

        # Create hourly profiles:
        profile = pd.DataFrame(index=hourly_index, columns=[self.vre_tag])
        for month in range(1,13):
            hourly_means = self.hourly_monthly_mean_profiles.loc[month]
            index = profile.index.month == month
            hours_of_month = profile.index[index]
            # profile.loc[index, self.vre_tag] = [np.clip(np.random.normal(loc=hourly_means.loc[hour,self.vre_tag], scale=hourly_stds.loc[hour,self.vre_tag]),0,1) for hour in hours_of_month.hour]
            profile.loc[index, self.vre_tag] = [hourly_means.loc[hour,self.vre_tag] for hour in hours_of_month.hour]
            profile.loc[index, self.vre_tag] *= hours_of_month.date.map(simulated_daily_max[self.vre_tag])
        
        # profile[self.vre_tag] *= profile.index.month.map(self.max_months[self.vre_tag])

        profile[self.vre_tag] *= capacity

        return profile

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

        max_lag = 168 # Maximum lag considered is a full week.
        self.ma_lag = self._get_significant_ma_lag(df, max_lag) # 11 is the most significant lag (ofc dependent on historical data).

        # Polyfit of 2nd order for positive differences, 1st order for negative and mode differences:
        self.pol_model_pos, self.pol_model_neg, self.pol_model_mode = self._get_exponential_models(df, self.ma_lag)

        self.pos_int_length_distributions, self.neg_int_length_distributions, diff = self._calculate_interval_probabilities(df)
        self.last_observed_diff = int(np.sign(diff.iloc[-1][self.vre_tag]))
        self.last_observation = df.iloc[-1][self.vre_tag]

        self.mu_laplace = np.median(diff) # Laplace distribution parameter
        self.sigma_laplace = sum(abs(diff[self.vre_tag] - self.mu_laplace)) / len(diff) # Laplace distribution parameter
        self.p5_deseason_observation = np.quantile(df, 0.05)
        self.p98_deseason_observation = np.quantile(df, 0.98)

        # self.arima_model = self._fit_arima_model(self.residuals, order=(5,0,2), )#seasonal_order=((1,7), 0, 0, 24))

    def fit_alternative(self):
        # Step 1: Remove effect from added capacities of each year.
        data = self.forecaster.train_data[[self.vre_tag]]
        df = self._del_capacity_trend(data) # Produces capacity utilization factor for all timestamps

        # Step 2: Identify hourly profiles for each month
        self.hourly_monthly_mean_profiles = df.groupby([df.index.month, df.index.hour]).mean()

        # Step 3: Establish a time series of daily maximum values
        self.daily_max = df.groupby(df.index.tz_localize(None).to_period('d')).max()
        self.daily_max.index = self.daily_max.index.to_timestamp()

        residuals = []
        seasonal_fits = []
        logit_transformed_max_values = []
        self.trigonometric_curve_fits = []

        for year in df.index.year.unique():
            y, beta_opt = self.deseasonalise_data(self.daily_max.loc[self.daily_max.index.year == year])
            residuals.append(y['residuals'])
            seasonal_fits.append(y['seasonal_fit'])
            logit_transformed_max_values.append(y[self.vre_tag])
            self.trigonometric_curve_fits.append(beta_opt)

        self.residuals = pd.DataFrame(index= self.daily_max.index, data={self.vre_tag: np.array(residuals).reshape(-1)})
        fits = pd.DataFrame(index= self.daily_max.index, data={self.vre_tag: np.array(seasonal_fits).reshape(-1)})
        self.yearly_cycle = fits.groupby(fits.index.day_of_year).mean()

        if self.documentation:
            for f in seasonal_fits:
                plt.plot(self.yearly_cycle.index,f, label = 'Fit')
            plt.plot(self.yearly_cycle, label = 'Average')
            plt.legend()
            plt.savefig(f'documentation/seasonal_fits_{self.vre_tag}.png')
            plt.close()

        # Step X:
        if self.documentation: plot_acf(self.residuals)

        self.arima_model = self._fit_arima_model(self.residuals, order=(1,0,0))

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
            plt.savefig('documentation/wind_corr_ma_n_diff.png')
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
            Coefficients of 1st-degree polynomial fit for negative mean deviations.
        pfParaMode : ndarray
            Coefficients of 1st-degree polynomial fit for mode values.
        """

        # Define intervals and centres, the lower bound interval is 0 but not explicitly written.
        intervals = np.array(
            [-np.inf, *np.arange(0.02, 0.15, 0.01), 0.18, 0.20, 0.23, 0.25, 0.28, 0.3,
            0.34, 0.38, 0.41, 0.44, 0.56, 0.70, np.inf]
        )
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
            plt.xlim(intervals[0],intervals[-1])
            plt.ylim(0,2*max(PosParaList))
            plt.legend()
            plt.savefig('documentation/exp_model_fit_positives.png')
            plt.close()
            plt.scatter(ma_neg[self.vre_tag], -diff_neg[self.vre_tag], color='black', s=1, alpha=0.2, label="Observations")
            plt.plot(centres, NegParaList, label="Bracket Centroids")
            plt.plot(centres, pol_model_neg(centres), color= "green", label = "Bracket Fit - weighted fit")
            plt.plot(centres, pol_model_neg_easy(centres), color='red', label = "Normal Fit")
            plt.xlim(intervals[0],intervals[-1])
            plt.ylim(0,2*max(NegParaList))
            plt.legend()
            plt.savefig('documentation/exp_model_fit_negatives.png')
            plt.close()
            plt.scatter(ma[self.vre_tag], diff[self.vre_tag], color='black', s=1, alpha=0.2, label="Observations")
            plt.plot(centres, ModeParaList, label="Bracket Centroids")
            plt.plot(centres, pol_model_mode(centres), color= "green", label = "Bracket Fit - weighted fit")
            plt.plot(centres, pol_model_mode_easy(centres), color='red', label = "Normal Fit")
            plt.xlim(intervals[0],intervals[-1])
            plt.ylim(min(ModeParaList),2*max(ModeParaList))
            plt.legend()
            plt.savefig('documentation/exp_model_fit_modevalues.png')
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
            plt.savefig('documentation/difference_streak_probability.png')
            plt.close()
            linestyles = ['--', '-.', ':', '-']
            fig, axs = plt.subplots(2, 1)
            for dom in range(len(self.domains)):
                axs[0].plot(posIntLengthDists[:, dom], label=f"Increasing wind periods in domain ({dom})")
                axs[1].plot(negIntLengthDists[:, dom], label=f"Decreasing wind periods in domain ({dom})")
            axs[0].legend()
            axs[1].legend()
            plt.savefig('documentation/difference_streak_probability_domains.png')
            plt.close()

        return posIntLengthDists, negIntLengthDists, diff

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
            plt.savefig('documentation/wind_load_duration_curves.png')
            plt.close()
        
        profile *= capacity

        return profile

    def simulate_alternative(self, capacity, year=2023):
        # Create hourly index for leap year
        hourly_index = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')
        day_index    = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='d')
        n_hours = len(hourly_index)
        n_days  = int(n_hours/24)
        df = self.yearly_cycle
        if n_days == 366: # Leap year - duplicate the 28th of February
            df = pd.concat([df.loc[:59], df.loc[59:59], df.loc[60:]])
            df = df.sort_index().reset_index(drop=True)
        daily_max_cycle = pd.DataFrame(index=day_index, data={self.vre_tag: df[self.vre_tag].values})

        # Simulate daily maximum values of solar production
        simulated_daily_residuals = self._arima_simulate(n_days)
        simulated_daily_max_logit = daily_max_cycle[[self.vre_tag]] + np.array(simulated_daily_residuals)
        simulated_daily_max       = expit(simulated_daily_max_logit) # Inverse of logit operation
        simulated_daily_max       = pd.DataFrame(index=day_index, data={self.vre_tag : simulated_daily_max})

        # Create hourly profiles:
        profile = pd.DataFrame(index=hourly_index, columns=[self.vre_tag])
        hourly_average_profiles = self.hourly_monthly_mean_profiles
        for month in range(1,13):
            hourly_profiles = hourly_average_profiles.loc[month]
            index = profile.index.month == month
            hours_of_month = profile.index[index]
            profile.loc[index, self.vre_tag] = hours_of_month.hour.map(hourly_profiles[self.vre_tag])
            profile.loc[index, self.vre_tag] *= hours_of_month.tz_localize(None).to_period('d').to_timestamp().map(simulated_daily_max[self.vre_tag])
        
        profile[self.vre_tag] *= capacity

        return profile

class DataForecaster:
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
            self.documentation  = documentation
            self.seasonal_price_regression = seasonal_price_regression
            self.day_night_price_regression = day_night_price_regression
            self.weather_years  = weather_years
            self.verbose        = verbose
            self.auto_arima     = auto_arima
            self.cache_id, self.cache_replace = cache_id, cache_replace
            self.create_train_test_data()
    
    def unpickle(self):
        if hasattr(self, 'unpickled'):
            return self.unpickled
        else:
            raise AttributeError("No unpickled object found. Please initialize with from_pickle=True.")

    def build_simulation_models(self, hmm = True, to_pickle=False):
        # Build time series models:
        self.solar_model = SolarSimulationTool(self, caps=self.database.caps, vre_tag=self.solar_tag, weather_years=False)
        self.solar_model.fit()
        self.wind_model  = WindSimulationTool(self, caps=self.database.caps, vre_tag=self.wind_tag, weather_years=self.weather_years)
        self.wind_model.fit()
        self.price_model = PriceSimulationTool(self, hmm)
        self.price_model.fit()
        if to_pickle:
            cache_path = os.getcwd() + "/models/ts_models/forecaster/" + str(self.cache_id) + ".pkl"
            if self.cache_id is not None: cache_write(self, cache_path, verbose=self.verbose)

    def create_train_test_data(self):
        ## Train and test split and y (prices) and X (renewables).
        self.train_data, self.test_data = pm.model_selection.train_test_split(self.data, test_size=8760) # A full year of test data, should be at least two years of data.
        # self.y_train, self.y_test = self.train_data[[self.price_tag]], self.test_data[[self.price_tag]]        
        # self.X_train, self.X_test = self.train_data[feature_tags], self.test_data[feature_tags]
        self.t_zero  = self.train_data.index[0].timestamp() # To be used when fitting trend and later on when reapplying trend.

    def simulate(self, year, caps, n_sims=1):
        sims = []
        for sim in tqdm(range(n_sims)):
            solar_simulation = self.solar_model.simulate(capacity = caps.loc[year, self.solar_tag], year=year)
            solar_simulation_cf = self.solar_model._del_capacity_trend(solar_simulation)
            wind_simulation  = self.wind_model.simulate(capacity  = caps.loc[year, self.wind_tag], solar_cf_profile=solar_simulation_cf,  year=year)
            # wind_simulation = self.simulate_wind(caps.loc[year, self.wind_tag], horizon=len(solar_simulation))
            price_simulation = self.price_model.simulate(wind_simulation, solar_simulation, year=year)
            sims.append({self.wind_tag : wind_simulation, self.solar_tag : solar_simulation, self.price_tag : price_simulation})
        return sims

    def investigate_test_simulation_monthly(self, simulations, resource='price'):
        real_data = self.test_data[resource]
        simulated_data = [sim[resource] for sim in simulations]
        year = simulated_data[0].index.year[0]
        if resource == 'price':
            cap = 1
        else:
            cap = self.database.caps.loc[year, resource]
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
            mtx = np.array([np.sort(sim.loc[sim.index.month == month, resource].values).reshape(-1) for sim in simulated_data])
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
        plt.savefig(f'documentation/monthly_duration_curve_{resource}.png')
        plt.close()

    def investigate_annual_duration_curves(self, simulations, year, resource='price'):
        plt.figure(figsize=(10, 6))
        train_data = self.train_data[resource]
        simulated_data = [sim[resource] for sim in simulations]
        year = simulated_data[0].index.year[0]
        if resource == 'price':
            cap = 1
        else:
            cap = self.database.caps.loc[year, resource]
        simulated_data = [sim[resource] for sim in simulations]
        for ix, sim in enumerate(simulated_data):
            lbl = "" if ix > 0 else f"Simulations of year {year}" 
            plt.plot(np.sort(sim[resource]), color='blue', alpha=0.2, label=lbl)
        
        for yr in train_data.index.year.unique():
            plt.plot(np.sort(train_data.loc[train_data.index.year==yr]), label=yr, alpha=0.8)

        test_data = self.test_data[resource]
        plt.plot(np.sort(test_data), label=test_data.index.year.unique(), color='black', alpha=0.8)

        txt = "Mean Price" if resource == 'price' else "Mean Production [MW]"
        plt.annotate(f'{txt}:\nTraining set: {np.mean(train_data):.2f}\nValidation set: {np.mean(test_data):.2f}\nSim: {np.mean(simulated_data):.2f}', xy=(0.25, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

        plt.legend()
        plt.savefig(f'documentation/annual_duration_curve_{resource}.png')
        plt.close()

