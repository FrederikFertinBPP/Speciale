#%% Initialization
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from hmmlearn import hmm
import statsmodels.api as sm

from data_scripts.data_loader import DataLoader
from common_scripts.RFP_initialization import RenewableFuelPlant
from common_scripts.utils import cache_exists, cache_read, cache_write, log_transform, laplace_rnd

# ---------------------------------------------------------------------------
# Base simulation class
# ---------------------------------------------------------------------------

class SimulationTool:
    """Abstract base for stochastic simulation tools.

    All configuration is injected explicitly; no reference to a parent
    forecaster is held.
    """
    tool_type: str = ''
    ylabel:    str = ''
    log_transform_garch_residuals = False

    def __init__(self,
                 target_tag: str = None,
                 *,
                 documentation: bool = False,
                 cache_id=None,
                 cache_replace: bool = False,
                 verbose: bool = True,
                 plot_dir: str = '',
                 ):
        self.target_tag    = target_tag
        self.documentation = documentation
        self.cache_id      = cache_id
        self.cache_replace = cache_replace
        self.verbose       = verbose
        self.plot_dir      = (plot_dir.rstrip('/') + '/') if plot_dir else ''
        self.arima_model   = None
        self.recent_data: dict = {}

    @staticmethod
    def _default_time_data(df: pd.DataFrame) -> pd.DataFrame:
        """Minimal fallback when no database callable is supplied."""
        out = pd.DataFrame(index=df.index)
        out['is_weekend'] = df.index.dayofweek >= 5
        out['is_weekday'] = df.index.dayofweek < 5
        return out

    @staticmethod
    def fourier_terms(t, period:int, K:int) -> pd.DataFrame:
        """Generate K Fourier harmonics for a given period."""
        t = np.asarray(t)
        terms = {}
        for k in range(1, K + 1):
            terms[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
            terms[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
        return pd.DataFrame(terms)

    # ------------------------------------------------------------------
    # Decomposition steps
    # ------------------------------------------------------------------

    def _del_trend(self, residuals: pd.DataFrame):
        hours = pd.DataFrame(
            {'timestamp': [h.timestamp() - self.t_zero for h in self.data.index]},
            index=self.data.index)
        model = LinearRegression()
        model.fit(hours, residuals)
        impact           = model.predict(hours)
        fitted_residuals = residuals - impact
        if self.documentation:
            self._plot_deseason(residuals, impact, fitted_residuals,
                                name=self.tool_type + 'removing_trend_effect')
        return fitted_residuals, model

    def _del_annual_cycle(self, residuals: pd.DataFrame):
        month_index      = residuals.index.month
        avg_months       = residuals.groupby(month_index).mean()
        monthly_avgs     = month_index.map(avg_months[residuals.columns[0]])
        impact           = monthly_avgs.values.reshape(-1, 1)
        fitted_residuals = residuals - impact
        if self.documentation:
            self._plot_deseason(residuals, impact, fitted_residuals,
                                name=self.tool_type + 'removing_seasonal_effect')
        return fitted_residuals, avg_months

    def _del_weekday_and_weekend_pattern(self, df_: pd.DataFrame):
        df           = df_.copy()
        is_weekend   = self.data['is_weekend']
        is_weekday   = self.data['is_weekday']
        weekend_data = df.loc[is_weekend, self.target_tag]
        weekday_data = df.loc[is_weekday, self.target_tag]
        avg_weekend  = weekend_data.groupby(weekend_data.index.hour).mean()
        avg_weekday  = weekday_data.groupby(weekday_data.index.hour).mean()
        df.loc[is_weekend, self.target_tag] -= weekend_data.index.hour.map(avg_weekend)
        df.loc[is_weekday, self.target_tag] -= weekday_data.index.hour.map(avg_weekday)
        return df, avg_weekday, avg_weekend

    # ------------------------------------------------------------------
    # Plotting helper
    # ------------------------------------------------------------------

    def _plot_deseason(self, data, model, residuals,
                       labels=('Data', 'Model', 'Residuals'), name='xx'):
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(data.index, data,      color='red',   s=2, alpha=0.4, label=labels[0])
        ax.scatter(data.index, model,     color='blue',  s=2, alpha=0.4, label=labels[1])
        ax.scatter(data.index, residuals, color='green', s=2, alpha=0.4, label=labels[2])
        ax.set_ylabel(self.ylabel)
        ax.legend()
        plt.savefig(f'documentation/{self.plot_dir}{name}.png')
        plt.close()

    # ------------------------------------------------------------------
    # ARIMA helpers
    # ------------------------------------------------------------------

    def _fit_arima_model(self, time_series, exog=None, order=(5, 0, 1),
                         seasonal_order=(0, 0, 0, 0), name='', old_model=None):
        s_tag      = '' if sum(seasonal_order) == 0 else f'_s{seasonal_order}'
        cache_path = (os.getcwd() + '/models/ts_models/' + self.tool_type + '/'
                      + name + str(order) + s_tag + str(self.cache_id) + '.pkl')

        if self.cache_id is not None and not self.cache_replace and cache_exists(cache_path):
            return sm.tsa.statespace.SARIMAXResults.load(cache_path)

        if self.verbose:
            print(f'{self.tool_type}{name} model initialisation')
            t0 = time()

        start_params  = old_model.params.values if old_model else None
        sarimax_kwargs = dict(endog=time_series, exog=exog, order=order, seasonal_order=seasonal_order,
                              dates=time_series.index, freq=time_series.index.freq)
        try:
            res = sm.tsa.statespace.SARIMAX(**sarimax_kwargs).fit(start_params=start_params)
        except Exception as exc:
            print('Error fitting ARIMA model:', exc)
            print('Retrying without enforcing stationarity …')
            res = sm.tsa.statespace.SARIMAX(**sarimax_kwargs,
                                            enforce_stationarity=False).fit(start_params=start_params)

        if self.verbose:
            print(f'{self.tool_type} model fitted in {time() - t0:.1f} s.')
        if self.cache_id is not None:
            res.save(cache_path)
        return res

    def _arima_simulate(self, horizon: int = 8760, exog=None, realize: bool = False,
                        repetitions=None) -> np.ndarray:
        if self.arima_model is None:
            raise RuntimeError('Call fit() before _arima_simulate().')
        sim = self.arima_model.simulate(
            nsimulations=horizon, exog=exog, anchor='end', repetitions=repetitions).values
        if realize:
            self.arima_model = self.arima_model.extend(sim)
        return sim

    def _arima_forecast(self, horizon: int = 24, exog = None) -> np.ndarray:
        if self.arima_model is None:
            raise RuntimeError('Call fit() before _arima_forecast().')
        return self.arima_model.forecast(steps=horizon, exog=exog).values

    # ------------------------------------------------------------------
    # GARCH helpers
    # ------------------------------------------------------------------

    def _fit_garch_model(self, residuals: pd.Series, exog=None, old_model=None) -> object:
        from arch import arch_model

        cache_path = (os.getcwd() + '/models/ts_models/price/garch'
                    + str(self.cache_id) + '.pkl')
        if self.cache_id and not self.cache_replace and cache_exists(cache_path) and old_model is None:
            import pickle
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        if self.log_transform_garch_residuals:
            self.min_residuals = np.min(residuals)
            y = np.log(residuals - self.min_residuals + np.exp(1)) # Log transform residuals.
            self.mean_residuals = np.mean(y)
            y -= self.mean_residuals
            y *= 10
        else:
            y = residuals

        am  = arch_model(y=y, x=exog, mean='HARX', lags=np.asarray([[1,2,13,24], [1,12,23,24]]), vol='GARCH', p=1, q=1, dist='skewt')

        res = am.fit(
            starting_values=old_model.params.values if old_model else None,
            disp='off',
            options={"maxiter": 10**5}
        )
        if self.verbose:
            print(res.summary())
        if self.cache_id and old_model is None:
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(res, f)
        return res

    def _garch_simulate(self, horizon: int = 8760, exog=None, realize: bool = False, forecasting: bool = False) -> np.ndarray:
        if self.garch_model is None:
            raise RuntimeError('Call fit() before _garch_simulate().')

        cols = self.feature_tags if self.stochastic_model == "GARCHX" else exog.columns if exog is not None else None
        # arch simulation anchors from the last observed state automatically
        sim_result = self.garch_model.forecast(
            x=dict(exog[cols]) if exog is not None else None,
            horizon=horizon,
            method='simulation' if not forecasting else 'analytic',
            simulations=1,
            reindex=False,
        )
        # shape: (1, horizon) — squeeze to 1D
        if forecasting:
            sim = sim_result.mean.values[0]
        else:
            sim = sim_result.simulations.values[-1, 0, :]
        if realize:
            # Re-fit or append — see note below
            self._extend_garch(sim, exog=exog[cols] if exog is not None else None)
        
        if self.log_transform_garch_residuals:
            sim = np.exp(sim / 10 + self.mean_residuals) + self.min_residuals - np.exp(1)

        return sim
    
    def _extend_garch(self, new_residuals: np.ndarray, exog=None):
        # Note: The arch package does not support extending fitted GARCH models with new data.
        # As a workaround, we re-fit the model with the new residuals appended to the original data.
        # This is less efficient than true extension but ensures that the model parameters are updated
        # based on the latest information, which is crucial for accurate simulation.
        cols = self.feature_tags if self.stochastic_model == "GARCHX" else exog.columns
        extended_residuals = np.concatenate([self.residuals[self.target_tag].values, new_residuals])
        self.residuals = pd.DataFrame(extended_residuals, columns=[self.target_tag])
        extended_exog = np.concatenate([self.X_train.values, exog.values]) if exog is not None else None
        self.X_train = pd.DataFrame(extended_exog, columns=cols) if exog is not None else None
        self.garch_model = self._fit_garch_model(self.residuals[self.target_tag], 
                                                 exog=self.X_train if self.stochastic_model == 'GARCHX' else None,
                                                 old_model=self.garch_model)

    # ------------------------------------------------------------------
    # Interface stubs
    # ------------------------------------------------------------------

    def fit(self):               pass
    def simulate(self, *a, **k): pass
    def realize(self,  *a, **k): pass


# ---------------------------------------------------------------------------
# Price simulation class
# ---------------------------------------------------------------------------

class PriceSimulationTool(SimulationTool):
    """Simulates hourly spot electricity prices from wind, solar, and seasonal patterns.

    Parameters
    ----------
    data : pd.DataFrame
        Historical hourly data. Must contain columns for ``price_tag``,
        ``wind_tag``, ``solar_tag``, ``is_weekend``, and ``is_weekday``.
    price_tag, wind_tag, solar_tag : str
        Column names for the respective time series.
    t_zero : float, optional
        Unix timestamp of the first training observation. Derived from
        ``data`` when omitted.
    specify_time_data : callable, optional
        ``fn(df) -> df`` that appends ``is_weekend`` / ``is_weekday`` boolean
        columns to an arbitrary future index. Falls back to a simple
        ``dayofweek``-based implementation when omitted.
    create_seasonal_features : callable, optional
        Required only when ``seasonal_price_regression=True``. Signature:
        ``fn(df, prod_columns) -> df_with_seasonal_features``.
    gamma : float
        Exponential decay for sample weights in the linear regressions.
    """
    tool_type = 'price'
    ylabel    = '€/MWh'
    fourier_period = 24
    fourier_order = 6

    def __init__(self,
                 data: pd.DataFrame,
                 price_tag: str,
                 wind_tag: str,
                 solar_tag: str,
                 main_exog_tags: list,
                 *,
                 other_exog_tags: list = None,
                 log_prices: bool = False,
                 log_vre: bool = False,
                 seasonal_price_regression: bool = False,
                 day_night_price_regression: bool = False,
                 t_zero: float = None,
                 specify_time_data=None,
                 create_seasonal_features=None,
                 exog_model: LinearRegression = None,
                 stochastic_model: str = 'GARCH',
                 **kwargs):
        super().__init__(target_tag=price_tag, **kwargs)
        self.wind_tag   = wind_tag
        self.solar_tag  = solar_tag
        self.main_exog_tags = main_exog_tags
        self.other_exog_tags = other_exog_tags or []
        self.log_prices = log_prices
        self.log_vre    = log_vre
        self.seasonal_price_regression  = seasonal_price_regression
        self.day_night_price_regression = day_night_price_regression
        self.t_zero     = t_zero if t_zero is not None else data.index[0].timestamp()
        self._specify_time_data        = specify_time_data or self._default_time_data
        self._create_seasonal_features = create_seasonal_features
        self.exog_model = exog_model or LinearRegression()
        self.stochastic_model = stochastic_model
        self.data           = data.copy()
    
    # ------------------------------------------------------------------
    # Decomposition steps
    # ------------------------------------------------------------------

    def _wss_price_regression(self, residuals: pd.DataFrame):
        """Wind-Solar-Season linear regression to capture the merit-order effect."""
        wt, st = self.wind_tag, self.solar_tag
        feature_tags = [wt, st]

        if self.seasonal_price_regression:
            feature_tags = [f'{wt}-is_summer', f'{st}-is_summer',
                            f'{wt}-is_winter', f'{st}-is_winter',
                            f'{wt}-is_spring', f'{st}-is_spring',
                            f'{wt}-is_autumn', f'{st}-is_autumn']
        elif self.day_night_price_regression:
            feature_tags = [f'{wt}-is_day', f'{st}-is_day',
                            f'{wt}-is_night', f'{st}-is_night']
        if self.log_vre:
            feature_tags += [f'log_{wt}', f'log_{st}']

        self.feature_tags = np.unique(feature_tags + self.main_exog_tags + self.other_exog_tags)
        X_train = self.data[self.feature_tags]

        model = self.exog_model
        model.fit(X_train, residuals)
        merit_order_effect = model.predict(X_train)
        fitted_residuals   = residuals - merit_order_effect.reshape(-1, 1)

        if self.verbose:
            from sklearn.metrics import root_mean_squared_error
            print('RMSE of WSS fit (train):', root_mean_squared_error(residuals, merit_order_effect))
        if self.documentation:
            self._plot_deseason(
                residuals, merit_order_effect - model.intercept_,
                fitted_residuals + model.intercept_,
                labels=['Historical Prices', 'Merit Order Effect', 'Residuals'],
                name='removing_merit_order_effect')
            self._plot_deseason(
                residuals, merit_order_effect, fitted_residuals,
                labels=['Historical Prices', 'Model', 'Residuals'],
                name='removing_merit_order_effect_and_bias')
        return fitted_residuals, model

    # ------------------------------------------------------------------
    # Regime model
    # ------------------------------------------------------------------

    def _calculate_rs_probabilities(self, prices: pd.DataFrame) -> pd.DataFrame:
        self.n_regimes = 3
        mu             = float(np.mean(prices))
        std            = float(np.std(prices.values))
        high_mask = (prices >  2 * std + mu).astype(int)
        low_mask  = (prices < -2 * std + mu).astype(int)
        regimes   = high_mask - low_mask + 1  # 0 = Low, 1 = Normal, 2 = High

        self.rs_prob_matrix = np.zeros((self.n_regimes, self.n_regimes))
        from_r = 1
        for to_r in regimes.values:
            self.rs_prob_matrix[from_r, to_r] += 1
            from_r = to_r
        self.price_regime_probabilities = self.rs_prob_matrix.sum(axis=0)
        self.rs_prob_matrix = np.transpose(
            np.transpose(self.rs_prob_matrix) / self.rs_prob_matrix.sum(axis=1))
        self.rs_prob_matrix = np.nan_to_num(self.rs_prob_matrix)

        high_prices   = prices.iloc[:, 0][high_mask.astype(bool).iloc[:, 0]]
        low_prices    = prices.iloc[:, 0][low_mask.astype(bool).iloc[:, 0]]
        normal_prices = prices.iloc[:, 0][
            ~low_mask.astype(bool).iloc[:, 0] & ~high_mask.astype(bool).iloc[:, 0]]

        if self.documentation:
            bw = 10
            bins_fn = lambda d: np.arange(min(d), max(d) + bw, bw)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.hist(high_prices,   label='High Regime',     color='darkblue',  bins=bins_fn(high_prices))
            ax.hist(low_prices,    label='Low Regime',      color='lightblue', bins=bins_fn(low_prices))
            ax.hist(normal_prices, label='Standard Regime', color='blue',      bins=bins_fn(normal_prices))
            ax.set_xlabel('€/MWh')
            ax.legend()
            ax.set_title('Price regimes of residuals')
            ax.set_xlim(float(np.min(prices)), float(np.max(prices)))
            plt.savefig(f'documentation/{self.plot_dir}regime_histogram.png')
            plt.close()

        self.high_prices = high_prices
        self.high_base   =  2 * std + mu
        self.high_std    = float(sum(abs(high_prices - self.high_base)) / max(1, len(high_prices)))
        self.low_prices  = low_prices
        self.low_base    = -2 * std + mu
        self.low_std     = float(sum(abs(low_prices - self.low_base)) / max(1, len(low_prices)))

        residuals = prices * (1 - high_mask) * (1 - low_mask) + mu * (high_mask | low_mask)
        self.recent_data['regime'] = int(regimes[self.target_tag].values[-1])
        return residuals

    def _investigate_heteroskedasticity(self, residuals):
        from statsmodels.stats.diagnostic import het_breuschpagan
        self.bp_test        = het_breuschpagan(
            residuals, sm.add_constant(self.data[[self.wind_tag, self.solar_tag]]))
        self.heteroskedastic = self.bp_test[3] < 0.05

        fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
        fig.tight_layout(pad=4.0, rect=[0.03, 0.03, 0.97, 0.95])
        ylabel = 'Residuals' + ('' if self.log_prices else ' [€/MWh]')
        axes[0].set_ylabel(ylabel)
        axes[0].scatter(self.data.index, residuals, s=1)
        axes[0].set_xticks(axes[0].get_xticks(),
                           labels=axes[0].get_xticklabels(), rotation=45)
        axes[1].scatter(self.data[self.wind_tag],  residuals, s=1)
        axes[2].scatter(self.data[self.solar_tag], residuals, s=1)
        plt.savefig(f'documentation/{self.plot_dir}heteroskedastic_visual.png')
        plt.close()

    def _extreme_residuals_garch(self, prices: pd.DataFrame) -> pd.DataFrame:
        self.n_regimes = 3
        self.extreme_cutoffs = np.quantile(prices, [0.02, 0.98])
        high_mask = prices[self.target_tag] > self.extreme_cutoffs[1]
        low_mask  = prices[self.target_tag] < self.extreme_cutoffs[0]

        high_prices   = prices.loc[high_mask, self.target_tag]
        low_prices    = prices.loc[low_mask, self.target_tag]
        normal_prices = prices.loc[~high_mask & ~low_mask, self.target_tag]

        if self.documentation:
            bw = 10
            bins_fn = lambda d: np.arange(min(d), max(d) + bw, bw)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.hist(high_prices,   label='High Regime',     color='darkblue',  bins=bins_fn(high_prices))
            ax.hist(low_prices,    label='Low Regime',      color='lightblue', bins=bins_fn(low_prices))
            ax.hist(normal_prices, label='Standard Regime', color='blue',      bins=bins_fn(normal_prices))
            ax.set_xlabel('€/MWh')
            ax.legend()
            ax.set_title('Price regimes of residuals')
            ax.set_xlim(float(np.min(prices)), float(np.max(prices)))
            plt.savefig(f'documentation/{self.plot_dir}regime_histogram.png')
            plt.close()

        self.high_prices = high_prices
        self.high_base   = self.extreme_cutoffs[1]
        self.high_std    = float(sum(abs(high_prices - self.high_base)) / max(1, len(high_prices)))
        self.low_prices  = low_prices
        self.low_base    = self.extreme_cutoffs[0]
        self.low_std     = float(sum(abs(low_prices - self.low_base)) / max(1, len(low_prices)))

        residuals = prices.copy()
        residuals.loc[high_mask | low_mask, self.target_tag] = 0
        residuals.loc[high_mask, self.target_tag] += self.high_base + np.random.normal(0, 10, size=sum(high_mask))
        residuals.loc[low_mask, self.target_tag] += self.low_base + np.random.normal(0, 10, size=sum(low_mask))
        return residuals

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, old_model=None):
        residuals = self.data[[self.target_tag]]
        self.max_historical = float(np.max(residuals))
        self.min_historical = float(np.min(residuals))
        self.low_price_cutoff = np.quantile(residuals[self.target_tag], 0.02)
        self.low_price_dist = residuals.loc[residuals[self.target_tag] < self.low_price_cutoff][self.target_tag]

        if not(self.stochastic_model in ('SARIMAX', 'ARIMAX', 'GARCHX')):
            residuals, self.wss_model = self._wss_price_regression(residuals)
        residuals, self.trend_model  = self._del_trend(residuals)
        residuals, self.monthly_avg  = self._del_annual_cycle(residuals)
        residuals, self.weekday_avg, self.weekend_avg = self._del_weekday_and_weekend_pattern(residuals)

        if self.stochastic_model in ('SARIMAX', 'SARIMA', 'ARIMA', 'ARIMAX'):
            residuals = self._calculate_rs_probabilities(residuals)
            residuals.index.freq = 'h'
            if self.documentation:
                from data_scripts.analysis_functions import plot_acf
                self._investigate_heteroskedasticity(residuals)
                plot_acf(residuals)
            if self.stochastic_model in ('SARIMAX', 'ARIMAX'):
                self.feature_tags = self.main_exog_tags + self.other_exog_tags
                X_train = self.data[self.feature_tags]
            else:
                X_train = None
                # X_train = self.fourier_terms(residuals.index.hour, period=self.fourier_period, K=self.fourier_order)
            seasonality = (1,0,0,24) if self.stochastic_model in ('SARIMA', 'SARIMAX') else (0,0,0,0)
            self.arima_model = self._fit_arima_model(
                residuals, exog=X_train, order=(2, 0, 1), seasonal_order=seasonality,
                old_model=old_model.arima_model if old_model else None)
        elif self.stochastic_model in ('GARCH', 'GARCHX'):
            residuals = self._extreme_residuals_garch(residuals)
            
            if self.stochastic_model == 'GARCHX':
                self.feature_tags = self.main_exog_tags + self.other_exog_tags
                X_train = self.data[self.feature_tags]
            else:
                X_train = None
                # X_train = self.fourier_terms(residuals.index.hour, period=self.fourier_period, K=self.fourier_order)
            self.garch_model = self._fit_garch_model(
                residuals[self.target_tag],
                exog=X_train,
                old_model=old_model.garch_model if old_model else None,
            )
        self.residuals = residuals
        self.X_train = X_train

    def simulate(self, exog_profiles: pd.DataFrame) -> pd.DataFrame:
        time_info = self._specify_time_data(
            pd.DataFrame(index=pd.to_datetime(exog_profiles.index, utc=True)))
        return self._simulate(exog_profiles.copy(), time_info)

    def forecast(self, exog_profiles: pd.DataFrame) -> pd.DataFrame:
        time_info = self._specify_time_data(
            pd.DataFrame(index=pd.to_datetime(exog_profiles.index, utc=True)))
        return self._simulate(exog_profiles.copy(), time_info, forecasting=True)

    def realize(self, exog_profiles: pd.DataFrame) -> pd.DataFrame:
        time_info = self._specify_time_data(
            pd.DataFrame(index=pd.to_datetime(exog_profiles.index, utc=True)))
        return self._simulate(exog_profiles.copy(), time_info, realize=True)

    def _simulate(self, exog_profiles: pd.DataFrame, time_info: pd.DataFrame,
                  realize: bool = False, forecasting: bool = False) -> pd.DataFrame:
        horizon = len(exog_profiles)
        df = exog_profiles.copy()

        if self.stochastic_model in ('ARIMA', 'ARIMAX', 'SARIMA', 'SARIMAX'):
            if self.stochastic_model in ('ARIMA', 'SARIMA'):
                X = None
                # X = self.fourier_terms(df.index.hour, period=self.fourier_period, K=self.fourier_order)
            elif self.stochastic_model in ('ARIMAX', 'SARIMAX'):
                X = exog_profiles.copy()
            if forecasting:
                df[self.target_tag] = self._arima_forecast(horizon, exog=X)
            else:
                df['stoch_residuals'] = self._arima_simulate(horizon, exog=X, realize=realize)
                extreme_impact, normal = self._simulate_price_regimes(horizon, realize=realize)
                df[self.target_tag]    = df['stoch_residuals'] * normal + extreme_impact
        else:
            if self.stochastic_model == 'GARCH':
                X = None
                # X = self.fourier_terms(df.index.hour, period=self.fourier_period, K=self.fourier_order)
            elif self.stochastic_model == 'GARCHX':
                X = exog_profiles.copy()
            df[self.target_tag] = self._garch_simulate(horizon, exog=X, realize=realize, forecasting=forecasting)
            df = self._simulate_garch_extremes(df)

        # Daily patterns
        df.loc[time_info.is_weekend, self.target_tag] += (
            df.loc[time_info.is_weekend].index.hour.map(self.weekend_avg))
        df.loc[time_info.is_weekday, self.target_tag] += (
            df.loc[time_info.is_weekday].index.hour.map(self.weekday_avg))
        df[self.target_tag] += df.index.month.map(self.monthly_avg[self.target_tag])

        # Trend
        u_hours = pd.DataFrame(
            {'timestamp': [h.timestamp() - self.t_zero for h in df.index]}, index=df.index)
        df[self.target_tag] += self.trend_model.predict(u_hours)[:, 0]

        # Merit-order effect
        _tags = [self.wind_tag, self.solar_tag]
        if self.seasonal_price_regression or self.day_night_price_regression:
            df = self._create_seasonal_features(df.copy(), prod_columns=_tags)
        if self.log_vre:
            df[f'log_{self.wind_tag}']  = log_transform(df[self.wind_tag])
            df[f'log_{self.solar_tag}'] = log_transform(df[self.solar_tag])
        X = df[self.feature_tags]
        if not(self.stochastic_model in ('ARIMAX', 'GARCHX', 'SARIMAX')):
            df[self.target_tag] += self.wss_model.predict(X).reshape(-1)

        df[self.target_tag] = np.clip(df[self.target_tag],
                                     self.min_historical, self.max_historical)
        _low_price_filter = df[self.target_tag] < self.low_price_cutoff
        df.loc[_low_price_filter, self.target_tag] = np.random.choice(self.low_price_dist, size=sum(_low_price_filter))
        return df[[self.target_tag]]

    def _simulate_price_regimes(self, horizon: int = 8760, realize: bool = False):
        regimes = [np.random.choice(self.n_regimes,
                                    p=self.rs_prob_matrix[self.recent_data['regime']])]
        for _ in range(1, horizon):
            regimes.append(np.random.choice(self.n_regimes, p=self.rs_prob_matrix[regimes[-1]]))

        def _sample(r):
            if r == 1: return 0
            if r == 0: return laplace_rnd(self.low_base,  self.low_std,  np.random.uniform(-0.5, 0))
            return          laplace_rnd(self.high_base, self.high_std, np.random.uniform( 0.0, 0.5))

        extreme_prices = [_sample(r) for r in regimes]
        normal_regime  = np.asarray(regimes) == 1
        if realize:
            self.recent_data['regime'] = regimes[-1]
        return extreme_prices, normal_regime
    
    def _simulate_garch_extremes(self, df):
        high_residuals = df[self.target_tag] > self.high_base
        low_residuals = df[self.target_tag] < self.low_base
        df.loc[high_residuals, self.target_tag] = laplace_rnd(self.high_base,  self.high_std,  np.random.uniform(0, 0.48, size=sum(high_residuals)))
        df.loc[low_residuals, self.target_tag] = laplace_rnd(self.low_base,  self.low_std,  np.random.uniform(-0.48, 0, size=sum(low_residuals)))
        return df


# ---------------------------------------------------------------------------
# Renewables base class
# ---------------------------------------------------------------------------

class RenewablesSimulationTool(SimulationTool):
    """Base class for wind and solar capacity-factor simulation.

    Parameters
    ----------
    data : pd.DataFrame
        Historical hourly data containing at least the ``vre_tag`` column plus
        any time-feature columns required by subclasses (e.g. ``is_day``).
    caps : dict
        Mapping from tag name to a monthly ``pd.Series`` of installed capacity.
    vre_tag : str
        Column name for the variable renewable energy time series.
    weather_years : bool
        Whether to generate synthetic weather-year seasonality on simulation.
    """
    tool_type = 'renewable'
    ylabel    = 'MW'

    def __init__(self, data: pd.DataFrame, caps: dict, vre_tag: str,
                 weather_years: bool = True, **kwargs):
        super().__init__(target_tag=vre_tag, **kwargs)
        self.data                 = data
        self.caps                 = caps
        self.generate_weather_years = weather_years

    def _del_capacity_trend(self, df_: pd.DataFrame, tag: str = None) -> pd.DataFrame:
        """Normalise ``df_`` by installed capacity, producing capacity factors.

        Parameters
        ----------
        tag : str, optional
            Column to normalise. Defaults to ``self.target_tag``.
        """
        tag = tag or self.target_tag
        df  = df_.copy()
        ym  = df.index.tz_localize(None).to_period('M')
        yearly_caps = ym.map(self.caps[tag])
        df.loc[:, tag] = df[tag] / yearly_caps
        if df[tag].max() > 1.0:
            print(f'Warning: {tag} capacity factor > 1.0 detected in training data. '
                  'Falling back to per-month maximum normalisation.')
            df = df_.copy()
            yearly_max  = df.groupby(ym).max()
            yearly_caps = ym.map(yearly_max[tag])
            df.loc[:, tag] = df[tag] / yearly_caps
        return df

    def _simulate_cf(self, hourly_index: pd.DatetimeIndex, **kwargs):
        raise NotImplementedError

    def realize(self, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
        return self._simulate_cf(hourly_index, realize=True)

    def simulate(self, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
        return self._simulate_cf(hourly_index)

    def forecast(self, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
        return self._simulate_cf(hourly_index, forecasting=True)


# ---------------------------------------------------------------------------
# Solar simulation class
# ---------------------------------------------------------------------------

class SolarSimulationTool(RenewablesSimulationTool):
    """Simulates hourly solar capacity factors.

    Expects ``data`` to contain an ``is_day`` boolean column for daytime
    normalisation during fitting.
    """
    tool_type = 'solar'

    def fit(self, old_model=None):
        df = self._del_capacity_trend(self.data[[self.target_tag]].copy())

        # Monthly × hourly statistics
        self.hourly_monthly_mean_profiles = df.groupby([df.index.month, df.index.hour]).mean()
        self.hourly_monthly_max_profiles  = df.groupby([df.index.month, df.index.hour]).max()
        self.hourly_monthly_min_profiles  = df.groupby([df.index.month, df.index.hour]).min()
        self.hourly_monthly_std_profiles  = df.groupby([df.index.month, df.index.hour]).std()
        self.monthly_mean_max             = self.hourly_monthly_mean_profiles.groupby(level=0).max()

        if self.documentation:
            self._plot_monthly_profiles()

        # Daily maximum time series
        daily_max       = df.groupby(df.index.date).max()
        daily_max.index = pd.to_datetime(daily_max.index)
        daily_month_ix  = daily_max.index.month
        self.historical_monthly_max = daily_max.groupby(daily_month_ix).max()

        impact = daily_month_ix.map(self.monthly_mean_max[self.target_tag])
        daily_max[self.target_tag] = daily_max[self.target_tag] / impact
        self.mu_daily_max       = float(daily_max.mean().values[0])
        daily_max[self.target_tag] -= self.mu_daily_max

        self.monthly_std_of_max = daily_max.groupby(daily_month_ix).std()

        daily_std = daily_month_ix.map(self.monthly_std_of_max[self.target_tag])
        daily_max[self.target_tag] /= daily_std

        self.daily_residuals           = daily_max
        self.daily_residuals.index.freq = 'D'
        if self.documentation:
            from data_scripts.analysis_functions import plot_acf
            plot_acf(self.daily_residuals)

        # Hourly residuals
        is_day         = self.data.get('is_day', pd.Series(True, index=self.data.index))
        hour_residuals = df.copy()
        for month in hour_residuals.index.month.unique():
            daily_mean = self.hourly_monthly_mean_profiles.loc[month, self.target_tag]
            daily_std_ = self.hourly_monthly_std_profiles.loc[month, self.target_tag]
            m_mask     = hour_residuals.index.month == month
            hour_residuals.loc[m_mask, self.target_tag] -= (
                hour_residuals.loc[m_mask].index.hour.map(daily_mean))
            day_mask = m_mask & is_day
            hour_residuals.loc[day_mask, self.target_tag] /= (
                hour_residuals.loc[day_mask].index.hour.map(daily_std_))
        hour_residuals.index.freq = 'h'

        old_hourly = old_model.hourly_arima_model if old_model else None
        old_daily  = old_model.arima_model        if old_model else None
        self.hourly_arima_model = self._fit_arima_model(
            hour_residuals, order=(1, 0, 1), name='hour', old_model=old_hourly)
        self.arima_model = self._fit_arima_model(
            self.daily_residuals, order=(2, 0, 0), old_model=old_daily)

    def _simulate_cf(self, hourly_index: pd.DatetimeIndex,
                     realize: bool = False,
                     forecasting: bool = False) -> pd.DataFrame:
        day_index = pd.date_range(hourly_index[0], hourly_index[-1], freq='D')

        if forecasting:
            sim_daily_max = pd.DataFrame(
                index=day_index, data={self.target_tag: self._arima_forecast(len(day_index))})
        else:
            sim_daily_max = pd.DataFrame(
                index=day_index,
                data={self.target_tag: self._arima_simulate(len(day_index), realize=realize)})

        impact = sim_daily_max.index.month.map(self.monthly_std_of_max[self.target_tag])
        sim_daily_max[self.target_tag] *= impact
        sim_daily_max[self.target_tag] += self.mu_daily_max

        n_hours = len(hourly_index)
        if forecasting:
            sim_hourly_var = self.hourly_arima_model.forecast(steps=n_hours).values
        else:
            sim_hourly_var = self.hourly_arima_model.simulate(nsimulations=n_hours).values
            if realize:
                self.hourly_arima_model = self.hourly_arima_model.extend(sim_hourly_var)

        profile = pd.DataFrame(index=hourly_index, data={self.target_tag: sim_hourly_var})
        for month in profile.index.month.unique():
            h_means = self.hourly_monthly_mean_profiles.loc[month]
            h_stds  = self.hourly_monthly_std_profiles.loc[month]
            h_min   = self.hourly_monthly_min_profiles.loc[month]
            h_max   = self.hourly_monthly_max_profiles.loc[month]
            ix             = profile.index.month == month
            hours_of_month = profile.index[ix]
            day_profile = [
                np.clip(
                    h_means.loc[ts.hour, self.target_tag]
                    * sim_daily_max.loc[pd.to_datetime(ts.date(), utc=True), self.target_tag]
                    + profile.loc[ts, self.target_tag] * h_stds.loc[ts.hour, self.target_tag],
                    h_min.loc[ts.hour, self.target_tag],
                    h_max.loc[ts.hour, self.target_tag])
                for ts in hours_of_month]
            profile.loc[ix, self.target_tag] = day_profile

        if self.documentation:
            hist_cf = self._del_capacity_trend(self.data[[self.target_tag]])
            for yr in hist_cf.index.year.unique():
                plt.plot(np.sort(hist_cf.loc[hist_cf.index.year == yr, self.target_tag]),
                         color='blue')
            plt.plot(np.sort(profile[self.target_tag]), color='red')
            plt.savefig(f'documentation/{self.plot_dir}solar_load_duration_curves.png')
            plt.close()

        return profile

    def _plot_monthly_profiles(self):
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        fig, axs = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
        axs      = axs.flatten()
        axs[0].set_ylim(0, 1)
        for i, month in enumerate(range(1, 13)):
            ax   = axs[i]
            mean = self.hourly_monthly_mean_profiles.loc[month, self.target_tag].values
            std  = self.hourly_monthly_std_profiles.loc[month,  self.target_tag].values
            mx   = self.hourly_monthly_max_profiles.loc[month,  self.target_tag].values
            mn   = self.hourly_monthly_min_profiles.loc[month,  self.target_tag].values
            h    = np.arange(1, 25)
            ax.fill_between(h, mean - std, mean + std, color='lightgray', label='Mean ± Std')
            ax.plot(h, mean, color='blue',              label='Mean')
            ax.plot(h, mx,   '--', color='gray',        label='Min-Max')
            ax.plot(h, mn,   '--', color='gray')
            ax.set_title(month_names[month - 1])
            ax.set_xticks([6, 10, 15, 20])
        handles, labels = axs[-1].get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels,
                   loc='upper center', bbox_to_anchor=(-1, -0.3), ncol=3)
        plt.tight_layout()
        plt.savefig(f'documentation/{self.plot_dir}monthly_profiles.png')
        plt.close()


# ---------------------------------------------------------------------------
# Wind simulation class
# ---------------------------------------------------------------------------

class WindSimulationTool(RenewablesSimulationTool):
    tool_type = 'wind'
    hmm = False
    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    
    def fit(self, old_model=None):
        capacity_factors       = self._del_capacity_trend(self.data[[self.target_tag]].copy())

        self.min_historical_production = float(np.min(capacity_factors))
        self.max_historical_production = float(np.max(capacity_factors))
        
        residuals = self._deseasonalise(capacity_factors)

        if self.documentation:
            from data_scripts.analysis_functions import plot_acf
            plot_acf(residuals)

        prev = residuals.shift(1).fillna(residuals.iloc[0][self.target_tag])
        diff1 = residuals - prev

        if self.hmm: # Does not work
            self._fit_hmm(residuals, diff1)
        else:
            old_arima = old_model.arima_model if old_model is not None else None
            self.arima_model = self._fit_arima_model(diff1, order=(1, 0, 1), old_model=old_arima)
            self.ar_term = self.arima_model.params["ar.L1"]
            self.ma_term = self.arima_model.params["ma.L1"]

            diff_insample_predictions = self.arima_model.predict(start=diff1.index[0], end=diff1.index[-1])
            diff_residuals = diff1.copy()
            diff_residuals[self.target_tag] -= diff_insample_predictions # ARMA process residuals which we model with a Laplace distribution, level-dependent variance, and interval probabilities 
            
            diff = diff_residuals.copy()
            self.mu_laplace    = float(np.median(diff))
            self.sigma_laplace = float(sum(abs(diff[self.target_tag] - self.mu_laplace)) / len(diff))

            # max_lag = 7 * 24 # One week of hourly data
            # self._calculate_significant_ma_lag(residuals, diff, max_lag)
            self.ma_lag = 1
            self._create_exponential_models(residuals, diff)
            self._calculate_interval_probabilities(residuals, diff)

            self.quantile_cutoffs_deseason = residuals.groupby(residuals.index.month).quantile([0,0.07,0.95,1]) # Tuned quantiles through visual inspection of simulated vs historical data

            # Data for continuing simulation/forecast from end of training data
            self.recent_data['observations'] = residuals.iloc[-self.ma_lag:][self.target_tag].values
            self.recent_data['differences']  = diff1.iloc[-self.ma_lag:][self.target_tag].values
            self.recent_data['ma1_value'] = diff_residuals.iloc[-1][self.target_tag]

    def _deseasonalise(self, df: pd.DataFrame) -> pd.DataFrame:
        ym             = df.index.tz_localize(None).to_period('M')
        monthly_groups = df.groupby(ym)
        monthly_p90s   = monthly_groups.quantile(0.9)
        monthly_p10s   = monthly_groups.quantile(0.1)
        month_ix       = monthly_p90s.index
        annual_p90s    = monthly_p90s.groupby(month_ix.year).mean()
        annual_p10s    = monthly_p10s.groupby(month_ix.year).mean()

        # Utilize deseasonalization technique from Keles et al. (2013) "A combined modeling approach for wind power feed-in and electricity spot prices"
        self.monthly_stretch_factors = (
            month_ix.year.map((annual_p90s - annual_p10s)[self.target_tag])
            / (monthly_p90s - monthly_p10s)[self.target_tag])
        self.monthly_move_factors = (
            -monthly_p10s[self.target_tag] * self.monthly_stretch_factors.values
            + month_ix.year.map(annual_p10s[self.target_tag]))

        deseasoned_cf  = (df[self.target_tag] * ym.map(self.monthly_stretch_factors)
                          + ym.map(self.monthly_move_factors))
        
        # The daily average profile is heavily dependent on the season - negatively correlated with high sun in the summer.
        daily_profiles = deseasoned_cf.groupby(
            [deseasoned_cf.index.month, deseasoned_cf.index.hour]).mean()

        residuals = pd.DataFrame(index=deseasoned_cf.index, data={self.target_tag: deseasoned_cf})
        for month in df.index.month.unique():
            dp = daily_profiles.loc[month]
            residuals.loc[residuals.index.month == month, self.target_tag] -= (
                residuals.loc[df.index.month == month].index.hour.map(dp))

        self.daily_profiles        = daily_profiles
        self.avg_monthly_stretch  = self.monthly_stretch_factors.groupby(
            self.monthly_stretch_factors.index.month).mean()
        self.avg_monthly_move     = self.monthly_move_factors.groupby(
            self.monthly_move_factors.index.month).mean()
        self.std_monthly_stretch  = self.monthly_stretch_factors.groupby(
            self.monthly_stretch_factors.index.month).std()
        self.std_monthly_move     = self.monthly_move_factors.groupby(
            self.monthly_move_factors.index.month).std()
        self.avg_monthly_p90s     = monthly_p90s.groupby(month_ix.month).mean()
        self.avg_monthly_p10s     = monthly_p10s.groupby(month_ix.month).mean()
        return residuals

    # ------------------------------------------------------------------
    # Model identification helpers
    # ------------------------------------------------------------------

    def _calculate_significant_ma_lag(self, df: pd.DataFrame, diff, maxlag: int) -> int:
        # prev   = df.shift(1).fillna(df.iloc[0][self.target_tag])
        # diff   = df - prev
        r_list = np.zeros(maxlag)
        for lag in range(1, maxlag + 1):
            ma = df.rolling(window=lag).mean()
            x, y = ma[lag:], diff[lag:]
            r_list[lag - 1] = (np.corrcoef(x[self.target_tag].values, y[self.target_tag].values)[0, 1]
                               if len(x) > 1 else 0)
        if self.documentation:
            plt.plot(r_list)
            plt.xlabel('Lag')
            plt.ylabel('Correlation')
            plt.title('Correlation between moving average and first-order difference')
            plt.savefig(f'documentation/{self.plot_dir}wind_corr_ma_n_diff.png')
            plt.close()
        self.ma_lag = int(np.argmax(np.abs(r_list))) + 1

    def _create_exponential_models(self, df: pd.DataFrame, diff):
        from numpy.polynomial import Polynomial

        intervals = np.arange(min(df[self.target_tag]), max(df[self.target_tag]),
                              (max(df[self.target_tag]) - min(df[self.target_tag])) / 20)
        centres       = np.zeros(len(intervals) - 1)
        centres[0]    = min(df[self.target_tag])
        centres[-1]   = max(df[self.target_tag])
        for l in range(1, len(centres) - 1):
            centres[l] = (intervals[l + 1] + intervals[l]) / 2

        ma   = df.rolling(window=self.ma_lag).mean()
        ma   = ma[self.ma_lag - 1:]
        diffy = diff[self.ma_lag - 1:].copy()

        pol_model_mode = Polynomial.fit(x=ma[self.target_tag], y=diffy[self.target_tag], deg=1).convert()

        diff_neg = diffy.loc[pol_model_mode(ma[self.target_tag]) > diffy[self.target_tag]]
        ma_neg   = ma.loc[diff_neg.index]
        diff_neg = diff_neg.copy()
        diff_neg.loc[:, self.target_tag] -= pol_model_mode(ma_neg[self.target_tag])
        pol_model_neg = Polynomial.fit(
            x=ma_neg[self.target_tag], y=-diff_neg[self.target_tag], deg=2).convert()

        diff_pos = diffy.loc[pol_model_mode(ma[self.target_tag]) < diffy[self.target_tag]]
        ma_pos   = ma.loc[diff_pos.index]
        diff_pos = diff_pos.copy()
        diff_pos.loc[:, self.target_tag] -= pol_model_mode(ma_pos[self.target_tag])
        pol_model_pos = Polynomial.fit(
            x=ma_pos[self.target_tag], y=diff_pos[self.target_tag], deg=2).convert()

        self.pol_model_pos, self.pol_model_neg, self.pol_model_mode = pol_model_pos, pol_model_neg, pol_model_mode

        if self.documentation:
            plt.scatter(ma_pos[self.target_tag], diff_pos[self.target_tag], color='black', s=1, alpha=0.2, label="Observations")
            plt.plot(centres, pol_model_pos(centres), color= "green", label = "Bracket Fit - weighted fit")
            plt.axhline(self.sigma_laplace, color='orange', linestyle='--', label='Mean Absolute Deviation')
            plt.xlim(intervals[0],intervals[-1])
            plt.legend()
            plt.savefig(f'documentation/{self.plot_dir}exp_model_fit_positives.png')
            plt.close()

            plt.scatter(ma_neg[self.target_tag], -diff_neg[self.target_tag], color='black', s=1, alpha=0.2, label="Observations")
            plt.plot(centres, pol_model_neg(centres), color= "green", label = "Bracket Fit - weighted fit")
            plt.axhline(self.sigma_laplace, color='orange', linestyle='--', label='Mean Absolute Deviation')
            plt.xlim(intervals[0],intervals[-1])
            plt.legend()
            plt.savefig(f'documentation/{self.plot_dir}exp_model_fit_negatives.png')
            plt.close()
            
            plt.scatter(ma[self.target_tag], diff[self.target_tag], color='black', s=1, alpha=0.2, label="Observations")
            plt.plot(centres, pol_model_mode(centres), color= "green", label = "Bracket Fit - weighted fit")
            plt.xlim(intervals[0],intervals[-1])
            plt.legend()
            plt.savefig(f'documentation/{self.plot_dir}exp_model_fit_modevalues.png')
            plt.close()

    def _calculate_interval_probabilities(self, df: pd.DataFrame, diff):
        # prev = df.shift(1).fillna(df.iloc[0][self.target_tag])
        # diff = df - prev

        max_len     = 24 * 3
        self.domains = np.quantile(df, [0, 0.1, 0.25, 0.75, 0.9])
        self.domains[0] = -np.inf
        n_domains   = len(self.domains)

        posIntLengthDist = np.zeros((max_len, n_domains, 3))
        negIntLengthDist = np.zeros((max_len, n_domains, 3))

        posCount = negCount = 0
        domain   = int(np.argwhere(df[self.target_tag].iloc[0] >= self.domains)[-1])

        for t in range(1, len(df)):
            delta     = diff[self.target_tag].iloc[t]
            indicator = 1 if delta >= 0 else -1
            if delta >= 0:
                posCount += 1
            else:
                negCount += 1

            if posCount > 0 and negCount > 0:
                if indicator < 0:
                    posIntLengthDist[posCount, domain, 0]  = posCount
                    posIntLengthDist[posCount, domain, 1] += 1
                    posCount = 0
                else:
                    negIntLengthDist[negCount, domain, 0]  = negCount
                    negIntLengthDist[negCount, domain, 1] += 1
                    negCount = 0
                domain = int(np.argwhere(df[self.target_tag].iloc[t - 1] >= self.domains)[-1])

        pos_totals = np.sum(posIntLengthDist[:, :, 1], axis=0)
        neg_totals = np.sum(negIntLengthDist[:, :, 1], axis=0)
        posIntLengthDist[:, :, 2] = np.divide(posIntLengthDist[:, :, 1], pos_totals)
        negIntLengthDist[:, :, 2] = np.divide(negIntLengthDist[:, :, 1], neg_totals)

        max_len_obs = int(max(
            list(posIntLengthDist[:, :, 0].flatten()) +
            list(negIntLengthDist[:, :, 0].flatten())))
        posIntLengthDists = posIntLengthDist[:max_len_obs + 1, :, 2]
        negIntLengthDists = negIntLengthDist[:max_len_obs + 1, :, 2]

        if self.documentation:
            fig, axs = plt.subplots(2, 1, sharex=True)
            for dom in range(n_domains):
                axs[0].plot(posIntLengthDists[:, dom] * 100,
                            label=f'Domain {dom + 1}', alpha=0.6)
                axs[1].plot(negIntLengthDists[:, dom] * 100,
                            label=f'Domain {dom + 1}', alpha=0.6)
            axs[0].plot(np.sum(posIntLengthDist[:max_len_obs + 1, :, 2] * 100, axis=1) / n_domains,
                        label='All domains', color='black')
            axs[1].plot(np.sum(negIntLengthDist[:max_len_obs + 1, :, 2] * 100, axis=1) / n_domains,
                        label='All domains', color='black')
            axs[0].legend(title='Increasing wind period probabilities',
                          loc='upper right', fontsize='small', ncols=2)
            axs[1].legend(title='Decreasing wind period probabilities',
                          loc='upper right', fontsize='small', ncols=2)
            axs[1].set_xlabel('Interval Length (hours)')
            axs[0].set_ylabel('Probability (%)')
            axs[1].set_ylabel('Probability (%)')
            axs[0].set_xlim(1, max_len_obs)
            plt.savefig(f'documentation/{self.plot_dir}difference_streak_probability_domains.png')
            plt.close()

        self.pos_int_length_distributions, self.neg_int_length_distributions = posIntLengthDists, negIntLengthDists

    def _fit_hmm(self, df: pd.DataFrame, diff, n_states=50, ):
        X = diff.copy()
        X.loc[:,'delta_lag1'] = diff.shift(1).fillna(0)
        X.loc[:,'delta_lag2'] = diff.shift(2).fillna(0)
        X.loc[:,'lag1'] = df.shift(1).fillna(df.iloc[0] * 0.9)
        X.loc[:,'lag24'] = df.shift(24)
        X.loc[X.index[0:24],'lag24'] = df.iloc[0:24][self.target_tag] * 0.3
        # The state is (delta_w(t-1), delta_w(t-24), wind(t-1), wind(t-24))
        # So we get a sarimax transition: delta_w(t) = fn((delta_w(t-1), delta_w(t-24), wind(t-1), wind(t-24)))
        X = X[['delta_lag1', 'delta_lag2', 'lag1', 'lag24']]
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        # Use scaler.partial_fit for incremental updates when seeing new observations.

        cache_path = os.getcwd() + "/models/ts_models/hmm_models/" + str(self.cache_id) + ".pkl"
        if self.cache_id is not None and not(self.cache_replace) and cache_exists(cache_path):
            hmm_model = cache_read(cache_path)
        else:
            if self.verbose: print(f"Fitting HMM model on {len(X_scaled)} observations...")
            t_start = time()
            hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type = "diag", n_iter = 50) # 620 seconds with 50 states and 50 iter and diag, 10 years of data.
            hmm_model.fit(X_scaled)
            if self.verbose: print(f"HMM model fit done in {time()-t_start} seconds")
            if self.cache_id is not None: cache_write(hmm_model, cache_path, verbose=self.verbose)
        # Data for continuing simulation/forecast from end of training data
        self.recent_data["state"] = X_scaled[-24:,:]
        self.hmm_model, self.scaler = hmm_model, scaler

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _stochastic_process_simulation(self, hourly_index:pd.DatetimeIndex,
                                       realize=False,
                                       forecasting=False) -> tuple:
        horizon = len(hourly_index)
        sim_cf     = np.zeros(horizon + self.ma_lag)
        deltas     = np.zeros(horizon + self.ma_lag)
        directions = np.zeros(horizon + self.ma_lag)

        sim_cf[:self.ma_lag]     = self.recent_data['observations']
        deltas[:self.ma_lag]     = self.recent_data['differences']
        directions[:self.ma_lag] = np.sign(self.recent_data['differences']).astype(int)
        prev_dir = int(directions[self.ma_lag - 1])

        latest_ma_residual = self.recent_data['ma1_value']

        for t in range(self.ma_lag, horizon + self.ma_lag):
            month = hourly_index[t - self.ma_lag].month
            min_obs, p10_obs, p95_obs, max_obs = self.quantile_cutoffs_deseason.loc[month, self.target_tag].values
            if (directions[t] == 0
                    or (sim_cf[t - 1] <= p10_obs  and directions[t] == -1)
                    or (sim_cf[t - 1] >= p95_obs and directions[t] ==  1)):
                domain = int(np.argwhere(self.domains <= sim_cf[t - 1])[-1][0])
                p = (self.neg_int_length_distributions[:, domain]
                     if prev_dir == 1
                     else self.pos_int_length_distributions[:, domain])
                interval_length = np.random.choice(range(len(p)), p=p)
                directions[t:min(t + interval_length, horizon)] = -prev_dir
                prev_dir *= -1
            
            moving_avg = np.mean(sim_cf[t - self.ma_lag:t])
            direction  = directions[t]
            mean_scale = (self.pol_model_pos(moving_avg)
                          if direction == 1
                          else self.pol_model_neg(moving_avg))
            delta_residual = self.pol_model_mode(moving_avg) + direction * np.random.exponential(max(self.sigma_laplace / 5, mean_scale)) * (1 - forecasting)
            delta = self.ar_term * deltas[t - 1] + self.ma_term * latest_ma_residual + delta_residual
            latest_ma_residual = delta_residual
            sim_cf[t]  = max(min_obs, min(max_obs, sim_cf[t - 1] + delta))
            deltas[t]  = sim_cf[t] - sim_cf[t - 1]
        
        if realize:
            self.recent_data["observations"] = sim_cf[-self.ma_lag:]
            self.recent_data["differences"]  = deltas[-self.ma_lag:]
            self.recent_data["ma1_value"]    = latest_ma_residual

        return sim_cf[self.ma_lag:]

    def _stochastic_hmm_simulation(self, hourly_index:pd.DatetimeIndex,
                                       realize=False,
                                       forecasting=False) -> tuple:
        horizon = len(hourly_index)
        earlier_states = self.recent_data["state"]
        delta_w, delta_w24, w, w24 = earlier_states[-1]
        states = np.zeros([horizon+1, len(currstate)])

        ### HMM simulation ###
        # Actual simulation time steps:
        for t in range(23):
            currstate = np.asarray([delta_w, delta_w24, w, w24])
            states[t] = currstate
            currstate_ix = np.argmin(np.sum((self.hmm_model.means_ - currstate)**2 , axis=1))
            nextstate, nextstate_ix = self.hmm_model.sample(n_samples = 1, currstate = currstate_ix)
            nextstate = nextstate[0]
            delta_w = nextstate[0]
            delta_w24 = earlier_states[t-23][0]
            w = currstate[2] + delta_w
            w24 = earlier_states[t-23][2]
        
        for t in range(23, horizon+1):
            currstate[1] = states[t-23][0]
            currstate[3] = states[t-23][2]
            states[t] = currstate
            currstate_ix = np.argmin(np.sum((self.hmm_model.means_ - currstate)**2 , axis=1))
            nextstate, nextstate_ix = self.hmm_model.sample(n_samples = 1, currstate = currstate_ix) # (delta_w(t), delta_w(t-23), wind(t), wind(t-23)) = fn((delta_w(t-1), delta_w(t-24), wind(t-1), wind(t-24)))
            nextstate = nextstate[0]
            delta_w = nextstate[0]
            w = currstate[2] + delta_w
            currstate = np.asarray([delta_w, 0, w, 0])

        sim_wind = self.scaler.inverse_transform(states) # Remove scaling
        sim_cf = sim_wind[1:,2]
        if realize:
            self.recent_data["state"] = states[-24:,:]
        
        return sim_cf

    def _simulate_cf(self, hourly_index: pd.DatetimeIndex,
                     realize: bool = False,
                     forecasting: bool = False) -> pd.DataFrame:
        if self.hmm:
            self._stochastic_hmm_simulation(hourly_index, realize=realize, forecasting=forecasting)
        else:
            sim_cf = self._stochastic_process_simulation(hourly_index, realize=realize, forecasting=forecasting)

        profile = pd.DataFrame(index=hourly_index, data={self.target_tag: sim_cf})

        for month in profile.index.month.unique():
            dp = self.daily_profiles.loc[month]
            profile.loc[profile.index.month == month, self.target_tag] += (
                profile.loc[profile.index.month == month].index.hour.map(dp))

        # Monthly re-seasonalisation
        if self.generate_weather_years:
            monthly_stretch = np.clip(
                np.random.normal(loc=self.avg_monthly_stretch, scale=self.std_monthly_stretch),
                self.avg_monthly_stretch - 2 * self.std_monthly_stretch,
                self.avg_monthly_stretch + 2 * self.std_monthly_stretch)
            monthly_move = np.clip(
                np.random.normal(loc=self.avg_monthly_move, scale=self.std_monthly_move),
                self.avg_monthly_move - 2 * self.std_monthly_move,
                self.avg_monthly_move + 2 * self.std_monthly_move)
            profile[self.target_tag] = (
                (profile[self.target_tag] - profile.index.month.map(monthly_move))
                / profile.index.month.map(monthly_stretch))
        else:
            profile[self.target_tag] = (
                (profile[self.target_tag] - profile.index.month.map(self.avg_monthly_move))
                / profile.index.month.map(self.avg_monthly_stretch))

        # Soft clip to historical range
        over  = profile[self.target_tag] > self.max_historical_production
        under = profile[self.target_tag] < self.min_historical_production
        profile.loc[over, self.target_tag] += (
            (self.max_historical_production - profile.loc[over, self.target_tag])
            - np.random.exponential(abs(self.pol_model_neg(self.max_historical_production)),
                                    size=over.sum()))
        profile.loc[under, self.target_tag] += (
            (self.min_historical_production - profile.loc[under, self.target_tag])
            + np.random.exponential(abs(self.pol_model_pos(self.min_historical_production)),
                                    size=under.sum()))

        if self.documentation:
            hist_cf = self._del_capacity_trend(self.data[[self.target_tag]])
            for yr in hist_cf.index.year.unique():
                plt.plot(np.sort(hist_cf.loc[hist_cf.index.year == yr, self.target_tag]),
                         color='blue')
            for yr in profile.index.year.unique():
                plt.plot(np.sort(profile.loc[profile.index.year == yr, self.target_tag]),
                         color='red')
            plt.savefig(f'documentation/{self.plot_dir}wind_load_duration_curves.png')
            plt.close()

        return profile

        return self._simulate_cf(hourly_index, realize=True)


# ---------------------------------------------------------------------------
# Load forecaster class
# ---------------------------------------------------------------------------

class LoadForecaster(SimulationTool):
    def __init__(self,
                 data: pd.DataFrame,
                 load_tag: str,
                 specify_time_data=None,
                 t_zero: float = None,
                 **kwargs,
                 ):
        super().__init__(target_tag=load_tag, **kwargs)
        self.data        = data
        self.target_data = data[[load_tag]]
        self._specify_time_data = specify_time_data or self._default_time_data
        self.t_zero      = t_zero if t_zero is not None else data.index[0].timestamp()

    def fit(self):
        residuals = self.data[[self.target_tag]]
        residuals, self.trend_model  = self._del_trend(residuals)
        residuals, self.monthly_avg  = self._del_annual_cycle(residuals)
        residuals, self.weekday_avg, self.weekend_avg = self._del_weekday_and_weekend_pattern(residuals)

    def _simulate(self, time_info: pd.DataFrame, realize: bool = False, forecasting: bool = False) -> pd.DataFrame:
        horizon = len(time_info)
        if horizon <= 168: # For short horizons, provide naive weekly forecasts, utilizing self.data.
            df = pd.DataFrame(index=time_info.index, data={self.target_tag: np.zeros(horizon)})
            for i in range(horizon):
                timestamp = time_info.index[i]
                past_timestamp = timestamp - pd.Timedelta(7, 'days')
                if past_timestamp in self.data.index:
                    df.loc[timestamp, self.target_tag] = self.data.loc[past_timestamp, self.target_tag]
                else:
                    df.loc[timestamp, self.target_tag] = self.data[self.target_tag].iloc[-1]
        else: 
            df = pd.DataFrame(index=time_info.index, data={self.target_tag: np.zeros(horizon)})
            
            # Daily patterns
            df.loc[time_info.is_weekend, self.target_tag] += (
                df.loc[time_info.is_weekend].index.hour.map(self.weekend_avg))
            df.loc[time_info.is_weekday, self.target_tag] += (
                df.loc[time_info.is_weekday].index.hour.map(self.weekday_avg))
            
            # Monthly pattern
            df[self.target_tag] += df.index.month.map(self.monthly_avg[self.target_tag])

            # Trend
            u_hours = pd.DataFrame(
                {'timestamp': [h.timestamp() - self.t_zero for h in df.index]}, index=df.index)
            df[self.target_tag] += self.trend_model.predict(u_hours)[:, 0]    

        return df[[self.target_tag]]

    def simulate(self, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
        time_info = self._specify_time_data(
            pd.DataFrame(index=pd.to_datetime(hourly_index, utc=True)))
        return self._simulate(time_info)

    def forecast(self, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
        time_info = self._specify_time_data(
            pd.DataFrame(index=pd.to_datetime(hourly_index, utc=True)))
        return self._simulate(time_info, forecasting=True)

    def realize(self, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
        time_info = self._specify_time_data(
            pd.DataFrame(index=pd.to_datetime(hourly_index, utc=True)))
        return self._simulate(time_info, realize=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DataForecaster:
    """ Orchestrates solar, wind, and price simulation tools.

    Wraps a ``DataLoader`` database and constructs the three simulation tools
    with all required parameters injected explicitly.
    """
    _tags = RenewableFuelPlant.uncertainties

    def __init__(self, database: DataLoader, *,
                 price_tag: str = 'price',
                 wind_tag:  str = 'wind',
                 solar_tag: str = 'solar',
                 load_tag: str = 'Actual Load',
                 r_load_tag: str = "",
                 other_exog_tags: list = [],
                 exog_price_model: type = LinearRegression(),
                 stochastic_price_model: str = "ARIMA",
                 log_prices: bool = False,
                 log_vre:    bool = False,
                 documentation: bool = False,
                 seasonal_price_regression:  bool = False,
                 day_night_price_regression: bool = False,
                 weather_years: bool = True,
                 verbose:  bool = True,
                 plot_dir: str  = '',
                 cache_id       = None,
                 cache_replace: bool = False):
        self.database   = database
        self.data       = database.data
        self.price_tag  = price_tag
        self.wind_tag   = wind_tag
        self.solar_tag  = solar_tag
        self.load_tag   = load_tag
        self.main_exog_tags = [self.solar_tag, self.wind_tag, self.load_tag]
        self.r_load_tag = r_load_tag
        if r_load_tag:
            self.data[self.r_load_tag] = self.data[self.load_tag] - self.data[self.solar_tag] - self.data[self.wind_tag]
            self.main_exog_tags += [self.r_load_tag]
        self.other_exog_tags = other_exog_tags
        self.exog_price_model = exog_price_model
        self.stochastic_price_model = stochastic_price_model
        self.log_prices = log_prices
        self.log_vre    = log_vre
        self.seasonal_price_regression  = seasonal_price_regression
        self.day_night_price_regression = day_night_price_regression
        self.weather_years  = weather_years
        self.verbose        = verbose
        self.plot_dir       = (plot_dir.rstrip('/') + '/') if plot_dir else ''
        self.documentation  = documentation or bool(plot_dir)
        self.cache_id       = cache_id
        self.cache_replace  = cache_replace
        self.t_zero         = self.data.index[0].timestamp()
        self.t_init         = self.data.index[-1] + pd.Timedelta(1, 'hour')
        self.solar_realization_cf = None
        self.wind_realization_cf  = None

        doc_dir = os.path.dirname(os.getcwd() + '/documentation/' + self.plot_dir)
        if self.plot_dir and not os.path.exists(doc_dir):
            os.mkdir(doc_dir)

    @classmethod
    def from_pickle(cls, cache_id, documentation: bool = False):
        """Load a previously pickled forecaster."""
        cache_path = os.getcwd() + '/models/ts_models/forecaster/' + str(cache_id) + '.pkl'
        if not cache_exists(cache_path):
            raise FileNotFoundError(f'No cached forecaster found for cache_id={cache_id!r}.')
        obj = cache_read(cache_path)
        obj.solar_model.documentation = documentation
        obj.wind_model.documentation  = documentation
        obj.price_model.documentation = documentation
        obj.documentation             = documentation
        return obj

    @property
    def _tool_kwargs(self) -> dict:
        """Common keyword arguments shared by all simulation tools."""
        return dict(
            documentation=self.documentation,
            cache_id=self.cache_id,
            cache_replace=self.cache_replace,
            verbose=self.verbose,
            plot_dir=self.plot_dir.rstrip('/'),
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_simulation_models(self, old_forecaster=None, to_pickle: bool = False):
        old_solar = old_forecaster.solar_model if old_forecaster else None
        old_price = old_forecaster.price_model if old_forecaster else None
        old_wind = old_forecaster.wind_model if old_forecaster else None

        self.solar_model = SolarSimulationTool(
            data=self.data, caps=self.database.caps, vre_tag=self.solar_tag,
            weather_years=False, **self._tool_kwargs)
        self.solar_model.fit(old_solar)

        self.wind_model = WindSimulationTool(
            data=self.data, caps=self.database.caps, vre_tag=self.wind_tag,
            weather_years=self.weather_years, **self._tool_kwargs)
        self.wind_model.fit(old_wind)
        
        self.load_model = LoadForecaster(
            data=self.data, load_tag=self.load_tag,
            specify_time_data=self.database._specify_time_data,
            t_zero=self.t_zero, **self._tool_kwargs)
        self.load_model.fit()

        self.price_model = PriceSimulationTool(
            data=self.data, price_tag=self.price_tag,
            wind_tag=self.wind_tag, solar_tag=self.solar_tag,
            main_exog_tags=self.main_exog_tags,
            other_exog_tags=self.other_exog_tags,
            log_prices=self.log_prices, log_vre=self.log_vre,
            seasonal_price_regression=self.seasonal_price_regression,
            day_night_price_regression=self.day_night_price_regression,
            t_zero=self.t_zero,
            specify_time_data=self.database._specify_time_data,
            create_seasonal_features=self.database._create_seasonal_features,
            exog_model=self.exog_price_model,
            stochastic_model=self.stochastic_price_model,
            **self._tool_kwargs)
        self.price_model.fit(old_price)

        self.exog_data = self.data[self.other_exog_tags].copy() if self.other_exog_tags else None

        if to_pickle and self.cache_id is not None:
            cache_path = (os.getcwd() + '/models/ts_models/forecaster/'
                          + str(self.cache_id) + '.pkl')
            cache_write(self, cache_path, verbose=self.verbose)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, year: int, n_sims: int = 1) -> list:
        hourly_index     = pd.to_datetime(
            pd.date_range(str(year), str(year + 1), freq='h'), utc=True)[:-1]
        ym               = hourly_index.tz_localize(None).to_period('M')
        solar_caps       = ym.map(self.database.caps[self.solar_tag])
        wind_caps        = ym.map(self.database.caps[self.wind_tag])
        sims = []
        for _ in tqdm(range(n_sims), disable=not self.verbose):
            solar_cf    = self.solar_model.simulate(hourly_index)
            wind_cf     = self.wind_model.simulate(hourly_index)
            solar_sim   = solar_cf[self.solar_tag] * solar_caps
            wind_sim    = wind_cf[self.wind_tag]   * wind_caps
            exog         = pd.DataFrame(index=hourly_index,
                                       data={self.solar_tag: solar_sim,
                                             self.wind_tag:  wind_sim})
            exog[self.load_tag] = self.load_model.simulate(hourly_index)
            if self.r_load_tag:
                exog[self.r_load_tag] = exog[self.load_tag] - exog[self.solar_tag] - exog[self.wind_tag]
            exog = self._add_exog_naive_forecast(exog)
            price_sim   = self.price_model.simulate(exog)[self.price_tag]
            sims.append({self.wind_tag:  wind_sim,
                         self.solar_tag: solar_sim,
                         self.price_tag: price_sim})
        return sims

    def simulate_year_ahead(self, start: pd.Timestamp, n_sims: int = 3,
                            deterministic: bool = False):
        end          = start + relativedelta(years=+1) - pd.Timedelta(1, 'hour')
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'), utc=True)
        ym           = hourly_index.tz_localize(None).to_period('M')
        solar_caps   = ym.map(self.database.caps[self.solar_tag])
        wind_caps    = ym.map(self.database.caps[self.wind_tag])

        t0   = time()
        sims = [self._single_simulation(hourly_index, solar_caps, wind_caps)
                for _ in range(n_sims)]
        print(f'Simulated {n_sims} year-aheads in {time() - t0:.1f} s.')

        if deterministic:
            return self._average_simulations(sims, hourly_index)
        return sims

    def simulate_period(self, start: pd.Timestamp, end: pd.Timestamp,
                        n_sims: int = 1) -> list:
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'), utc=True)
        ym           = hourly_index.tz_localize(None).to_period('M')
        solar_caps   = ym.map(self.database.caps[self.solar_tag])
        wind_caps    = ym.map(self.database.caps[self.wind_tag])
        t0   = time()
        sims = [self._single_simulation(hourly_index, solar_caps, wind_caps)
                for _ in range(n_sims)]
        print(f'Simulated {n_sims} periods in {time() - t0:.1f} s.')
        return sims

    def _single_simulation(self, hourly_index, solar_capacities, wind_capacities) -> pd.DataFrame:
        solar_cf  = self.solar_model._simulate_cf(hourly_index)
        wind_cf   = self.wind_model._simulate_cf(hourly_index)
        exog       = pd.DataFrame(
            index=hourly_index,
            data={self.solar_tag: solar_cf[self.solar_tag] * solar_capacities.values,
                  self.wind_tag:  wind_cf[self.wind_tag]   * wind_capacities.values})
        exog[self.load_tag] = self.load_model.simulate(hourly_index)
        if self.r_load_tag:
            exog[self.r_load_tag] = exog[self.load_tag] - exog[self.solar_tag] - exog[self.wind_tag]
        exog = self._add_exog_naive_forecast(exog)
        prices    = self.price_model.simulate(exog)
        df        = exog.copy()
        df[self.price_tag] = prices[self.price_tag]
        df[self.solar_tag] = solar_cf[self.solar_tag]
        df[self.wind_tag]  = wind_cf[self.wind_tag]
        return df[self._tags]

    def _average_simulations(self, sims: list, hourly_index) -> pd.DataFrame:
        avg = lambda tag: np.mean([df[tag].values for df in sims], axis=0)
        return pd.DataFrame(index=hourly_index,
                            data={self.price_tag: avg(self.price_tag),
                                  self.solar_tag: avg(self.solar_tag),
                                  self.wind_tag:  avg(self.wind_tag)})[self._tags]

    def _add_exog_naive_forecast(self, exog: pd.DataFrame) -> pd.DataFrame:
        horizon = len(exog)
        if self.exog_data is None:
            return exog
        for tag in self.other_exog_tags:
            forecast = np.zeros(horizon)
            for t in range(horizon):
                forecast[t] = np.mean(self.exog_data[tag].iloc[-t-1:])
            exog[tag] = forecast # Naive forecast: mean of last observed values
        return exog

    # ------------------------------------------------------------------
    # Realization
    # ------------------------------------------------------------------

    def realize_vre(self, start: pd.Timestamp, end: pd.Timestamp):
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'), utc=True)
        self.solar_realization_cf = self.solar_model.realize(hourly_index)
        self.wind_realization_cf  = self.wind_model.realize(
            hourly_index, solar_profile=self.solar_realization_cf)
        return self.solar_realization_cf.copy(), self.wind_realization_cf.copy()

    def realize_prices(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if self.solar_realization_cf is None or self.wind_realization_cf is None:
            raise RuntimeError('Call realize_vre() before realize_prices().')
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'), utc=True)
        assert (hourly_index == self.solar_realization_cf.index).all()
        ym         = hourly_index.tz_localize(None).to_period('M')
        solar_caps = ym.map(self.database.caps[self.solar_tag])
        wind_caps  = ym.map(self.database.caps[self.wind_tag])
        exog = pd.DataFrame(
            index=hourly_index,
            data={self.solar_tag: self.solar_realization_cf[self.solar_tag] * solar_caps.values,
                  self.wind_tag:  self.wind_realization_cf[self.wind_tag]   * wind_caps.values})
        exog[self.load_tag] = self.load_model.simulate(hourly_index)
        if self.r_load_tag:
            exog[self.r_load_tag] = exog[self.load_tag] - exog[self.solar_tag] - exog[self.wind_tag]
        exog = self._add_exog_naive_forecast(exog)
        return self.price_model.realize(exog)

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, start: pd.Timestamp, end: pd.Timestamp,
                 n_forecasts: int = 10, deterministic: bool = False,
                 simulate_prices: bool = False) -> list:
        hourly_index = pd.to_datetime(pd.date_range(start, end, freq='h'), utc=True)
        ym           = hourly_index.tz_localize(None).to_period('M')
        solar_caps   = ym.map(self.database.caps[self.solar_tag])
        wind_caps    = ym.map(self.database.caps[self.wind_tag])

        if self.solar_realization_cf is not None:
            vre_forecast_index = pd.to_datetime(
                pd.date_range(self.solar_realization_cf.index[-1] + pd.Timedelta(1, 'hour'),
                              end, freq='h'), utc=True)
        else:
            vre_forecast_index = hourly_index

        forecasts = []
        for ix in range(n_forecasts):
            df = pd.DataFrame(index=hourly_index,
                              columns=[self.price_tag, self.solar_tag, self.wind_tag])

            if len(vre_forecast_index) > 0:
                solar_cf = (self.solar_model.simulate(vre_forecast_index) if ix > 0
                            else self.solar_model.forecast(vre_forecast_index))
                wind_cf  = (self.wind_model.simulate(vre_forecast_index)
                            if ix > 0
                            else self.wind_model.forecast(vre_forecast_index))
                solar_cf = solar_cf[self.solar_tag]
                wind_cf  = wind_cf[self.wind_tag]
            else:
                solar_cf = wind_cf = None

            if self.solar_realization_cf is not None and solar_cf is not None:
                solar_cf = pd.concat([self.solar_realization_cf[self.solar_tag], solar_cf])
                wind_cf  = pd.concat([self.wind_realization_cf[self.wind_tag],   wind_cf])

            solar_prod = solar_cf * solar_caps.values
            wind_prod  = wind_cf  * wind_caps.values
            df.loc[hourly_index, self.solar_tag] = solar_cf
            df.loc[hourly_index, self.wind_tag]  = wind_cf

            exog = pd.DataFrame(index=hourly_index,
                               data={self.solar_tag: solar_prod.values,
                                     self.wind_tag:  wind_prod.values})
            exog[self.load_tag] = self.load_model.simulate(hourly_index)
            if self.r_load_tag:
                exog[self.r_load_tag] = exog[self.load_tag] - exog[self.solar_tag] - exog[self.wind_tag]
            exog = self._add_exog_naive_forecast(exog)
            price_forecast = (self.price_model.simulate(exog)
                              if simulate_prices and ix > 0
                              else self.price_model.forecast(exog))
            df.loc[hourly_index, self.price_tag] = price_forecast[self.price_tag]
            forecasts.append(df)

        if deterministic:
            return self._average_simulations(forecasts, hourly_index)
        return forecasts

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def investigate_test_simulation_monthly(self, simulations: list, resource: str = 'price'):
        real_data      = self.data[resource]
        simulated_data = [sim[resource] for sim in simulations]
        ylabel         = '[€/MWh]' if resource == 'price' else 'MW'

        fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharey=True)
        axes = axes.flatten()
        plt.tight_layout(pad=4.0, rect=[0.03, 0.03, 0.97, 0.95])

        for i, month in enumerate(range(1, 13)):
            monthly_real = real_data[real_data.index.month == month]
            sorted_real  = np.sort(monthly_real.values)
            mean_real    = float(np.mean(monthly_real.values))
            sim_means    = [float(np.mean(s[s.index.month == month].values))
                            for s in simulated_data]
            ax  = axes[i]
            mtx = np.asarray([np.sort(s.loc[s.index.month == month].values).reshape(-1)
                               for s in simulated_data])
            ax.fill_between(range(len(mtx[0])),
                            np.percentile(mtx, 5, axis=0),
                            np.percentile(mtx, 95, axis=0),
                            color='blue', alpha=0.2, label='90% CI')
            ax.plot(sorted_real, label='Realized', color='black')
            ax.set_xlim(0, len(sorted_real))
            if resource != 'price':
                ax.set_ylim(0, max(sorted_real) * 1.3)
            ax.set_title(f'Month {month}')
            lbl = 'Mean Price' if resource == 'price' else 'Mean Production [MW]'
            ax.annotate(f'{lbl}\nRealized: {mean_real:.2f}\nSim: {np.mean(sim_means):.2f}',
                        xy=(0.25, 0.75), xycoords='axes fraction',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1))
            ax.legend()
            ax.set_xlabel('Hours')
            ax.set_ylabel(ylabel)

        plt.savefig(f'documentation/{self.plot_dir}monthly_duration_curve_{resource}.png')
        plt.close()

    def investigate_annual_duration_curves(self, simulations: list, resource: str = 'price'):
        train_data     = self.data[resource]
        simulated_data = [sim[resource] for sim in simulations]
        ylabel         = '[€/MWh]' if resource == 'price' else 'MW'

        mtx    = np.asarray([np.sort(s.values).reshape(-1) for s in simulated_data])
        p_low  = np.percentile(mtx, 5, axis=0)
        p_high = np.percentile(mtx, 95, axis=0)

        plt.figure(figsize=(10, 6))
        plt.fill_between(range(len(p_low)), p_low, p_high,
                         color='blue', alpha=0.2, label='90% CI')
        plt.plot(np.sort(np.mean(mtx, axis=0)), color='blue', alpha=0.8,
                 label='Mean of simulations')
        for yr in train_data.index.year.unique():
            plt.plot(np.sort(train_data.loc[train_data.index.year == yr]),
                     label=str(yr), alpha=0.8)

        lbl = 'Mean Price' if resource == 'price' else 'Mean Production [MW]'
        plt.annotate(
            f'{lbl}:\nTraining: {float(np.mean(train_data)):.2f}'
            f'\nSim: {float(np.mean(simulated_data)):.2f}',
            xy=(0.25, 0.75), xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1))
        plt.legend()
        plt.savefig(f'documentation/{self.plot_dir}annual_duration_curve_{resource}.png')
        plt.close()


# %%
