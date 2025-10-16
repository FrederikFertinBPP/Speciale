import numpy as np
import pandas as pd
from model_scripts.hourly_lp_model import HourlyDeterministicLPModel
from model_scripts import Agent
from model_scripts.RFP_operational_environment import RFPOperationalEnv

class DeterministicHierarchicalAgent(Agent):
    """ For now, the agent and the environment share the forecaster object between them. """
    guideline_options = ('strike_price', 'planning_target', None)

    def __init__(self, env:RFPOperationalEnv, guideline:str|None = "strike_price"):
        self.env = env
        self.annual_contract_commitments = self.env.rfp.get_annual_contracts() # Get annual contract commitments
        self.monthly_contract_commitments = self.env.rfp.get_monthly_contracts() # Get monthly contract commitments
        self.electricity_consumption = {}
        self.electricity_consumption['hydrogen'] = self.env.rfp.get_component('Electrolyzer').parameters.get('rate', 1/50) # tH2/MWh
        self.electricity_consumption['ammonia'] = self.electricity_consumption['hydrogen'] * self.env.rfp.get_component('Haber-Bosch Plant').parameters.get('rate', 5.5) # tNH3/MWh
        self.electricity_consumption['ammonia'] = 1/self.electricity_consumption['ammonia'] + self.env.rfp.get_component('Haber-Bosch Plant').parameters.get('charge_rate', 1) # MWh/tNH3
        self.electricity_consumption['hydrogen'] = 1/self.electricity_consumption['hydrogen'] # MWh/tH2

        assert guideline in self.guideline_options
        self.guideline = guideline # Guideline strategy for long-term contracts.

        self.ammonia_strike_price = self.env.rfp.get_contract('Ammonia1').parameters.get('price', 1000) / self.electricity_consumption['ammonia'] # Max electricity price in €/MWh for ammonia production to be the good decision.
        self.ammonia_daily_target = self.env.rfp.get_contract('Ammonia1').parameters.get('volume', 50000) / 365 # tNH3/day
        self.hydrogen_hourly_target = self.env.rfp.get_contract('Hydrogen1').parameters.get('volume', 1) # tH2/h
        self.hydrogen_strike_price = self.env.rfp.get_contract('Hydrogen1').parameters.get('price', 2000) # €/tH2

        self.hourly_model = HourlyDeterministicLPModel(self.env.rfp,
                                                       planning_horizon=self.env.planning_horizon,
                                                       decision_horizon=self.env.decision_horizon,
                                                       guideline=self.guideline,
                                                       solver='gurobi',
                                                       )
        
        self.logbook = {'ammonia_strike_price': [],
                         'ammonia_daily_target': [],
                         'hydrogen_hourly_target': [],
                         'hydrogen_strike_price': []}
    
    def _get_metric(self, array, metric='mean'):
        if metric == 'mean':
            return np.mean(array)
        elif metric == 'median':
            return np.median(array)
        elif metric == 'min':
            return np.min(array)
        elif metric == 'max':
            return np.max(array)
        else:
            print("Could not recognize metric given.")
            return np.mean(array)

    def _get_forecast_available_power(self, simulation):
        simulation = simulation.copy()
        simulation['wind']                          = self.env.wind_mapper(simulation['wind']) * self.env.wind_capacity
        simulation['solar']                         = self.env.solar_mapper(simulation['solar']) * self.env.solar_capacity
        simulation['available_power']               = simulation['wind'] + simulation['solar'] + self.env.get_baseload_ppa_power()
        simulation['Hydrogen1']                     = self.hydrogen_hourly_target * self.electricity_consumption['hydrogen']
        simulation['available_power_for_ammonia']   = simulation['available_power'] - simulation['Hydrogen1']
        simulation['shifted_available_power']       = simulation['available_power_for_ammonia']
        # Create shifted availability profile to account for negatives in case of large constant hydrogen outflow.
        for ix in range(len(simulation)-1, 0, -1):
            p = simulation.iloc[ix]['shifted_available_power']
            if p<0: # Shift demand if there is not enough supply.
                simulation.loc[simulation.index[ix], 'shifted_available_power'] = 0
                simulation.loc[simulation.index[ix-1], 'shifted_available_power'] += p
        simulation.loc[simulation.index[0], 'shifted_available_power'] = np.max([simulation.iloc[0]['shifted_available_power'], 0])
        simulation['potential_ammonia_production'] = np.clip(simulation['shifted_available_power'] / self.electricity_consumption['ammonia'], 0,
                                                            self.env.rfp.get_component('Haber-Bosch Plant').parameters.get('capacity', 50)) # tNH3 for every hour
        return simulation

    def _set_new_hydrogen_volume(self, t:pd.Timestamp, n_forecasts=3, metric='median'):
        hours_in_forecast = 7*24
        forecasts = self.env.forecaster.forecast(start=t, end=t+pd.Timedelta(hours_in_forecast-1, 'h'), n_forecasts=n_forecasts)
        opt_volumes = np.zeros(n_forecasts)
        for fore_ix, forecast in enumerate(forecasts):
            forecast = self._get_forecast_available_power(forecast)
            opt_volumes[fore_ix] = np.clip(a = np.sum(forecast.loc[(forecast['price'] < self.hydrogen_strike_price) & 
                                                                    (self.ammonia_strike_price < self.hydrogen_strike_price), 
                                                                    'available_power']) / hours_in_forecast,
                                           a_min = self.env.rfp.get_contract('Hydrogen1').parameters.get('min_volume', 0),
                                           a_max = self.env.rfp.get_contract('Hydrogen1').parameters.get('max_volume', 3))
        # Dependent on metric, return an estimate of optimal contracted volume: (Much higher variance than on strike price - decision of metric more important)
        return self._get_metric(opt_volumes, metric=metric)
    
    def _estimate_strike_price(self, s, t:pd.Timestamp, info:dict, n_sims=3, metric='mean'):
        missing_production_year = {contract.type: contract.parameters.get('volume', 8760*25) - info[contract.name + '_produced_ytd'] for contract in self.annual_contract_commitments}

        year_simulations = self.env.forecaster.simulate_year_ahead(start = t, n_sims=n_sims) # Creates a list of n_sims simulated year-ahead forecasts (pd.DataFrame with hourly index and 'price', 'wind', 'solar' columns)
        strike_prices = np.zeros(n_sims)
        for sim_ix, simulation in enumerate(year_simulations):
            simulation = self._get_forecast_available_power(simulation)
            fc_rest_of_annual_contract = simulation.loc[pd.to_datetime(pd.date_range(start=t, end=info['annual_contract_deadline'], freq='h'), utc=True)]
            df_sorted = fc_rest_of_annual_contract.sort_values(by='price', ascending=True)
            df_sorted['cumulative_prod'] = np.cumsum(df_sorted['potential_ammonia_production'])
            idxs = np.where(df_sorted['cumulative_prod'] >= missing_production_year['ammonia_offtake'])[0]
            strike_idx = len(df_sorted) - 1 if len(idxs) == 0 else idxs[0]
            strike_prices[sim_ix] = df_sorted.iloc[strike_idx]['price']
        # Dependent on metric, return an estimate of strike price:
        return self._get_metric(strike_prices, metric=metric)

    def _define_daily_target(self, t:pd.Timestamp, n_forecasts=1, metric='mean'):
        forecasts = self.env.forecaster.forecast(start=t, end=t+pd.Timedelta(self.env.planning_horizon*2-1, 'h'), n_forecasts=n_forecasts)
        day_avg_targets = np.zeros(n_forecasts)
        for fore_ix, forecast in enumerate(forecasts):
            forecast = self._get_forecast_available_power(forecast)
            day_avg_targets[fore_ix] = np.sum(forecast.loc[forecast['price']<self.ammonia_strike_price,'potential_ammonia_production'])/self.env.planning_horizon * 24
        # Dependent on metric, return an estimate of daily target: (Much higher variance than on strike price - decision of metric more important)
        return self._get_metric(day_avg_targets, metric=metric)

    def _solve_hourly_decisions(self, s, t:pd.Timestamp, info:dict):
        data = {} # We need to set up the necessary data for the LP Concrete Model
        forecasts = self.env.forecaster.forecast(start=t, end=t+pd.Timedelta(self.env.planning_horizon-1, 'h'), n_forecasts=1) # list of DFs
        wind_profile = self.env.wind_mapper(forecasts[0]['wind'])
        wind_profile.loc[info['asset_wind_realization'].index] = info['asset_wind_realization']['wind']
        solar_profile = self.env.solar_mapper(forecasts[0]['solar'])
        solar_profile.loc[info['asset_solar_realization'].index] = info['asset_solar_realization']['solar']
        wind_cf = {('WindPower', t): wind_profile.iloc[t] for t in range(self.env.planning_horizon)}
        solar_cf = {('SolarPower', t): solar_profile.iloc[t] for t in range(self.env.planning_horizon)}
        nuclear_cf = {('NuclearPower', t): 1.0 for t in range(self.env.planning_horizon)}
        electricity_price = {t: forecasts[0].iloc[t]['price'] for t in range(self.env.planning_horizon)}
        data = {
            None: {
                'init_soc': {
                    'Hydrogen Storage': info['final_soc_H2'],
                    'Ammonia Storage': info['final_soc_NH3'],
                },
                'contract_target': {
                    'Hydrogen1': self.hydrogen_hourly_target,
                    'Ammonia1': self.ammonia_daily_target * self.env.planning_horizon / 24,
                },
                'supplier_cf': {
                    **wind_cf,
                    **solar_cf,
                    **nuclear_cf,
                },
                'electricity_price': electricity_price,
                'ammonia_strike_price': {None: self.ammonia_strike_price * self.electricity_consumption['ammonia']},
            }
        }
        self.hourly_model.build_concrete_instance(data=data)
        self.hourly_model.run(verbose=False)
        return self.hourly_model.get_actions()
    
    def pi(self, s, k, info:dict):
        """ Hierarchical policy for the agent.
        We start by defining the guidelines for the hourly decisions. """
        t = info["time"]
        # if t.day_of_week == 0: # Then revisit H2 volume. Needs to be accepted by environment as an action.
        #     n_forecasts = 5
        #     self.hydrogen_hourly_target = self._set_new_hydrogen_volume(t=t, n_forecasts=n_forecasts, metric='median')
        if t.day_of_year % 15 == 1: # We do not expect big changes in strike price throughout the year - update two times a month.
            n_sims = 2
            self.ammonia_strike_price = self._estimate_strike_price(s=s, t=t, info=info, n_sims=n_sims, metric='mean')
        if t.day_of_year % 3 == 1: # We should update targets more often as they are based on short-term forecasts
            n_forecasts = 3
            self.ammonia_daily_target = self._define_daily_target(t=t, n_forecasts=n_forecasts) # Hierarchical heuristic

        actions = self._solve_hourly_decisions(s=s, t=t, info=info) # Day-ahead solving

        self.logbook['ammonia_strike_price'].append(self.ammonia_strike_price)
        self.logbook['ammonia_daily_target'].append(self.ammonia_daily_target)
        self.logbook['hydrogen_hourly_target'].append(self.hydrogen_hourly_target)
        self.logbook['hydrogen_strike_price'].append(self.hydrogen_strike_price)

        return np.asarray(actions)
    
    def save_logbook_as_csv(self, filepath=""):
        if len(filepath)>0:
            df = pd.DataFrame(self.logbook)
            df.to_csv(filepath)

