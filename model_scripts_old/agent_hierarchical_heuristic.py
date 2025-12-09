import numpy as np
import pandas as pd
from model_scripts_old.lp_deterministic import HourlyDeterministicLPModel, SpotBuyHourlyDeterministicLPModel
# from model_scripts_old.lp_stochastic import HourlyStochasticLPModel, SpotBuyHourlyStochasticLPModel
from common_scripts import Agent
from model_scripts_old.RFP_operational_environment import RFPOperationalEnv, SpotBuyRFPEnv
import matplotlib.pyplot as plt


class HierarchicalAgent(Agent):
    guideline_options = ("production_value", "planning_target", "contract_value", None) # Consider adding hourly_target for constant production.

    def __init__(self,
                 env,
                 *args,
                 writer=None,
                 guideline:str|None = "contract_value",
                 hourly_model_class=HourlyDeterministicLPModel,
                 solver='gurobi',
                 documentation=False,
                 **kwargs,
                 ):
        super().__init__(env, writer)
        self.documentation = documentation

        assert guideline in self.guideline_options
        self.guideline = guideline # Guideline strategy for long-term contracts.
        self.planning_horizon = self.env.planning_horizon
        self.allow_spot_buy = self.env.allow_spot_buy

        self.hourly_model = hourly_model_class(rfp = self.env.rfp,
                                               planning_horizon = self.env.planning_horizon,
                                               decision_horizon = self.env.decision_horizon,
                                               solver = solver,
                                               guideline = self.guideline,
                                               allow_spot_buy = self.allow_spot_buy,
                                               **kwargs, # **kwargs could meaningfully include allocate_production
                                               )
        self.hourly_model.initialize_model()
        self.logbook = {}
    
    def _update_logbook(self):
        """ Function to update the logbook of the hierarchical agent. See extra stats for purpose. """
        pass

    def extra_stats(self):
        """ Called by training algorithm to log agent stats about the experiments. """
        return self.logbook


class DeterministicHA(HierarchicalAgent):
    def __init__(self,
                 env:RFPOperationalEnv,
                 writer=None,
                 guideline:str|None = "contract_value",
                 hourly_model_class=HourlyDeterministicLPModel,
                 solver='gurobi',
                 documentation=False,
                 **kwargs,
                 ):
        super().__init__(env=env, writer=writer, guideline=guideline, hourly_model_class=hourly_model_class, solver=solver, documentation=documentation, **kwargs)

        self.electricity_consumption = {}
        self.electricity_consumption['hydrogen'] = self.env.rfp.get_component('Electrolyzer').parameters.get('charge_rate', 1/50) # MWh/tH2
        self.electricity_consumption['ammonia'] = self.electricity_consumption['hydrogen'] / self.env.rfp.get_component('Haber-Bosch Plant').parameters.get('rate', 5.5) # MWh/tNH3
        self.electricity_consumption['ammonia'] += self.env.rfp.get_component('Haber-Bosch Plant').parameters.get('charge_rate', 1) # MWh/tNH3

        # Internal value of ammonia production for Ammonia1 contract.
        self.ammonia_strike_price = self.env.rfp.get_contract('Ammonia1').parameters.get('price', 1000) # €/tNH3
        self.ammonia_hourly_target = self.env.rfp.get_contract('Ammonia1').parameters.get('volume', 50000) / 8760 # tNH3/hour
        if self.guideline == 'planning_target':
            self.ammonia_planning_target = self.ammonia_hourly_target * self.planning_horizon
        # Internal value of hydrogen production for Hydrogen1 contract.
        self.hydrogen_hourly_target = self.env.rfp.get_contract('Hydrogen1').parameters.get('volume', 1) # tH2/h
        self.hydrogen_strike_price = self.env.rfp.get_contract('Hydrogen1').parameters.get('price', 2000) # €/tH2
        
        self.logbook = {'ammonia_strike_price': [],
                        'ammonia_hourly_target': [],
                        'hydrogen_hourly_target': [],
                        'hydrogen_strike_price': [],
                        }
    
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
        if self.allow_spot_buy:
            # We always have our full GCP capacity available.
            simulation['available_power'] = self.env.rfp.get_component('Grid-Connection-Point').parameters.get('capacity')
        else:
            # Otherwise we are limited to PPAs.
            simulation['wind']            = self.env.wind_mapper(simulation['wind']) * self.env.wind_capacity
            simulation['solar']           = self.env.solar_mapper(simulation['solar']) * self.env.solar_capacity
            simulation['available_power'] = simulation['wind'] + simulation['solar'] + self.env.get_baseload_ppa_power()
        simulation['Hydrogen1']                   = self.hydrogen_hourly_target * self.electricity_consumption['hydrogen']
        simulation['available_power_for_ammonia'] = simulation['available_power'] - simulation['Hydrogen1']
        simulation['shifted_available_power']     = simulation['available_power_for_ammonia']
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
        missing_production_year = {contract.type: contract.parameters.get('volume', 8760*25) - info[name + '_produced_ytd'] for name, contract in self.env.rfp.get_annual_contracts().items()}

        year_simulations = self.env.forecaster.simulate_year_ahead(start = t, n_sims=n_sims) # Creates a list of n_sims simulated year-ahead forecasts (pd.DataFrame with hourly index and 'price', 'wind', 'solar' columns)
        strike_prices = np.zeros(n_sims)
        if self.documentation:
            fig, ax = plt.subplots(figsize=(12,10))
            ax.axvline(x=missing_production_year['ammonia_offtake'], label="Missing Contracted Ammonia Production", color='red', linestyle='--')
            ax.set_ylabel(r"€/MWh$_e$")
            ax.set_xlabel(r"t/NH$_3$")
        for sim_ix, simulation in enumerate(year_simulations):
            simulation = self._get_forecast_available_power(simulation)
            fc_rest_of_annual_contract = simulation.loc[pd.to_datetime(pd.date_range(start=t, end=info['annual_contract_deadline'], freq='h'), utc=True)]
            df_sorted = fc_rest_of_annual_contract.sort_values(by='price', ascending=True)
            df_sorted['cumulative_prod'] = np.cumsum(df_sorted['potential_ammonia_production'])
            idxs = np.where(df_sorted['cumulative_prod'] >= missing_production_year['ammonia_offtake'])[0]
            strike_idx = len(df_sorted) - 1 if len(idxs) == 0 else idxs[0]
            strike_prices[sim_ix] = df_sorted.iloc[strike_idx]['price']
            if self.documentation:
                lbl1 = "Simulated Weighted Price-Duration Curve" if sim_ix==0 else ""
                lbl2 = "Ammonia Strike Price" if sim_ix==0 else ""
                ax.plot(df_sorted['cumulative_prod'], df_sorted['price'], color='black', label=lbl1, alpha=0.7)
                ax.axhline(y=strike_prices[sim_ix], label=lbl2, color='blue', linestyle='--', alpha=0.7)
        if self.documentation:
            ax.legend()
            plt.savefig(f"documentation/heuristic_agent/strike_price_visulization_{str(t.date())}")
            plt.close()

        # Dependent on metric, return an estimate of strike price:
        return self._get_metric(strike_prices, metric=metric) * self.electricity_consumption['ammonia']

    def _define_hourly_target(self, t:pd.Timestamp, n_forecasts=1, metric='mean'):
        forecasts = self.env.forecaster.forecast(start=t, end=t+pd.Timedelta(self.env.planning_horizon*2-1, 'h'), n_forecasts=n_forecasts)
        hour_avg_targets = np.zeros(n_forecasts)
        for fore_ix, forecast in enumerate(forecasts):
            forecast = self._get_forecast_available_power(forecast)
            hour_avg_targets[fore_ix] = np.sum(forecast.loc[forecast['price']<self.ammonia_strike_price,'potential_ammonia_production'])/self.planning_horizon
        # Dependent on metric, return an estimate of daily target: (Much higher variance than on strike price - decision of metric more important)
        return self._get_metric(hour_avg_targets, metric=metric)

    def _solve_hourly_decisions(self, s, t:pd.Timestamp, info:dict):
        # Forecast prices and renewables for the planning horizon
        forecasts       = self.env.forecaster.forecast(start=t, end=t+pd.Timedelta(self.env.planning_horizon-1, 'h'), n_forecasts=1) # list of DFs
        wind_profile    = self.env.wind_mapper(forecasts[0]['wind'])
        wind_profile.loc[info['asset_wind_realization'].index] = info['asset_wind_realization']['wind']
        solar_profile   = self.env.solar_mapper(forecasts[0]['solar'])
        solar_profile.loc[info['asset_solar_realization'].index] = info['asset_solar_realization']['solar']

        # Format for Pyomo data input:
        wind_cf     = {('WindPower', t): wind_profile.iloc[t] for t in range(self.env.planning_horizon)}
        solar_cf    = {('SolarPower', t): solar_profile.iloc[t] for t in range(self.env.planning_horizon)}
        nuclear_cf  = {('NuclearPower', t): 1.0 for t in range(self.env.planning_horizon)}
        supplier_cf = {**wind_cf, **solar_cf, **nuclear_cf}

        # Scenario dependent electricity price forecasts:
        electricity_price_forecast = {t: forecasts[0].iloc[t]['price'] for t in range(self.env.planning_horizon)}
        datetime_data = {t: solar_profile.index[t] for t in range(self.env.planning_horizon)}

        data = { # Set up the necessary data for the LP Concrete Model
            None: {
                'T_datetime': datetime_data, 
                'init_soc': info['final_soc'],
                'supplier_cf': supplier_cf,
                'init_contract_status' : info['final_contract_status'],
                'init_virtual_contract_status' : info['final_virtual_contract_status'],
                'electricity_price': electricity_price_forecast,
                'contract_value': {'Ammonia1': self.ammonia_strike_price, 'Hydrogen1': self.hydrogen_strike_price},
            }
        }

        # Solve hourly LP model
        self.hourly_model.build_concrete_instance(data=data)
        self.hourly_model.run(verbose=False)

        return self.hourly_model.get_actions()

    def _update_logbook(self):
        self.logbook['ammonia_strike_price'].append(self.ammonia_strike_price)
        self.logbook['ammonia_hourly_target'].append(self.ammonia_hourly_target)
        self.logbook['hydrogen_strike_price'].append(self.hydrogen_strike_price)
        self.logbook['hydrogen_hourly_target'].append(self.hydrogen_hourly_target)

    def pi(self, s, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        t = info["time"]
        # if t.day_of_week == 0: # Then revisit H2 volume. Needs to be accepted by environment as an action.
        #     n_forecasts = 5
        #     self.hydrogen_hourly_target = self._set_new_hydrogen_volume(t=t, n_forecasts=n_forecasts, metric='median')
        if t.day_of_year % 15 == 1: # We do not expect big changes in strike price throughout the year - update two times a month.
            n_sims = 2
            self.ammonia_strike_price = self._estimate_strike_price(s=s, t=t, info=info, n_sims=n_sims, metric='mean')
        if t.day_of_year % 3 == 1 and self.guideline == 'planning_target': # We should update targets more often as they are based on short-term forecasts
            n_forecasts = 3
            self.ammonia_daily_target = self._define_hourly_target(t=t, n_forecasts=n_forecasts) # Hierarchical heuristic

        actions = self._solve_hourly_decisions(s=s, t=t, info=info) # Day-ahead solving

        self._update_logbook()

        return np.asarray(actions)


# class StochasticHA(DeterministicHA):
#     def __init__(self,
#                  env:RFPOperationalEnv,
#                  writer=None,
#                  guideline:str|None = "contract_value",
#                  hourly_model_class=HourlyStochasticLPModel,
#                  solver='gurobi',
#                  documentation=False,
#                  n_scenarios=None,
#                  **kwargs,
#                  ):
#         super().__init__(env=env, writer=writer, guideline=guideline, hourly_model_class=hourly_model_class, solver=solver, documentation=documentation, **kwargs)
#         if n_scenarios is not None:
#             self.hourly_model.n_scenarios = n_scenarios # Default can be found in HourlyStochasticLPModel
#             self.hourly_model.initialize_model() # If we want to specify number of scenarios at this point we rebuild the hourly model.
    
#     def _solve_hourly_decisions(self, s, t:pd.Timestamp, info:dict):
#         # Forecast scenarios for prices and renewables for the planning horizon:
#         forecasts = self.env.forecaster.forecast(start=t,
#                                                  end=t+pd.Timedelta(self.env.planning_horizon-1, 'h'),
#                                                  n_forecasts=self.hourly_model.n_scenarios,
#                                                  simulate_prices=True) # Returns list of DFs
#         wind_profile = self.env.wind_mapper(forecasts[0]['wind']) # Still deterministic VRE availability
#         wind_profile.loc[info['asset_wind_realization'].index] = info['asset_wind_realization']['wind']
#         solar_profile = self.env.solar_mapper(forecasts[0]['solar']) # Still deterministic VRE availability
#         solar_profile.loc[info['asset_solar_realization'].index] = info['asset_solar_realization']['solar']

#         # Format for Pyomo data input:
#         wind_cf = {('WindPower', t): wind_profile.iloc[t] for t in range(self.env.planning_horizon)}
#         solar_cf = {('SolarPower', t): solar_profile.iloc[t] for t in range(self.env.planning_horizon)}
#         nuclear_cf = {('NuclearPower', t): 1.0 for t in range(self.env.planning_horizon)}
#         supplier_cf = {**wind_cf, **solar_cf, **nuclear_cf}

#         # Scenario dependent electricity price forecasts:
#         electricity_price_forecasts = {(s, t): forecasts[s].iloc[t]['price'] for t in range(self.env.planning_horizon) for s in range(self.hourly_model.n_scenarios)}
#         datetime_data = {t: solar_profile.index[t] for t in range(self.env.planning_horizon)}

#         data = { # Set up the necessary data for the LP Concrete Model
#             None: {
#                 'T_datetime': datetime_data, 
#                 'init_soc': info['final_soc'],
#                 'supplier_cf': supplier_cf,
#                 'init_contract_status' : info['final_contract_status'],
#                 'init_virtual_contract_status' : info['final_virtual_contract_status'],
#                 'electricity_price': electricity_price_forecasts,
#                 'contract_value': {'Ammonia1': self.ammonia_strike_price, 'Hydrogen1': self.hydrogen_strike_price},
#             }
#         }

#         # Solve hourly LP model
#         self.hourly_model.build_concrete_instance(data=data)
#         self.hourly_model.run(verbose=False)

#         return self.hourly_model.get_actions()


class ConstantHA(HierarchicalAgent):
    guideline_options = ('hourly_target', None)

    def __init__(self,
                 env:RFPOperationalEnv,
                 writer=None,
                 hourly_model_class=HourlyDeterministicLPModel,
                 solver='gurobi',
                 documentation=False,
                 **kwargs,
                 ):
        super().__init__(env=env, writer=writer, guideline="hourly_target", hourly_model_class=hourly_model_class, solver=solver, documentation=documentation, **kwargs)
        self.hydrogen_hourly_target = self.env.rfp.get_contract('Hydrogen1').parameters.get('volume', 1) # tH2/h
        self.ammonia_hourly_target = self.env.rfp.get_contract('Ammonia1').parameters.get('volume', 50000) / 8760 # tNH3/hour
        self.logbook["ammonia_hourly_target"] = []
        self.logbook["hydrogen_hourly_target"] = []
    
    def _calculate_hourly_ammonia_target(self, s, t):
        hours_in_year = (self.env.year_end - t).value / 3600 * 1e-9 + 1 # The timedelta is in nanoseconds, we convert to hours inclusive (+1)
        self.ammonia_hourly_target = (self.env.rfp.get_contract('Ammonia1').parameters.get('volume', 50000) - s[2] ) / hours_in_year # tNH3/day
    
    def _solve_hourly_decisions(self, s, t:pd.Timestamp, info:dict):
        # Forecast scenarios for prices and renewables for the planning horizon:
        forecasts = self.env.forecaster.forecast(start=t, end=t+pd.Timedelta(self.env.planning_horizon-1, 'h'), n_forecasts=1) # list of DFs
        wind_profile = self.env.wind_mapper(forecasts[0]['wind']) # Get forecast wind
        wind_profile.loc[info['asset_wind_realization'].index] = info['asset_wind_realization']['wind'] # First 24 hours are known
        solar_profile = self.env.solar_mapper(forecasts[0]['solar']) # Get forecast solar 
        solar_profile.loc[info['asset_solar_realization'].index] = info['asset_solar_realization']['solar'] # First 24 hours are known

        # Format for Pyomo data input:
        wind_cf = {('WindPower', t): wind_profile.iloc[t] for t in range(self.env.planning_horizon)}
        solar_cf = {('SolarPower', t): solar_profile.iloc[t] for t in range(self.env.planning_horizon)}
        nuclear_cf = {('NuclearPower', t): 1.0 for t in range(self.env.planning_horizon)}
        supplier_cf = {**wind_cf, **solar_cf, **nuclear_cf}

        # Electricity price forecasts:
        electricity_price_forecast = {t: forecasts[0].iloc[t]['price'] for t in range(self.env.planning_horizon)}
        datetime_data = {t: solar_profile.index[t] for t in range(self.env.planning_horizon)}

        data = { # Set up the necessary data for the LP Concrete Model
            None: {
                'T_datetime': datetime_data, 
                'init_soc': info['final_soc'],
                'supplier_cf': supplier_cf,
                'init_contract_status' : info['final_contract_status'],
                'init_virtual_contract_status' : info['final_virtual_contract_status'],
                'electricity_price': electricity_price_forecast,
                'hourly_target': {'Ammonia1': self.ammonia_hourly_target, 'Hydrogen1': self.hydrogen_hourly_target},
            }
        }

        # Solve hourly LP model
        self.hourly_model.build_concrete_instance(data=data)
        self.hourly_model.run(verbose=False)
        
        return self.hourly_model.get_actions()

    def _update_logbook(self):
        self.logbook['ammonia_hourly_target'].append(self.ammonia_hourly_target)
        self.logbook['hydrogen_hourly_target'].append(self.hydrogen_hourly_target)

    def pi(self, s, k, info:dict):
        # Get current time as a datetime object:
        t = info["time"]
        
        # Constant heuristic:
        self._calculate_hourly_ammonia_target(s, t)
        
        # Solve hourly LP:
        actions = self._solve_hourly_decisions(s=s, t=t, info=info) # Day-ahead solving
        
        # Log hierarchical agent specifics:
        self._update_logbook()

        # Return action to environment:
        return np.asarray(actions)
