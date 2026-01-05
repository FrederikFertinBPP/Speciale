import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
sns.set_theme("notebook")

from model_scripts.hourly_models import HourlyDeterministicLPModel, HourlyStochasticLPModel
from common_scripts import Agent, cache_write
from model_scripts.environment import RFPShieldEnv, RFPEnv


class HierarchicalAgent(Agent):
    guideline_options = ("production_value", "hourly_target", None) # Consider adding hourly_target for constant production.
    error_margin = 1e-5

    def __init__(self,
                 env,
                 *args,
                 writer=None,
                 planning_horizon:int = 4*24,
                 guideline:str|None = "production_value",
                 hourly_model_class=HourlyDeterministicLPModel,
                 solver='gurobi',
                 documentation=False,
                 objective_logic=None,
                 **kwargs,
                 ):
        super().__init__(env, writer)
        self.documentation = documentation

        assert guideline in self.guideline_options
        self.guideline = guideline # Guideline strategy for long-term contracts.
        self.decision_horizon = self.env.decision_horizon
        self.planning_horizon = max(planning_horizon, self.decision_horizon)
        self.allow_spot_buy = self.env.allow_spot_buy

        if hourly_model_class is not None:
            self.hourly_model = hourly_model_class(rfp = self.env.rfp,
                                                planning_horizon = self.planning_horizon,
                                                decision_horizon = self.decision_horizon,
                                                solver = solver,
                                                guideline = self.guideline,
                                                allow_spot_buy = self.allow_spot_buy,
                                                objective_logic=objective_logic,
                                                **kwargs,
                                                )
            self.hourly_model.initialize_model()
        self.logbook = {}

    def _get_supplier_cf(self, obs, forecast):
        wind_profile    = self.env.wind_mapper(forecast['wind'])
        solar_profile   = self.env.solar_mapper(forecast['solar'])
        
        supplier_cf = {}
        for ix, ppa_name in enumerate(self.env.ppa_names):
            ppa = self.env.rfp.get_ppa(ppa_name)
            forecast_profile = np.ones(self.planning_horizon)
            if ppa.parameters.get("consumes") == 'wind':
                forecast_profile = wind_profile.values
            elif ppa.parameters.get("consumes") == 'solar':
                forecast_profile = solar_profile.values
            forecast_profile[:self.env.ppa_context_space.shape[0]] = obs['context']['ppas'][:,ix]
            supplier_cf = {**supplier_cf, **{(ppa_name, t): forecast_profile[t] for t in range(self.planning_horizon)}}
        return supplier_cf
    
    def _get_forecasts_and_electricity(self, t):
        # Forecast prices and renewables for the planning horizon
        if self.env.load_data:
            timestamp_str = t.strftime("%Y%m%d")
            forecasts = [pd.read_csv(f"{self.env.scenario_path}forecast_{timestamp_str}_0.csv")]
        else:
            forecasts = self.env.forecaster.forecast(start=t, end=t+pd.Timedelta(self.planning_horizon-1, 'h'), n_forecasts=1) # list of DFs
        # Scenario dependent electricity price forecasts:
        electricity_price_forecast = {t: forecasts[0].iloc[t]['price'] for t in range(self.planning_horizon)}
        return forecasts, electricity_price_forecast

    def _save_obs_for_debug(self, obs, info):
        debug_info = {**obs, **info}
        file_path = os.getcwd() + "/debug_info.pkl"
        cache_write(debug_info, file_path, verbose=True)

    def _update_logbook(self):
        """ Function to update the logbook of the hierarchical agent. See extra stats for purpose. """
        pass

    def extra_stats(self):
        """ Called by training algorithm to log agent stats about the experiments. """
        return self.logbook
    
    def close(self):
        if self.hourly_model.uses_persistent_solver:
            self.hourly_model.solver.close()

    def __repr__(self):
        return self.__class__.__name__


class DeterministicHA(HierarchicalAgent):
    def __init__(self,
                 env:RFPShieldEnv,
                 writer=None,
                 planning_horizon:int=4*24,
                 guideline:str|None = "production_value",
                 hourly_model_class=HourlyDeterministicLPModel,
                 solver='gurobi',
                 documentation=False,
                 n_sims=2,
                 **kwargs,
                 ):
        super().__init__(env=env, writer=writer, planning_horizon=planning_horizon, guideline=guideline, hourly_model_class=hourly_model_class, solver=solver, documentation=documentation, **kwargs)
        self.n_sims = n_sims

        self.electricity_consumption = {}
        self.electricity_consumption['hydrogen'] = self.env.rfp.get_component('Electrolyzer').parameters.get('electricity_consumption', 1/50) # MWh/tH2
        self.electricity_consumption['ammonia'] = self.electricity_consumption['hydrogen'] / self.env.rfp.get_component('Haber Bosch Plant').parameters.get('rate', 5.5) # MWh/tNH3
        self.electricity_consumption['ammonia'] += self.env.rfp.get_component('Haber Bosch Plant').parameters.get('electricity_consumption', 1) # MWh/tNH3

        # Internal value of ammonia production for Ammonia1 contract.
        self.ammonia_spot_price = self.env.rfp.get_contract("AmmoniaSpot").parameters.get("price")
        self.ammonia_strike_price = self.env.rfp.get_contract('Ammonia1').parameters.get('price', 1000) # €/tNH3
        self.ammonia_hourly_target = self.env.rfp.get_contract('Ammonia1').parameters.get('volume', 50000) / 8760 # tNH3/hour
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
        elif metric is None:
            return array
        else:
            print("Could not recognize metric given. Returning mean.")
            return np.mean(array)

    def _get_forecast_available_power(self, simulation):
        simulation = simulation.copy()
        gcp_cap = self.env.rfp.get_component('Grid Connection Point').parameters.get('capacity')
        if self.allow_spot_buy:
            # We always have our full GCP capacity available.
            simulation['available_power'] = gcp_cap
        else:
            # Otherwise we are limited to PPAs.
            simulation['wind']            = self.env.wind_mapper(simulation['wind']) * self.env.wind_capacity
            simulation['solar']           = self.env.solar_mapper(simulation['solar']) * self.env.solar_capacity
            simulation['available_power'] = np.clip(simulation['wind'] + simulation['solar'] + self.env.get_baseload_ppa_power(),0,gcp_cap)
        hourly_need = 0
        for name, contract in self.env.rfp.get_contracts().items():
            is_spot_contract = bool(contract.parameters.get("spot_contract", 0))
            if is_spot_contract == False:
                if contract.target_frequency == "hourly":
                    hourly_need += contract.parameters.get('volume') * self.electricity_consumption[contract.parameters.get('resource')]
        simulation['hourly_contract_need'] = hourly_need
        simulation['available_power_for_ammonia'] = simulation['available_power'] - simulation['hourly_contract_need']
        simulation['shifted_available_power']     = simulation['available_power_for_ammonia']
        # Create shifted availability profile to account for negatives in case of large constant hydrogen outflow.
        for ix in range(len(simulation)-1, 0, -1):
            p = simulation.iloc[ix]['shifted_available_power']
            if p<0: # Shift demand if there is not enough supply.
                simulation.loc[simulation.index[ix], 'shifted_available_power'] = 0
                simulation.loc[simulation.index[ix-1], 'shifted_available_power'] += p
        simulation.loc[simulation.index[0], 'shifted_available_power'] = np.max([simulation.iloc[0]['shifted_available_power'], 0])
        simulation['potential_ammonia_production'] = np.clip(simulation['shifted_available_power'] / self.electricity_consumption['ammonia'], 0,
                                                            self.env.rfp.get_component('Haber Bosch Plant').parameters.get('capacity', 50)) # tNH3 for every hour
        return simulation

    def _calculate_hourly_ammonia_target(self, obs, time):
        hours_in_year = (self.env.episode_end - time).value / 3600 * 1e-9 + 1 # The timedelta is in nanoseconds, we convert to hours inclusive (+1)
        contract_ix = np.where(np.asarray(self.env.contract_names) == "Ammonia1")[0][0]
        storage_ix = np.where(np.asarray(self.env.storage_names) == "Ammonia Storage")[0][0]
        self.ammonia_hourly_target = (self.env.contract_state_space.high[contract_ix] - obs['state']['contracts'][contract_ix] - obs['state']['storages'][storage_ix]) / hours_in_year # tNH3/day
    
    def _set_new_hydrogen_volume(self, time:pd.Timestamp, n_forecasts=3, metric='median'):
        if self.env.load_data:
            hours_in_forecast = 4*24
            timestamp_str = time.strftime("%Y%m%d")
            assert n_forecasts <= 10
            forecasts = [pd.read_csv(f"{self.env.scenario_path}forecast_{timestamp_str}_{ix}.csv") for ix in range(n_forecasts)]
        else:
            hours_in_forecast = 7*24
            forecasts = self.env.forecaster.forecast(start=time, end=time+pd.Timedelta(hours_in_forecast-1, 'h'), n_forecasts=n_forecasts)
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

    def _estimate_strike_price(self, obs, time:pd.Timestamp, info:dict, n_sims=3, metric='mean'):
        if self.env.load_data:
            # if n_sims > 5:
            #     print("Only generated 5 simulations year ahead - setting number of sims to 5.")
            n_sims = 5
            timestamp_str = time.strftime("%Y%m%d")
            updated_timestamp_str = timestamp_str[:-2] + '01'
            hours_extra = (int(timestamp_str) - int(updated_timestamp_str))*24 # We have only generate for the start of each month. So we remove days which have already passed.
            year_simulations = [pd.read_csv(f"{self.env.scenario_path}year_sim_{updated_timestamp_str}_{ix}.csv").iloc[hours_extra:] for ix in range(n_sims)]
        else:
            year_simulations = self.env.forecaster.simulate_year_ahead(start = time, n_sims=n_sims) # Creates a list of n_sims simulated year-ahead forecasts (pd.DataFrame with hourly index and 'price', 'wind', 'solar' columns)
        deadlines = {}
        deadlines['daily']   = pd.to_datetime(time.date() + pd.Timedelta(23,'h'), utc=True)
        deadlines['monthly'] = pd.to_datetime(pd.Timestamp(time.year,time.month,time.days_in_month) + pd.Timedelta(23,'h'), utc=True)
        deadlines['yearly']  = pd.to_datetime(pd.Timestamp(time.year,12,31) + pd.Timedelta(23,'h'), utc=True)
        
        freq_options = self.env.rfp.frequency_options
        freq_rank = self.env.rfp.frequency_rank
        strike_prices = np.zeros(n_sims)
        if self.documentation:
            fig, ax = plt.subplots(figsize=(12,10))
            ax.set_ylabel(r"€/MWh$_e$")
            ax.set_xlabel(r"t/NH$_3$")
        for sim_ix, simulation in enumerate(year_simulations):
            simulation = self._get_forecast_available_power(simulation)
            target_volume_current = dict(zip(freq_options, np.zeros(len(freq_options))))
            target_volume_normal = dict(zip(freq_options, np.zeros(len(freq_options))))
            strike_price = 0
            for freq in freq_options:
                if freq == 'monthly':
                    target_volume_normal['monthly'] += target_volume_current['daily'] + target_volume_normal['daily'] * (deadlines['monthly'] - time).days
                elif freq == 'yearly':
                    target_volume_normal['yearly'] += target_volume_current['monthly'] + target_volume_normal['monthly'] * (deadlines['yearly'].month - time.month)
                target_volume_current[freq] = target_volume_normal[freq]
                if freq != 'hourly':
                    if self.env.load_data:
                        sim_slice = simulation # We have only pre-generated sliced data. 
                    else:
                        sim_slice = simulation.loc[pd.to_datetime(pd.date_range(start=time, end=deadlines[freq], freq='h'), utc=True)]
                    _sorted = sim_slice.sort_values(by='price', ascending=True)
                    _sorted['cumulative_prod'] = np.cumsum(_sorted['potential_ammonia_production'])
                    for ix, name in enumerate(self.env.contract_names):
                        contract = self.env.rfp.get_contract(name)
                        frequency = contract.target_frequency
                        if freq_rank[freq] >= freq_rank[frequency] and contract.parameters.get("resource") == 'ammonia':
                            allocated_contract_status = obs['state']['contracts'][ix]
                            allocated_contract_status += obs['state']['storages'][1]
                            target_volume = self.env.contract_state_space.high[ix]
                            target_volume_normal[freq] += target_volume
                            missing_volume_current = target_volume - allocated_contract_status
                            target_volume_current[freq] += missing_volume_current
                            strike_idx = max(0, np.sum(_sorted['cumulative_prod'] <= target_volume_current[freq]) - 1)
                            strike_price = max(strike_price, _sorted.iloc[strike_idx]['price'])
            strike_prices[sim_ix] = strike_price
            if self.documentation:
                lbl1 = "Simulated Weighted Price-Duration Curve" if sim_ix==0 else ""
                lbl2 = "Ammonia Strike Price" if sim_ix==0 else ""
                ax.plot(_sorted['cumulative_prod'], _sorted['price'], color='black', label=lbl1, alpha=0.7)
                ax.axhline(y=strike_prices[sim_ix], label=lbl2, color='blue', linestyle='--', alpha=0.7)
        if self.documentation:
            ax.axvline(x=target_volume_current['yearly'], label="Missing Contracted Ammonia Production", color='red', linestyle='--')
            ax.legend() 
            plt.savefig(f"documentation/heuristic_agent/strike_price_visulization_{str(time.date())}")
            plt.close()

        # Dependent on metric, return an estimate of strike price:
        sp = self._get_metric(strike_prices, metric=metric) * self.electricity_consumption['ammonia']
        if isinstance(sp, np.ndarray): # The lowest possible internal value of ammonia is the spot value (not completely true with storage limits, but good enough)
            sp[sp<self.ammonia_spot_price] = self.ammonia_spot_price
        else:
            sp = max(self.ammonia_spot_price, sp)
        return sp

    def _construct_concrete_data(self, obs, time):
        """ Creates time, state, and steering signal of data object. """
        time_index = pd.to_datetime(pd.date_range(time, time+pd.Timedelta(self.planning_horizon-1,'h'),freq='h'), utc=True)
        datetime_data = {t: time_index[t] for t in range(self.planning_horizon)}

        soc = dict(zip(self.env.storage_names, obs['state']['storages']))
        for key, val in soc.items():
            if val<0 and abs(val) < self.error_margin:
                soc[key] = 0
        contract_status = dict(zip(self.env.contract_names, obs['state']['contracts']))
        for key, val in contract_status.items():
            if val<0 and abs(val) < self.error_margin:
                contract_status[key] = 0
        # offtaker_availability = {(self.env.offtaker_names[ix], t) : obs['context']['offtakers'][t,ix]
        #                          for ix in range(len(self.env.offtaker_names)) for t in range(self.planning_horizon)}

        steering = None
        if self.guideline == 'production_value':
            steering = {'Haber Bosch Plant': self.ammonia_strike_price}
        elif self.guideline == 'hourly_target':
            steering = {None: self.ammonia_hourly_target}
        
        data = { # Set up the necessary data for the LP Concrete Model
            None: {
                'T_datetime': datetime_data, 
                'init_soc': soc,
                'init_contract_status' : contract_status,
                # 'offtaker_availability': offtaker_availability,
                self.guideline: steering,
            }
        }
        return data

    def _run_hourly_model(self, obs, time, info):
        self.hourly_model.run(verbose=False)

        actions = self.hourly_model.get_actions()
        if actions is not None:
            return actions
        else:
            self._save_obs_for_debug(obs, info)
            raise ValueError(f"Could not get actions.\nTime: {time}.\nState at termination: {obs['state']}.") 

    def _solve_hourly_decisions(self, obs, time:pd.Timestamp, info:dict):
        data = self._construct_concrete_data(obs, time)
        
        forecasts, electricity_price_forecast = self._get_forecasts_and_electricity(time)
        supplier_cf = self._get_supplier_cf(obs, forecasts[0])
        data[None]["supplier_cf"] = supplier_cf
        data[None]["electricity_price"] = electricity_price_forecast

        # Solve hourly LP model
        self.hourly_model.build_concrete_instance(data=data)
        return self._run_hourly_model(obs, time, info)

    def _update_logbook(self):
        self.logbook['ammonia_strike_price'].append(self.ammonia_strike_price)
        self.logbook['ammonia_hourly_target'].append(self.ammonia_hourly_target)
        self.logbook['hydrogen_strike_price'].append(self.hydrogen_strike_price)
        self.logbook['hydrogen_hourly_target'].append(self.hydrogen_hourly_target)

    def pi(self, obs, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        time = info["time"]
        if self.guideline == 'hourly_target':
            # Constant heuristic:
            self._calculate_hourly_ammonia_target(obs, time)
        else:
            if time.day % 15 == 1 and self.guideline == 'production_value': # We do not expect big changes in strike price throughout the year - update two times a month.
                self.ammonia_strike_price = self._estimate_strike_price(obs=obs, time=time, info=info, n_sims=self.n_sims, metric='mean')

        actions = self._solve_hourly_decisions(obs=obs, time=time, info=info) # Day-ahead solving

        self._update_logbook()

        return np.asarray(actions)


class StochasticHA(DeterministicHA):
    def __init__(self,
                 env:RFPShieldEnv,
                 writer=None,
                 planning_horizon:int=4*24,
                 guideline:str|None = "production_value",
                 hourly_model_class=HourlyStochasticLPModel,
                 solver='gurobi',
                 documentation=False,
                 n_scenarios=None,
                 **kwargs,
                 ):
        super().__init__(env=env, writer=writer, planning_horizon=planning_horizon, guideline=guideline, hourly_model_class=hourly_model_class, solver=solver, documentation=documentation, **kwargs)
        if n_scenarios is not None:
            self.n_scenarios = n_scenarios
            self.hourly_model.n_scenarios = n_scenarios # Default can be found in HourlyStochasticLPModel
            self.hourly_model.initialize_model() # If we want to specify number of scenarios at this point we rebuild the hourly model.
    
    def _get_forecasts_and_electricity(self, t):
        if self.env.load_data:
            timestamp_str = t.strftime("%Y%m%d")
            assert self.hourly_model.n_scenarios <= 10
            forecasts = [pd.read_csv(f"{self.env.scenario_path}forecast_{timestamp_str}_{ix}.csv") for ix in range(self.hourly_model.n_scenarios)]
        else:
            forecasts = self.env.forecaster.forecast(start=t,
                                                    end=t+pd.Timedelta(self.planning_horizon-1, 'h'),
                                                    n_forecasts=self.hourly_model.n_scenarios,
                                                    simulate_prices=True) # Returns list of DFs
        # Scenario dependent electricity price forecasts:
        electricity_price_forecasts = {(s, t): forecasts[s].iloc[t]['price'] for t in range(self.planning_horizon) for s in range(self.hourly_model.n_scenarios)}
        return forecasts, electricity_price_forecasts

    def __repr__(self):
        return self.__class__.__name__ + str(self.n_scenarios)


from model_scripts.hourly_models import DecisionRuleModel, HourlyRecourseModel, StochasticRecourseModel
from model_scripts.environment import RFPYearEnv, RFPRecourseEnv


class RecourseAgent(StochasticHA):
    """ If n_scenarios = 1, then the model is just like a deterministic one.
    """
    def __init__(self,
                env:RFPRecourseEnv,
                *args,
                writer=None,
                planning_horizon = 4 * 24,
                guideline = "production_value",
                hourly_model_class=StochasticRecourseModel,
                solver='gurobi',
                documentation=False,
                n_sims=2,
                n_scenarios=1,
                da_model_type="non-recourse DA",
                **kwargs):
        super().__init__(env, writer, planning_horizon, guideline,
                         hourly_model_class, solver, documentation,
                         n_sims=n_sims, n_scenarios=n_scenarios, **kwargs)
        self.da_model_type = da_model_type
        self.da_bid_model = StochasticRecourseModel(rfp = self.env.rfp,
                                                planning_horizon = self.planning_horizon,
                                                decision_horizon = self.decision_horizon,
                                                solver = solver,
                                                guideline = self.guideline,
                                                allow_spot_buy = self.allow_spot_buy,
                                                n_scenarios=n_scenarios,
                                                model_type=da_model_type,
                                                **kwargs,
                                                )
        self.hourly_model = StochasticRecourseModel(rfp = self.env.rfp,
                                        planning_horizon = self.planning_horizon,
                                        decision_horizon = self.decision_horizon,
                                        solver = solver,
                                        guideline = self.guideline,
                                        allow_spot_buy = self.allow_spot_buy,
                                        n_scenarios=n_scenarios,
                                        model_type="non-recourse flows",
                                        **kwargs,
                                        )
        self.previous_da_decisions = None # np.ndarray(24)

    def _get_supplier_cf(self, obs, forecasts):
        supplier_cf = {}
        for s in range(len(forecasts)):
            wind_profile    = self.env.wind_mapper(forecasts[s]['wind'])
            solar_profile   = self.env.solar_mapper(forecasts[s]['solar'])

            for ix, ppa_name in enumerate(self.env.ppa_names):
                ppa = self.env.rfp.get_ppa(ppa_name)
                forecast_profile = np.ones(self.planning_horizon)
                if ppa.parameters.get("consumes") == 'wind':
                    forecast_profile = wind_profile.values
                    forecast_profile[:self.env.ppa_context_space.shape[0]-12] = obs['context']['ppas'][12:,ix]
                elif ppa.parameters.get("consumes") == 'solar':
                    forecast_profile = solar_profile.values
                    forecast_profile[:self.env.ppa_context_space.shape[0]-12] = obs['context']['ppas'][12:,ix]
                supplier_cf = {**supplier_cf, **{(ppa_name, s, t): forecast_profile[t] for t in range(self.planning_horizon)}}
        return supplier_cf

    def _construct_concrete_data(self, obs, k, time):
        """ Creates time, state, and steering signal of data object. """
        data = super()._construct_concrete_data(obs, time)
        if k > 0: # Then we have 12 hours more to consider than the super() setup.
            time_index = pd.to_datetime(pd.date_range(-pd.Timedelta(12,'h')+time, time+pd.Timedelta(self.planning_horizon-1,'h'),freq='h'), utc=True)
            datetime_data = {t: time_index[t] for t in range(self.planning_horizon+12)}
            data[None]['T_datetime'] = datetime_data
        return data

    def _create_data_dict_for_bidding(self, obs, k, time):
        data = self._construct_concrete_data(obs, k, time)

        # & Known at current time: PPA volumes, 12 hours of prices.
        # Returns price forecasts for D and D+:
        forecasts, electricity_price_forecasts = self._get_forecasts_and_electricity(time)
        full_electricity_price_forecasts = {}
        # Returns PPA volumes for D and forecasts for D+:
        supplier_cf = self._get_supplier_cf(obs, forecasts)
        full_supplier_cf = {}
        if k > 0:
            # We have to insert the first 12 hours of information (get this from the environment)
            real_prices_D_minus_1 = list(self.env.realized_prices)[:12]
            supplier_cf_D_minus_1 = np.asarray(self.env.realized_ppa)[:,:12]
            for s in range(self.n_scenarios):
                for t in range(12):
                    full_electricity_price_forecasts[(s,t)] = real_prices_D_minus_1[t]
                    for ix, ppa in enumerate(self.env.ppa_names):
                        full_supplier_cf[(ppa, s, t)] = supplier_cf_D_minus_1[ix,t]
                for t in range(12, 12+self.planning_horizon):
                    full_electricity_price_forecasts[(s,t)] = electricity_price_forecasts[(s,t-12)]
                    for ix, ppa in enumerate(self.env.ppa_names):
                        full_supplier_cf[(ppa, s, t)] = supplier_cf[(ppa, s, t-12)]
            fixed_da = {t: int(t<12) for t in range(self.planning_horizon+12)}
            cleared_power = {t: self.previous_da_decisions[t] for t in range(12)}
        else:
            full_supplier_cf = supplier_cf
            full_electricity_price_forecasts = electricity_price_forecasts
            fixed_da = {t: 0 for t in range(self.planning_horizon)}
            cleared_power = {}
        data[None]["supplier_cf"] = full_supplier_cf
        data[None]["electricity_price"] = full_electricity_price_forecasts
        data[None]["fixed_da"] = fixed_da
        data[None]["cleared_power"] = cleared_power
        
        return data

    def _bid_and_clear_dayahead(self, obs, k, time:pd.Timestamp, info:dict):
        data = self._create_data_dict_for_bidding(obs, k, time)

        if k == 0:
            self.da_bid_model.fixed_horizon = 0
            self.da_bid_model.initialize_model()
        if k == 1:
            self.da_bid_model.fixed_horizon = 12
            self.da_bid_model.initialize_model()
        
        self.da_bid_model.build_concrete_instance(data=data)
        self.da_bid_model.run(verbose=False)

        # * Here we clear the DA market:
        # If it is the first day, we only get 24 DA volumes, otherwise 36:
        newly_realized_prices = list(self.env.realized_prices)[12:36]
        if self.da_model_type == "recourse DA":
            desired_volumes = self.da_bid_model.get_da_volumes() # List of lists
            accepted_volumes = []
            for t in range(self.decision_horizon):
                price_volume_pairs = [[data[None]["electricity_price"][(s,t+self.da_bid_model.fixed_horizon)], desired_volumes[s][t]] for s in self.da_bid_model.inst.S]
                sorted_bids = np.asarray(sorted(price_volume_pairs)) # Sorts ascending by first index (price)
                sorted_prices = sorted_bids[:,0]
                sorted_volumes = sorted_bids[:,1]
                accepted_volume = sorted_volumes[min(len(sorted_volumes)-1,sum(sorted_prices<newly_realized_prices[t]))]
                accepted_volumes.append(accepted_volume)
            if self.documentation:
                fig, ax = plt.subplots(figsize=(16,12))
                plt.title(f"Bid, realized price, and resulting Day-Ahead Power")
                min_price, max_price = 100, 0
                min_volume, max_volume = 0, 1
                for t in range(self.decision_horizon):
                    rp = newly_realized_prices[t]
                    price_volume_pairs = [[data[None]["electricity_price"][(s,t+self.da_bid_model.fixed_horizon)], desired_volumes[s][t]] for s in self.da_bid_model.inst.S]
                    sorted_bids = np.asarray(sorted(price_volume_pairs)) # Sorts ascending by first index (price)
                    prices = sorted_bids[:,0]
                    max_price = max(max_price, max(prices))
                    min_price = min(min_price, min(prices))
                    volumes = sorted_bids[:,1]
                    max_volume = max(max_volume, max(volumes))
                    min_volume = min(min_volume, min(volumes))
                    if t == 12:
                        lbl = "Bidding curves"
                    else:
                        lbl = ""
                    ax.step([-500]+list(prices), [volumes[0]]+list(volumes), label=lbl, color=(0.1,1/self.decision_horizon*t,0.1))
                    ax.annotate(str(t), (rp, accepted_volumes[t]), color="red")
                    # ax.axvline(rp, color="red", linestyle="--", label="Cleared Price")
                    # ax.axhline(accepted_volumes[t], color="green", linestyle="-.", label="Power Bought")
                ax.scatter(10000,0,color="red",marker="x",label="Clearing")
                ax.axvline(self.ammonia_strike_price/self.electricity_consumption["ammonia"], color="black", alpha=0.7, linestyle="--", label="Internal Strike Price (NH3)")
                plt.xlim(min_price-0.05*np.abs(min_price), max_price+0.05*np.abs(max_price))
                plt.ylim(min_volume-0.05*np.abs(min_volume), max_volume+0.05*np.abs(max_volume))
                plt.ylabel("Day ahead bid buy (MW)")
                plt.xlabel("€/MWh")
                plt.grid(True)
                plt.legend()
                plt.savefig(f'documentation/heuristic_agent/recourseDA_bidding_agent.png')
                plt.close()
        else:
            accepted_volumes = self.da_bid_model.get_da_volumes() #
        
        return accepted_volumes, newly_realized_prices

    def _fix_hourly_decisions(self, obs, k, time, info, data=None):
        if k == 0:
            self.hourly_model.decision_horizon = 12
            self.hourly_model.fixed_horizon = 0
            self.hourly_model.initialize_model()
        elif k == 1:
            self.hourly_model.decision_horizon = self.decision_horizon
            self.hourly_model.fixed_horizon = 12
            self.hourly_model.initialize_model()
        if info.get("terminates_next", False):
            self.hourly_model.decision_horizon = self.decision_horizon + 12
            self.hourly_model.fixed_horizon = 12
            self.hourly_model.initialize_model()
        self.hourly_model.build_concrete_instance(data=data)
        return self._run_hourly_model(obs, time, info)

    def _solve_hourly_decisions(self, obs, k, time:pd.Timestamp, info:dict):
        # Accepted DA volumes and realized prices for the following day.
        # We are bidding at D-1 12:00 and realize the DA market for D 00:00-24:00:
        newly_accepted_volumes, newly_realized_prices = self._bid_and_clear_dayahead(obs, k, time, info)
        
        data = self._create_data_dict_for_bidding(obs, k, time)
        # & Now we update the data with the realized prices and DA volumes for day D.
        for s in range(self.n_scenarios):
            for t in range(self.decision_horizon):
                t_ = t+12 if k>0 else t
                data[None]["electricity_price"][(s,t_)] = newly_realized_prices[t]
        data[None]["fixed_da"] = {t: int(t<self.decision_horizon + 12 * (k>0))
                                  for t in range(self.planning_horizon + 12 * (k>0))}
        data[None]["cleared_power"] = {**data[None]["cleared_power"],
                                       **{(t + 12 * (k>0)): vol for t, vol in enumerate(newly_accepted_volumes)}}
        
        # 2. Get recourse decisions using cleared bids.
        hourly_decisions = self._fix_hourly_decisions(obs, k, time, info, data=data)

        self.previous_da_decisions = np.asarray(newly_accepted_volumes[-12:])

        return hourly_decisions

    def pi(self, obs, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        time = info["time"]
        if time.day % 15 == 1 and self.guideline == 'production_value': # We do not expect big changes in strike price throughout the year - update two times a month.
            self.ammonia_strike_price = self._estimate_strike_price(obs=obs, time=time, info=info, n_sims=self.n_sims, metric='mean')
        
        actions = self._solve_hourly_decisions(obs=obs, k=k, time=time, info=info) # Day-ahead solving

        self._update_logbook()

        return np.asarray(actions)

    def __repr__(self):
        rep = self.__class__.__name__ + str(self.n_scenarios) 
        if self.da_model_type == "recourse DA":
            rep += "_DAbidding"
        return rep


class StrikePriceBiddingAgent(RecourseAgent):
    """ Bids a bidding curve based on estimated strike prices.
    """
    def __init__(self, 
                 env:RFPRecourseEnv, 
                 *args,
                 writer=None,
                 planning_horizon:int=4*24,
                 guideline:str|None = "production_value",
                 hourly_model_class=StochasticRecourseModel,
                 solver='gurobi',
                 documentation=False,
                 n_strike_prices=3,
                 n_scenarios=1,
                 n_sims=2,
                 **kwargs):
        super().__init__(env, *args, writer=writer, planning_horizon=planning_horizon,
                         guideline=guideline, hourly_model_class=hourly_model_class,
                         solver=solver, documentation=documentation, n_scenarios=n_scenarios,
                         n_sims=n_sims, **kwargs)
        self.n_strike_prices=n_strike_prices
        if self.n_strike_prices > 1:
            self.n_sims = self.n_strike_prices
        self.ammonia_strike_price_list = None
        self.logbook['ammonia_strike_price_list'] = []
        self.gcp_cap = self.env.rfp.get_component("Grid Connection Point").parameters.get("capacity")
    
    def _update_logbook(self):
        super()._update_logbook()
        self.logbook['ammonia_strike_price_list'].append(self.ammonia_strike_price_list)

    def _bid_and_clear_dayahead(self, obs, k, time:pd.Timestamp, info:dict):
        ppa_power = np.sum(obs["context"]["ppas"] * self.env.ppa_context_space.high, axis=1)[12:]
        T = ppa_power.shape[0]
        if self.n_strike_prices == 1:
            strike_prices = [self.ammonia_strike_price]
        else:
            strike_prices = self.ammonia_strike_price_list
        prices = np.concatenate(([-500], np.sort(strike_prices)/self.electricity_consumption["ammonia"])) # We sell all if it is above our max estimated strike price. We buy all if it is below.

        def _interpolate_volumes(t):
            max_volume = (self.gcp_cap - ppa_power[t]) * self.allow_spot_buy # How much we can max buy
            min_volume = -ppa_power[t] # How much we can max sell (negative value because of convention)
            return np.linspace(max_volume, min_volume, len(prices))
        volumes = np.asarray([_interpolate_volumes(t) for t in range(T)])
        
        # Clear market:
        real_prices = np.asarray(list(self.env.realized_prices)[12:])
        realized_idxs = np.asarray([sum(prices<real_prices[t])-1 for t in range(T)])
        # We now calculate the accepted volumes for the day ahead market based on the bid curves
        # the volumes are limited so we cannot sell more than we have from our PPA or buy more than we have available capacity at our GCP: 
        accepted_volumes = np.asarray([volumes[t, realized_idxs[t]] for t in range(T)]) # Positive is buy, negative is sell.
        
        if self.documentation:
            t=15
            rp = real_prices[t]
            plt.title(f"Bid, realized price, and resulting Day-Ahead Power ({t}:00)")
            plt.step(list(prices) + [4000], [volumes[t][0]] + list(volumes[t]), label="Bidding curve")
            plt.axvline(rp, color="red", linestyle="--", label="Cleared Price")
            plt.axhline(accepted_volumes[t], color="green", linestyle="-.", label="Power Bought")
            plt.xlim(min(60,rp*0.9, prices[1]*0.8),max(120, rp*1.1, prices[-1]*1.2))
            plt.ylabel("Day ahead bid buy (MW)")
            plt.xlabel("€/MWh")
            plt.grid(True)
            plt.legend()
            plt.savefig('documentation/heuristic_agent/sp_bidding_agent.png')
            plt.close()

        self.previous_da_decisions = np.asarray(accepted_volumes[-12:])

        return accepted_volumes, real_prices

    def pi(self, obs, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        time = info["time"]
        if time.day % 15 == 1: # We do not expect big changes in strike price throughout the year - update two times a month.
            self.ammonia_strike_price_list = self._estimate_strike_price(obs=obs, time=time, info=info, n_sims=self.n_sims, metric=None)
            self.ammonia_strike_price = self._get_metric(self.ammonia_strike_price_list,metric='mean')

        actions = self._solve_hourly_decisions(obs=obs, k=k, time=time, info=info) # Day-ahead solving

        self._update_logbook()

        return np.asarray(actions)

    def __repr__(self):
        return self.__class__.__name__ + str(self.n_scenarios) + "_SP" + str(self.n_strike_prices)


class BiddingCurveAgent(RecourseAgent):
    """ Uses a Linear Decision Rule model to learn a bidding curve mapping features to bid volumes.
    The features used are bias term, price forecast, PPA power forecast, and optionally realized prices.
    """
    def __init__(self, 
                 env:RFPRecourseEnv|RFPYearEnv, 
                 *args, 
                 writer=None,
                 planning_horizon:int=4*24,
                 guideline:str|None = "production_value",
                 hourly_model_class=StochasticRecourseModel,
                 solver='gurobi',
                 documentation=False,
                 n_scenarios=1,
                 n_features=3,
                 n_price_domains=1,
                 domain_prices=[],
                 price_steps=30, # Max 200
                 mode="train",
                 no_train=False,
                 **kwargs):
        super().__init__(env, *args, writer=writer, planning_horizon=planning_horizon,
                         guideline=guideline, hourly_model_class=hourly_model_class,
                         n_scenarios=n_scenarios, solver=solver, documentation=documentation, **kwargs)
        
        self.train_model = DecisionRuleModel(rfp = self.env.rfp, planning_horizon = self.decision_horizon, 
                                               decision_horizon = self.decision_horizon, solver = solver,
                                               allow_spot_buy = self.allow_spot_buy, guideline=guideline,
                                               n_features=n_features+1, n_price_domains=n_price_domains, domain_prices=domain_prices,
                                               **kwargs,)
        self.train_model.initialize_model()
        
        self.gcp_cap = self.env.rfp.get_component("Grid Connection Point").parameters.get("capacity") / self.env.rfp.get_component("Grid Connection Point").parameters.get("rate")
        self.ammonia_strike_price_list = None
        self.logbook['ammonia_strike_price_list'] = []
        self.n_features = n_features
        self.price_steps = price_steps
        self.n_price_domains = n_price_domains
        self.domain_prices = np.asarray(domain_prices, dtype=float)
        self.max_seen_price = 220
        self.min_seen_price = -2
        self.mode = mode # Either "eval" or "train"
        self.weights = None
        self.steps = 0
        self.alpha = lambda steps: min(1,max(1/(steps/8760), 0.05))
        self.memory = None
        self.no_train = no_train

    def _get_feature_set(self, obs):
        price_forecast = obs["context"]["prices"]
        bias = np.ones(len(price_forecast))
        ppa_power = np.sum(obs["context"]["ppas"] * self.env.ppa_context_space.high, axis=1)[-self.decision_horizon:]
        feature_array = np.concatenate((np.asarray([bias]).T, np.asarray([price_forecast]).T, np.asarray([ppa_power]).T), axis=1)
        
        real_prices = list(self.env.realized_prices)[-self.decision_horizon:]
        if self.mode == "train":
            feature_array = np.concatenate((feature_array, np.asarray([real_prices]).T), axis=1)
            assert feature_array.shape[1] == self.train_model.n_features, "No match between constructed feature set and decided features for training"
        else:
            assert feature_array.shape[1] == self.n_features, "No match between constructed feature set and decided features for eval"
        return feature_array, ppa_power, real_prices

    def _get_supplier_cf(self, obs, forecasts=None):
        if self.mode == "train":
            supplier_cf = {}
            for ix, ppa_name in enumerate(self.env.ppa_names):
                forecast_profile = obs['context']['ppas'][-self.decision_horizon:, ix]
                supplier_cf = {**supplier_cf, **{(ppa_name, t): forecast_profile[t] for t in range(self.decision_horizon)}}
            return supplier_cf
        else:
            return super()._get_supplier_cf(obs, forecasts)

    def train(self, obs, a, r, obs_p, done=False, info_s=None, info_sp=None):
        if self.no_train:
            feature_array, ppa_power, real_prices = self._get_feature_set(obs)
            self.max_seen_price = max(self.max_seen_price, np.max(real_prices))
            self.min_seen_price = min(self.min_seen_price, np.min(real_prices))
        else:
            self.mode = "train"
            self.steps += self.decision_horizon
            time = info_s["time"]

            data = self._construct_concrete_data(obs=obs, k=0, time=time)

            # Change the time index from spanning the planning horizon to only the decision horizon:
            data[None]["T_datetime"] = {t: data[None]["T_datetime"][t] for t in range(self.decision_horizon)} 

            # Construct feature data and updated price regime transition prices
            feature_array, ppa_power, real_prices = self._get_feature_set(obs)
            T, n_features = feature_array.shape
            feature_dict = {(f,t): feature_array[t,f] for t in range(T) for f in range(n_features)}
            data[None]["feature_data"] = feature_dict
            if self.n_price_domains > 1:
                data[None]["domain_prices"] = {key:val for key,val in enumerate(self.domain_prices)}

            supplier_cf = self._get_supplier_cf(obs)
            electricity_price = {t: real_prices[t] for t in range(self.decision_horizon)}
            data[None]["supplier_cf"] = supplier_cf
            data[None]["electricity_price"] = electricity_price

            # Solve hourly LP model estimating optimal model weights for the day:
            self.train_model.build_concrete_instance(data=data)
            self.train_model.run(verbose=False)

            # Update the model logic:
            self.max_seen_price = max(self.max_seen_price, np.max(real_prices))
            self.min_seen_price = min(self.min_seen_price, np.min(real_prices))
            w, trunc = self.train_model.get_weights()
            if trunc == False:
                if self.weights is None:
                    self.weights = w
                else:
                    self.weights = self.weights * (1-self.alpha(self.steps)) + w * self.alpha(self.steps)
            else:
                self._save_obs_for_debug(obs, info_s)
                Warning(f"Could not get actions.\nTime: {time}.\nState at termination: {obs['state']}.")

    def _bid_and_clear_dayahead(self, obs, k, time:pd.Timestamp, info:dict):
        # 1. Get DA cleared power using bidding strategy from self.weights
        feature_array, ppa_power, real_prices = self._get_feature_set(obs)
        T = feature_array.shape[0]
        get_hour_of_day = lambda t_: pd.to_datetime(time + pd.Timedelta(t_,'h'),utc=True).hour
        intercepts = np.asarray([[self.weights[_pd,:-1, get_hour_of_day(t)] @ feature_array[t,:]
                                for t in range(T)]
                                for _pd in range(self.n_price_domains)])
        slopes = np.asarray([[self.weights[_pd, -1, get_hour_of_day(t)]
                                for t in range(T)]
                                for _pd in range(self.n_price_domains)])
        prices = np.concatenate(([-500], np.linspace(self.min_seen_price,self.max_seen_price,self.price_steps-2), [4000]))
        # Ensure that we cannot sell more power than we have available from our PPAs,
        # Ensure that we cannot buy more power than our grid connection capacity minus our PPA power.
        volumes = np.asarray([[np.clip(
                        intercepts[sum(self.domain_prices < price),t] + slopes[sum(self.domain_prices < price),t] * price,
                        a_min=-ppa_power[t], a_max=(self.gcp_cap - ppa_power[t])*self.allow_spot_buy)
                    for price in prices] for t in range(T)])
        realized_idxs = np.asarray([sum(prices<real_prices[t])-1 for t in range(T)])
        # We now calculate the accepted volumes for the day ahead market based on the bid curves
        accepted_volumes = np.asarray([volumes[t, realized_idxs[t]] for t in range(T)]) # Positive is buy, negative is sell.

        if self.documentation:
            linestyles = ['-', '--', '-.', ':']
            for t in range(T):
                rp = real_prices[t]
                plt.title(f"Bid, realized price, and resulting Day-Ahead Power ({t}:00)")
                plt.step(list(volumes[t]), list(prices), label="Bidding curve", color="blue")
                for ix in range(len(self.domain_prices)):
                    plt.axhline(self.domain_prices[ix], color="black", alpha=0.3, linestyle=linestyles[ix], label=f"Price Domain {ix+1} Border")
                plt.axhline(rp, color="red", linestyle="--", label="Cleared Price")
                plt.axvline(accepted_volumes[t], color="green", linestyle="-.", label="Power Bought")
                plt.ylim(self.min_seen_price, self.max_seen_price)
                plt.xlim(min(volumes[t]*1.1), max(volumes[t]*1.1))
                plt.ylabel("Price [€/MWh]")
                plt.xlabel("Volume - Buying [MW]")
                plt.grid(True)
                plt.legend()
                plt.savefig(f'documentation/ldr_agent/D{self.n_price_domains}_hour{t}.png')
                plt.close()

        self.previous_da_decisions = np.asarray(accepted_volumes[-12:])

        return accepted_volumes, real_prices

    def _solve_hourly_decisions(self, obs, k, time:pd.Timestamp, info:dict):
        if self.weights is None or type(self.env) == RFPYearEnv:
            return self.env.action_space.sample()
        else:
            self.mode = "eval"
            return super()._solve_hourly_decisions(obs, k, time, info)

    def pi(self, obs, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        time = info["time"]
        if time.day % 15 == 1 and self.guideline == 'production_value': # We do not expect big changes in strike price throughout the year - update two times a month.
            self.ammonia_strike_price_list = self._estimate_strike_price(obs=obs, time=time, info=info, n_sims=self.n_sims, metric=None)
            self.ammonia_strike_price = self._get_metric(self.ammonia_strike_price_list,metric='mean')
            if self.n_price_domains > 1:
                self.domain_prices[0] = self.ammonia_strike_price / self.electricity_consumption["ammonia"]
                if self.n_price_domains > 2:
                    # Consider possible general idea.
                    # But if it it to dependent on other stuff, then we for sure need to retrain the weights continuously.
                    self.domain_prices[1] = 1.4 * self.ammonia_strike_price / self.electricity_consumption["ammonia"]
        actions = self._solve_hourly_decisions(obs=obs, k=k, time=time, info=info) # Day-ahead solving

        self._update_logbook()

        return np.asarray(actions)

    def save(self, path):
        filepath = path + "/weights.csv"
        if not os.path.exists(os.path.dirname(filepath)):
            os.mkdir(os.path.dirname(filepath))
        for n in range(self.n_price_domains):
            w = self.weights[n]
            df = pd.DataFrame(w.T)
            filepath = path + f"/weights_pd{n}.csv"
            df.to_csv(filepath, index_label = self.steps) # Save the number of steps that the agent has done in the csv.
    
    def load(self, path):
        self.weights = []
        for n in range(self.n_price_domains):
            filepath = path + f"/weights_pd{n}.csv"
            df = pd.read_csv(filepath, index_col=0)
            self.weights.append(np.asarray(df).T)
        self.steps = int(df.index.name)
        self.weights = np.asarray(self.weights)

    def _update_logbook(self):
        super()._update_logbook()
        self.logbook['ammonia_strike_price_list'].append(self.ammonia_strike_price_list)

    def __repr__(self):
        return self.__class__.__name__ + str(self.n_scenarios) + "_D" + str(self.n_price_domains)


