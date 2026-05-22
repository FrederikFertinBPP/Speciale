import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from common_scripts.utils import set_plotting_style
set_plotting_style()

from model_scripts.hourly_models import HourlyDeterministicLPModel, HourlyStochasticLPModel, AggregativeModel
from common_scripts import Agent, cache_write
from model_scripts.environment import RFPShieldEnv, RFPModelActionsEnv


class HierarchicalAgent(Agent):
    guideline_options = ("production_value", "hourly_target", None) # Consider adding hourly_target for constant production.
    error_margin = 1e-5

    def __init__(self,
                 env,
                 *args,
                 writer=None,
                 planning_horizon:int = 4*24,
                 guideline:str|None = None,
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
                                                inflexible = self.env.inflexible,
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

    def _get_availability(self, frequency, hourly_index):
        """ Binary vector of availabilities. """
        H = len(hourly_index)
        a = np.zeros(H)
        for t in range(H):
            time_stamp = hourly_index[t]
            if frequency=='hourly':
                a[t] = 1
            if frequency=='daily':
                a[t] = int(time_stamp.hour == 23)
            if frequency=='monthly':
                a[t] = int(time_stamp.is_month_end and time_stamp.hour == 23)
            if frequency=='yearly':
                a[t] = int(time_stamp.is_year_end and time_stamp.hour == 23)
        return a

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
    
    def _get_forecasts_and_electricity(self, time, info):
        # Forecast prices and renewables for the planning horizon
        is_backcasting = bool(info.get("forecast_path", False))
        if is_backcasting:
            timestamp_str = time.strftime("%Y-%m-%d")
            forecasts = [pd.read_csv(info["forecast_path"] + f"forecast_{timestamp_str}.csv", index_col=0)]
        elif self.env.load_data:
            timestamp_str = time.strftime("%Y%m%d")
            forecasts = [pd.read_csv(f"{self.env.scenario_path}forecast_{timestamp_str}_0.csv")]
        else:
            forecasts = self.env.forecaster.forecast(start=time, end=time+pd.Timedelta(self.planning_horizon-1, 'h'), n_forecasts=1) # list of DFs
        # Scenario dependent electricity price forecasts:
        electricity_price_forecast = {t: forecasts[0].iloc[t]['price'] for t in range(self.planning_horizon)}
        return forecasts, electricity_price_forecast

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

    def _run_hourly_model(self, obs, time, info, data):
        def _run_model(data=data):
            self.hourly_model.build_concrete_instance(data=data)
            self.hourly_model.run(verbose=False)
            return self.hourly_model.get_actions()

        actions = _run_model(data=data)
        if actions is not None:
            return actions
        else:
            data[None]["spot_shipment"] = {None: 1} # If it was infeasible, try to allow an extraordinary spot shipment.
            actions = _run_model(data=data)
            if actions is not None:
                return actions
            else:
                self._save_obs_for_debug(obs, info)
                raise ValueError(f"Could not get actions.\nTime: {time}.\nState at termination: {obs['state']}.") 

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
                 isp_metric='mean',
                 **kwargs,
                 ):
        super().__init__(env=env, writer=writer, planning_horizon=planning_horizon, guideline=guideline, hourly_model_class=hourly_model_class, solver=solver, documentation=documentation, **kwargs)
        self.n_sims = n_sims
        self.isp_metric = isp_metric

        self.electricity_consumption = {}
        self.electricity_consumption['hydrogen'] = self.env.rfp.get_component('Electrolyzer').parameters.get('electricity_consumption', 1/50) # MWh/tH2
        self.electricity_consumption['ammonia'] = self.electricity_consumption['hydrogen'] / self.env.rfp.get_component('Haber Bosch Plant').parameters.get('rate', 5.5) # MWh/tNH3
        self.electricity_consumption['ammonia'] += self.env.rfp.get_component('Haber Bosch Plant').parameters.get('electricity_consumption', 1) # MWh/tNH3

        # Internal value of ammonia production for Ammonia1 contract.
        self.ammonia_spot_price = self.env.rfp.get_contract("AmmoniaSpot").parameters.get("price")
        self.ammonia_contract_price = self.env.rfp.get_contract('Ammonia1').parameters.get('price', 1000) # €/tNH3
        self.ammonia_strike_price = self.ammonia_contract_price
        self.ammonia_max_value = self.env.rfp.get_contract('Ammonia1').parameters.get('penalty', 4000) + self.ammonia_contract_price # €/tNH3
        self.raw_isp = self.ammonia_strike_price
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
            simulation['available_power'] = 0
            for ix, (name, ppa) in enumerate(self.env.rfp.get_ppas().items()):
                cap = ppa.parameters.get('capacity', 0)
                if ppa.parameters.get("consumes") == 'wind':
                    simulation['available_power'] += self.env.wind_mapper(simulation['wind']) * cap
                elif ppa.parameters.get("consumes") == 'solar':
                    simulation['available_power'] += self.env.solar_mapper(simulation['solar']) * cap
                else:
                    simulation['available_power'] += cap # Assumes full availability of non-variable PPAs.
            simulation['available_power'] = np.clip(simulation['available_power'], 0, gcp_cap)
            # simulation['wind']            = self.env.wind_mapper(simulation['wind']) * self.env.wind_capacity
            # simulation['solar']           = self.env.solar_mapper(simulation['solar']) * self.env.solar_capacity
            # simulation['available_power'] = np.clip(simulation['wind'] + simulation['solar'] + self.env.get_baseload_ppa_power(),0,gcp_cap)
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
        # for ix in range(len(simulation)-1, 0, -1):
        #     p = simulation.iloc[ix]['shifted_available_power']
        #     if p<0: # Shift demand if there is not enough supply.
        #         simulation.loc[simulation.index[ix], 'shifted_available_power'] = 0
        #         simulation.loc[simulation.index[ix-1], 'shifted_available_power'] += p
        # simulation.loc[simulation.index[0], 'shifted_available_power'] = np.max([simulation.iloc[0]['shifted_available_power'], 0])
        # simulation['potential_ammonia_production'] = np.clip(simulation['shifted_available_power'] / self.electricity_consumption['ammonia'], 0,
        #                                                     self.env.rfp.get_component('Haber Bosch Plant').parameters.get('capacity', 50)) # tNH3 for every hour
        return simulation

    def _calculate_hourly_ammonia_target(self, obs, time, info):
        contract_ix = np.where(np.asarray(self.env.contract_names) == "Ammonia1")[0][0]
        storage_ix = np.where(np.asarray(self.env.storage_names) == "Ammonia Storage")[0][0]
        
        is_backcasting = bool(info.get("forecast_path", False))
        if is_backcasting:
            hourly_index = pd.to_datetime(pd.date_range(start=time, end=self.env.episode_end, freq='h'), utc=True)
            target_frequency = self.env.rfp.get_contract("Ammonia1").parameters.get("target_frequency")
            availability = self._get_availability(target_frequency, hourly_index)
            hours_to_deadline = next((ix+1 for ix, x in enumerate(availability) if x > 0), 1)
        else:
            hours_to_deadline = (self.env.episode_end - time).value / 3600 * 1e-9 + 1 # The timedelta is in nanoseconds, we convert to hours inclusive (+1)
        self.ammonia_hourly_target = (self.env.contract_state_space.high[contract_ix] - obs['state']['contracts'][contract_ix] - obs['state']['storages'][storage_ix]) / hours_to_deadline # tNH3/h
    
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

    def _estimate_strike_price_old(self, obs, time:pd.Timestamp, info:dict, n_sims=3, metric='mean'):
        is_backcasting = bool(info.get("forecast_path", False))
        if is_backcasting:
            n_sims = 1
            timestamp_str = time.strftime("%Y-%m-%d")
            year_simulations = [pd.read_csv(info["forecast_path"] + f"long-term-sim_{timestamp_str}.csv", index_col=0)]
        elif self.env.load_data:
            # if n_sims > 5:
            #     print("Only generated 5 simulations year ahead - setting number of sims to 5.")
            n_sims = 5
            timestamp_str = time.strftime("%Y%m%d")
            # updated_timestamp_str = timestamp_str[:-2] + '01'
            # hours_extra = (int(timestamp_str) - int(updated_timestamp_str))*24 # We have only generate for the start of each month. So we remove days which have already passed.
            # year_simulations = [pd.read_csv(f"{self.env.scenario_path}year_sim_{updated_timestamp_str}_{ix}.csv").iloc[hours_extra:] for ix in range(n_sims)]
            year_simulations = [pd.read_csv(f"{self.env.scenario_path}year_sim_{timestamp_str}_{ix}.csv") for ix in range(n_sims)]
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
            max_x = 0
            ax.set_ylabel(r"€/MWh$_e$")
            ax.set_xlabel(r"t/NH$_3$")
        for sim_ix, simulation in enumerate(year_simulations):
            simulation = self._get_forecast_available_power(simulation)
            target_volume_current = dict(zip(freq_options, np.zeros(len(freq_options))))
            target_volume_normal = dict(zip(freq_options, np.zeros(len(freq_options))))
            strike_price = 0
            for freq, rank in freq_rank.items():
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
                lbl1 = "Projected Production-Weighted Price-Duration Curves" if sim_ix==0 else ""
                lbl2 = "Internal Strike Price Estimates" if sim_ix==0 else ""
                ax.plot(_sorted['cumulative_prod'], _sorted['price'], color='black', label=lbl1, alpha=0.7)
                ax.axhline(y=strike_prices[sim_ix], label=lbl2, color='blue', linestyle='--', alpha=0.7)
                max_x = max(max_x, _sorted['cumulative_prod'].values[-1])
        if self.documentation:
            ax.set_xlim(0, max_x)
            ax.axhline(y=np.mean(strike_prices), color='purple', linestyle='--', lw=3, alpha=1,
                       label=r"ISP$_e$ = "+str(round(np.mean(strike_prices),1))+" €/MWh")
            ax.axvline(x=target_volume_current['yearly'], label="Missing Contracted Ammonia Production", color='red', linestyle='--')
            ax.legend()
            ax.set_title(f"Estimate on {str(time.date())}")
            plt.savefig(f"documentation/heuristic_agent/strike_price_visualization/{str(time.date())}")
            plt.close()
        
        # Dependent on metric, return an estimate of strike price:
        sp = self._get_metric(strike_prices, metric=metric) * self.electricity_consumption['ammonia']
        if isinstance(sp, np.ndarray): # The lowest possible internal value of ammonia is the spot value (not completely true with storage limits, but good enough)
            sp[sp<self.ammonia_spot_price] = self.ammonia_spot_price
        else:
            sp = max(self.ammonia_spot_price, sp)
        return sp

    def _estimate_strike_price(self, obs, time:pd.Timestamp, info:dict, n_sims=3, metric='mean'):
        try:
            # -------------------- Produce projections -------------------- #
            is_backcasting = bool(info.get("forecast_path", False))
            if is_backcasting:
                n_sims = 1
                timestamp_str = time.strftime("%Y-%m-%d")
                year_simulations = [pd.read_csv(info["forecast_path"] + f"long-term-sim_{timestamp_str}.csv", index_col=0)]
            elif self.env.load_data:
                # if n_sims > 5:
                #     print("Only generated 5 simulations year ahead - setting number of sims to 5.")
                n_sims = 5
                timestamp_str = time.strftime("%Y%m%d")
                # updated_timestamp_str = timestamp_str[:-2] + '01'
                # hours_extra = (int(timestamp_str) - int(updated_timestamp_str))*24 # We have only generate for the start of each month. So we remove days which have already passed.
                # year_simulations = [pd.read_csv(f"{self.env.scenario_path}year_sim_{updated_timestamp_str}_{ix}.csv").iloc[hours_extra:] for ix in range(n_sims)]
                year_simulations = [pd.read_csv(f"{self.env.scenario_path}year_sim_{timestamp_str}_{ix}.csv") for ix in range(n_sims)]
            else:
                year_simulations = self.env.forecaster.simulate_year_ahead(start = time, n_sims=n_sims) # Creates a list of n_sims simulated year-ahead forecasts (pd.DataFrame with hourly index and 'price', 'wind', 'solar' columns)
            
            simulation_length = len(year_simulations[0])
            hourly_index = pd.to_datetime(pd.date_range(start=time, end=time+pd.Timedelta(simulation_length-1, 'h'), freq='h'), utc=True)
            freq_options = self.env.rfp.frequency_options

            # -------------------- Get current status -------------------- #
            allocated_contract_status = dict(zip(self.env.contract_names, obs['state']['contracts']))
            storage_levels = dict(zip([storage.parameters.get("consumes")
                                    for storage in self.env.rfp.get_storages().values()],
                                    obs['state']['storages']))
            stored_energy = sum(storage_levels[resource] * self.electricity_consumption[resource]
                                    for resource in storage_levels.keys())

            # -------------------- Calculate current shortfall -------------------- #
            target_energy_standard  = dict(zip(freq_options, np.zeros(len(freq_options)))) # Power needed in MWh
            target_energy_current = target_energy_standard.copy()
            limits = ["min_volume", "max_volume"]
            target_energy_standard_limits = {lim: target_energy_standard.copy() for lim in limits}
            target_energy_current_limits = {lim: target_energy_standard.copy() for lim in limits}
            for ix, name in enumerate(self.env.contract_names):
                contract = self.env.rfp.get_contract(name)
                target_frequency = contract.target_frequency
                if target_frequency is not None:
                    resource = contract.parameters.get("resource")
                    target_volume = contract.parameters.get("volume")
                    target_energy_standard[target_frequency] += target_volume * self.electricity_consumption[resource]
                    target_energy_current[target_frequency] += max(0, target_volume-allocated_contract_status[name]) * self.electricity_consumption[resource]
                    for lim in limits:
                        _volume = contract.parameters.get(lim, target_volume)
                        target_energy_standard_limits[lim][target_frequency] += _volume * self.electricity_consumption[resource]
                        target_energy_current_limits[lim][target_frequency] += max(0, _volume-allocated_contract_status[name]) * self.electricity_consumption[resource]

            deadline_ts = {freq: self._get_availability(freq, hourly_index)
                        for freq in freq_options}
            first_target = {freq: next((ix for ix, x in enumerate(availability) if x > 0), 1)
                                    for freq, availability in deadline_ts.items()}
            energy_to_allocate = {freq: target_energy_standard[freq]*deadline_ts[freq]
                                for freq in freq_options}
            energy_to_allocate_limits = {lim: {freq: target_energy_standard_limits[lim][freq]*deadline_ts[freq]
                                for freq in freq_options} for lim in limits}
            for freq in freq_options:
                energy_to_allocate[freq][first_target[freq]] = target_energy_current[freq]
                for lim in limits:
                    energy_to_allocate_limits[lim][freq][first_target[freq]] = target_energy_current_limits[lim][freq]

            # -------------------- Calculate internal strike price - based on contracts -------------------- #
            strike_prices = np.zeros(n_sims)
            strike_prices_hydrogen = np.zeros(n_sims)
            deadline_timeindices = np.where((energy_to_allocate["monthly"] > 0) | (energy_to_allocate["yearly"] > 0))[0]

            year_simulations_appended = [self._get_forecast_available_power(simulation) for simulation in year_simulations]

            for t_ix in deadline_timeindices:
                if t_ix < self.planning_horizon:
                    continue
                energy_needed = np.asarray(list(energy_to_allocate.values()))[1:,:(t_ix+1)].sum() # Excludes hourly contract energy need, which is subtracted from every hour instead.
                hourly_energy_needed = np.asarray(list(energy_to_allocate.values()))[0,:(t_ix+1)].sum()
                energy_needed_limits = {lim: np.asarray(list(energy_to_allocate_limits[lim].values()))[1:,:(t_ix+1)].sum() for lim in limits}
                hourly_energy_needed_limits = {lim: np.asarray(list(energy_to_allocate_limits[lim].values()))[0,:(t_ix+1)].sum() for lim in limits}
                missing_energy = max(0, energy_needed - stored_energy)
                missing_energy_limits = {lim: max(0, energy_needed_limits[lim] - stored_energy) for lim in limits}
                for sim_ix, simulation in enumerate(year_simulations_appended):
                    sim_slice = simulation.iloc[:(t_ix+1)]
                    _sorted = sim_slice.sort_values(by='price', ascending=True)
                    
                    # Now we can estimate the strike price of ammonia production:
                    _sorted['cumulative_energy'] = np.cumsum(_sorted['available_power_for_ammonia'])
                    n_hours = len(_sorted)
                    strike_idx_hour = max(0, np.sum(_sorted['cumulative_energy'] <= missing_energy))
                    if strike_idx_hour >= n_hours*0.98: # If 98% of the hours are needed, then we incentivise non-stop max production of ammonia.
                        sp = self.ammonia_max_value / self.electricity_consumption['ammonia']
                    else: # Otherwise we get a strike price estimate based on the marginal hour needed to produce the missing energy.
                        sp = _sorted.iloc[strike_idx_hour]['price']
                    
                    strike_idx_hour_limits = {lim: max(0, np.sum(_sorted['cumulative_energy'] <= missing_energy_limits[lim])) for lim in limits}
                    sp_limits = {}
                    for lim in limits:
                        if strike_idx_hour_limits[lim] >= n_hours*0.98:
                            sp_limits[lim] = self.ammonia_max_value / self.electricity_consumption['ammonia']
                        else: # Otherwise we get a strike price estimate based on the marginal hour needed to produce the missing energy.
                            sp_limits[lim] = _sorted.iloc[strike_idx_hour_limits[lim]]['price']
                    sp_clipped = np.clip(self.ammonia_contract_price / self.electricity_consumption['ammonia'], sp_limits['min_volume'], sp_limits['max_volume'])
                    # The contract/deadline encouraging the highest ISP is the one we should be considering in order to meet all deadlines:
                    strike_prices[sim_ix] = max(strike_prices[sim_ix], sp_clipped)
                    
                    # We can alternatively get an estimate of the strike price of the electrolyzer:
                    _sorted['hourly_cumulative_energy'] = np.cumsum(_sorted['available_power'])
                    hourly_strike_idx = max(0, np.sum(_sorted['hourly_cumulative_energy'] <= missing_energy + hourly_energy_needed) - 1)
                    hourly_sp = _sorted.iloc[hourly_strike_idx]['price']
                    strike_prices_hydrogen[sim_ix] = max(strike_prices_hydrogen[sim_ix], hourly_sp)
            
            # -------------------- Calculate internal strike price - capped by storages -------------------- #
            # Storage limitations represent an upper bound for value of production:
            storage_capacities = dict(zip([storage.parameters.get("consumes")
                                    for storage in self.env.rfp.get_storages().values()],
                                    [storage.parameters.get("capacity")
                                    for storage in self.env.rfp.get_storages().values()]))
            storage_energy_capacity = sum(storage_capacities[resource] * self.electricity_consumption[resource]
                                    for resource in storage_capacities.keys())
            available_storage = storage_energy_capacity - stored_energy
            t_ix = first_target["monthly"] # Next ammonia shipment.
            if t_ix >= self.planning_horizon:
                for sim_ix, simulation in enumerate(year_simulations_appended):
                    sim_slice = simulation.iloc[:(t_ix+1)]
                    _sorted = sim_slice.sort_values(by='price', ascending=True)
                    _sorted['cumulative_energy'] = np.cumsum(_sorted['available_power_for_ammonia'])
                    n_hours = len(_sorted)
                    strike_idx_hour = max(0, np.sum(_sorted['cumulative_energy'] <= available_storage)-1)
                    if strike_idx_hour <= n_hours*0.5:
                        sp = _sorted.iloc[strike_idx_hour]['price']
                        strike_prices[sim_ix] = min(strike_prices[sim_ix], sp)

            # -------------------- Convert, rationalize and save ISP -------------------- #
            strike_prices_ammonia = strike_prices - strike_prices_hydrogen
            # Dependent on metric, return an estimate of strike price:
            isp = self._get_metric(strike_prices, metric=metric) * self.electricity_consumption['ammonia']
            self.raw_isp = isp
            isp = np.clip(isp, self.ammonia_spot_price, self.ammonia_max_value)
        except FileNotFoundError:
            print("Could not find simulation file for strike price estimation. Setting strike price to latest estimate.")
            isp = self.ammonia_strike_price
        return isp

    def _solve_hourly_decisions(self, obs, time:pd.Timestamp, info:dict):
        data = self._construct_concrete_data(obs, time)
        if self.guideline == 'production_value':
            data[None]["shipment_value"] = {"AmmoniaSpot": self.ammonia_spot_price - self.raw_isp, 
                                             "Ammonia1": self.ammonia_strike_price}
        
        forecasts, electricity_price_forecast = self._get_forecasts_and_electricity(time, info)
        supplier_cf = self._get_supplier_cf(obs, forecasts[0])
        data[None]["supplier_cf"] = supplier_cf
        data[None]["electricity_price"] = electricity_price_forecast

        # Solve hourly LP model
        return self._run_hourly_model(obs, time, info, data)

    def _update_logbook(self):
        self.logbook['ammonia_strike_price'].append(self.ammonia_strike_price)
        self.logbook['ammonia_hourly_target'].append(self.ammonia_hourly_target)
        self.logbook['hydrogen_strike_price'].append(self.hydrogen_strike_price)
        self.logbook['hydrogen_hourly_target'].append(self.hydrogen_hourly_target)

    def pi(self, obs, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        time = info["time"]
        is_backcasting = bool(info.get("forecast_path", False))

        if self.guideline == 'hourly_target':
            # Constant heuristic:
            self._calculate_hourly_ammonia_target(obs, time, info)
        elif self.guideline == 'production_value':
            if (is_backcasting and k % 7 == 0) ^ (not is_backcasting and time.day_of_week == 0):
                # We do not expect dynamic changes in strike price throughout the year - updates weekly.
                self.ammonia_strike_price = self._estimate_strike_price(obs=obs, time=time, info=info, n_sims=self.n_sims, metric=self.isp_metric)

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
    
    def _get_forecasts_and_electricity(self, time, info):
        is_backcasting = bool(info.get("forecast_path", False))
        if is_backcasting:
            timestamp_str = time.strftime("%Y-%m-%d")
            forecasts = [pd.read_csv(info["forecast_path"] + f"forecast_{timestamp_str}.csv", index_col=0)]
        elif self.env.load_data:
            timestamp_str = time.strftime("%Y%m%d")
            assert self.hourly_model.n_scenarios <= 10
            forecasts = [pd.read_csv(f"{self.env.scenario_path}forecast_{timestamp_str}_{ix}.csv") for ix in range(self.hourly_model.n_scenarios)]
        else:
            forecasts = self.env.forecaster.forecast(start=time,
                                                    end=time+pd.Timedelta(self.planning_horizon-1, 'h'),
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
                                                inflexible = self.env.inflexible,
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
                                        inflexible = self.env.inflexible,
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

    def _create_data_dict_for_bidding(self, obs, k, time, info):
        data = self._construct_concrete_data(obs, k, time)

        # & Known at current time: PPA volumes, 12 hours of prices.
        # Returns price forecasts for D and D+:
        forecasts, electricity_price_forecasts = self._get_forecasts_and_electricity(time, info)
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

    def _resolve_DA_volumes(self, newly_realized_prices, data):
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
            if False:
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

            ISP = self.ammonia_strike_price / self.electricity_consumption["ammonia"]
            fig, axs = plt.subplots(4, 6, figsize=(16,12), sharex=True, sharey=True)
            plt.suptitle(f"LP-Integrated Bidding Strategy", fontweight="bold")
            axs = axs.flatten()
            max_v = 0
            for t in range(self.decision_horizon):
                ax = axs[t]
                rp = newly_realized_prices[t]
                vol = accepted_volumes[t]
                price_volume_pairs = [[data[None]["electricity_price"][(s,t+self.da_bid_model.fixed_horizon)], desired_volumes[s][t]] for s in self.da_bid_model.inst.S]
                sorted_bids = np.asarray(sorted(price_volume_pairs)) # Sorts ascending by first index (price)
                buy_profile = [(bid[0], bid[1]) for bid in sorted_bids if bid[1] > 0]
                if len(buy_profile) == 0:
                    buy_profile = [(-500,0), (-500,0)]
                sell_profile = [(bid[0], bid[1]) for bid in sorted_bids if bid[1] < 0]
                if len(sell_profile) == 0:
                    buy_profile.append((4000,buy_profile[-1][1]))
                    sell_profile = [(4000,0), (4000,0)]
                sell_profile.insert(0, (buy_profile[-1][0], sell_profile[0][1]))
                sell_profile.insert(0, (buy_profile[-1][0], 0))
                sell_profile.append((4000,sell_profile[-1][1]))
                sell_profile = np.asarray(sell_profile)
                buy_profile.insert(0, (-500, buy_profile[0][1]))
                buy_profile.append((buy_profile[-1][0], 0))                
                buy_profile = np.asarray(buy_profile)

                max_v = max(max_v, np.max(-sell_profile[:,1]), np.max(buy_profile[:,1]))
                ax.set_title(f"{t}:00-{t+1}:00", fontweight="normal")
                ax.step(-sell_profile[:,1][::-1], sell_profile[:,0][::-1], label="Selling curve", color="orange", lw=5, alpha=0.5)
                ax.step(buy_profile[:,1][::-1], buy_profile[:,0][::-1], label="Buying curve", color="blue", lw=5, alpha=0.5)
                ax.axhline(rp, color="red", linestyle="--", label="DA market clearing", lw=2)
                ax.scatter([np.abs(vol)], [rp], color="black", marker='x', s=100, label="Power traded")
                ax.set_ylim(0,180)
                ax.set_xlabel("Day-ahead volume (MW)", fontweight="normal")
                ax.set_ylabel("€/MWh", fontweight="normal")
                ax.grid(True)
            ax.set_xlim(0, max_v*1.05)
            ax.axhline(4000, color="purple", linestyle="dashdot", label=f"ISP={ISP}", lw=2)
            hl, lb = ax.get_legend_handles_labels()
            fig.legend(handles=hl, labels=lb, bbox_to_anchor=(0.5, 0.07), ncol=5, loc='upper center')
            fig.tight_layout(rect=[0,0.05,1,1])
            plt.savefig(f'documentation/heuristic_agent/recourseDA_bidding_agent_hourly.png')
            plt.close()
        
        return accepted_volumes

    def _bid_and_clear_dayahead(self, obs, k, time:pd.Timestamp, info:dict):
        data = self._create_data_dict_for_bidding(obs, k, time, info)

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
            accepted_volumes = self._resolve_DA_volumes(newly_realized_prices, data)
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
        return self._run_hourly_model(obs, time, info, data)

    def _solve_hourly_decisions(self, obs, k, time:pd.Timestamp, info:dict):
        # Accepted DA volumes and realized prices for the following day.
        # We are bidding at D-1 12:00 and realize the DA market for D 00:00-24:00:
        newly_accepted_volumes, newly_realized_prices = self._bid_and_clear_dayahead(obs, k, time, info)
        
        data = self._create_data_dict_for_bidding(obs, k, time, info)
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
        if time.day_of_week == 0 and self.guideline == 'production_value': # We do not expect big changes in strike price throughout the year - update two times a month.
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
        self.n_strike_prices = n_strike_prices
        if self.n_strike_prices > 1:
            self.n_sims = self.n_strike_prices
        self.ammonia_strike_price_list = None
        self.logbook['ammonia_strike_price_list'] = []
        self.gcp_cap = self.env.rfp.get_component("Grid Connection Point").parameters.get("capacity")
        self.min_load = 0
        if self.env.inflexible:
            for name, link in self.env.rfp.get_links().items():
                self.min_load += link.parameters.get("electricity_consumption", 0) * link.parameters.get("min_load", 0) * link.parameters.get("capacity")

    def _update_logbook(self):
        super()._update_logbook()
        self.logbook['ammonia_strike_price_list'].append(self.ammonia_strike_price_list)

    def _bid_and_clear_dayahead(self, obs, k, time:pd.Timestamp, info:dict):
        if self.n_strike_prices == 1:
            strike_prices = [self.ammonia_strike_price]
        else:
            strike_prices = self.ammonia_strike_price_list
        ISP = np.mean(strike_prices)/self.electricity_consumption["ammonia"]
        ppa_power = np.sum(obs["context"]["ppas"] * self.env.ppa_context_space.high, axis=1)[12:]
        T = ppa_power.shape[0]
        prices = np.concatenate(([-500], np.sort(strike_prices)/self.electricity_consumption["ammonia"])) # We sell all if it is above our max estimated strike price. We buy all if it is below.

        def _interpolate_volumes(t):
            max_volume = (self.gcp_cap - ppa_power[t]) * self.allow_spot_buy # How much we can max buy
            min_volume = -(ppa_power[t] - self.min_load) # How much we can max sell (negative value because of convention)
            return np.linspace(max_volume, min_volume, len(prices))
        volumes = np.asarray([_interpolate_volumes(t) for t in range(T)])
        
        # Clear market:
        real_prices = np.asarray(list(self.env.realized_prices)[12:])
        realized_idxs = np.asarray([sum(prices<real_prices[t])-1 for t in range(T)])
        # We now calculate the accepted volumes for the day ahead market based on the bid curves
        # the volumes are limited so we cannot sell more than we have from our PPA or buy more than we have available capacity at our GCP: 
        accepted_volumes = np.asarray([volumes[t, realized_idxs[t]] for t in range(T)]) # Positive is buy, negative is sell.
        
        if self.documentation and self.n_strike_prices == 1:
            t=16 # Example time index
            rp = real_prices[t]
            vol = accepted_volumes[t]
            buy_profile = np.asarray([(ISP,0), (ISP,max(volumes[t])), (-500, max(volumes[t]))])
            sell_profile = np.asarray([(ISP+0.01,0), (ISP+0.01,-min(volumes[t])), (4000, -min(volumes[t]))])

            fig, ax = plt.subplots(figsize=(12,8))
            plt.title(f"Internal Strike Price Bidding Strategy", fontweight="bold")
            plt.step(sell_profile[:,1], sell_profile[:,0], label="Selling curve", color="orange", lw=5, alpha=0.5)
            plt.step(buy_profile[:,1], buy_profile[:,0], label="Buying curve", color="blue", lw=5, alpha=0.5)
            plt.axhline(rp, color="red", linestyle="--", label="DA market clearing", lw=2)
            plt.scatter([np.abs(vol)], [rp], color="black", marker='x', s=50, label="Power bought" if vol>0 else "Power Sold")
            plt.axhline(ISP, color="purple", linestyle="dashdot", label="ISP", lw=2)
            plt.ylim(0,180)
            plt.xlim(0)
            plt.xlabel("Day-ahead volume (MW)", fontweight="bold")
            plt.ylabel("€/MWh", fontweight="bold")
            plt.grid(True)
            plt.legend()
            plt.savefig('documentation/heuristic_agent/sp_bidding_agent.png')
            plt.close()

        return accepted_volumes, real_prices

    def pi(self, obs, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        time = info["time"]
        if time.day_of_week == 0: # We do not expect big changes in strike price throughout the year - update two times a month.
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
                 price_steps=25, # Max 25 in OMIE
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
        ISP = self.ammonia_strike_price/self.electricity_consumption["ammonia"]
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
        prices = list(np.concatenate(([4000], np.linspace(self.max_seen_price,self.min_seen_price,self.price_steps-(2+len(self.domain_prices)*2)), [-500])))
        for p in self.domain_prices:
            prices.insert(sum(np.asarray(prices)>p), p+0.5)
            prices.insert(sum(np.asarray(prices)>p), p-0.5)
        # Ensure that we cannot sell more power than we have available from our PPAs,
        # Ensure that we cannot buy more power than our grid connection capacity minus our PPA power.
        buy_volumes = {}
        sell_volumes = {}
        for t in range(T):
            buy_volumes[t] = {}
            sell_volumes[t] = {}
            for price in prices:
                desired_volume = np.clip(intercepts[sum(self.domain_prices < price),t] + slopes[sum(self.domain_prices < price),t] * price,
                                         a_min=-ppa_power[t], a_max=(self.gcp_cap - ppa_power[t])*self.allow_spot_buy)
                if desired_volume >= 0:
                    buy_volumes[t][price] = desired_volume
                else:
                    sell_volumes[t][price] = desired_volume
        
        buy_cutoff_price = [prices[sum(prices > real_prices[t])-1] for t in range(T)]
        sell_cutoff_price = [prices[sum(prices > real_prices[t])] for t in range(T)]
        power_bought = [buy_volumes[t].get(buy_cutoff_price[t], 0) for t in range(T)]
        power_sold = [sell_volumes[t].get(sell_cutoff_price[t], 0) for t in range(T)]
        power_traded = np.asarray(power_bought) + np.asarray(power_sold)

        if self.documentation:
            linestyles = ['--', '-.', ':']
            t=16
            rp = real_prices[t]
            vol = power_traded[t]
            buy_profile = list(buy_volumes[t].items())
            buy_profile.insert(0, (buy_profile[0][0], 0))
            buy_profile = np.asarray(buy_profile)
            sell_profile = list(sell_volumes[t].items())
            sell_profile.append((sell_profile[-1][0], 0))
            sell_profile = np.asarray(sell_profile)

            fig, ax = plt.subplots(figsize=(12,8))
            plt.title(f"Decision Rule Bidding Strategy ({self.n_price_domains} domains)", fontweight="bold")
            plt.step(-sell_profile[:,1][::-1], sell_profile[:,0][::-1], label="Selling curve", color="orange", lw=5, alpha=0.5)
            plt.step(buy_profile[:,1], buy_profile[:,0], label="Buying curve", color="blue", lw=5, alpha=0.5)
            plt.axhline(rp, color="red", linestyle="--", label="DA market clearing", lw=2)
            plt.scatter([np.abs(vol)], [rp], color="black", marker='x', s=50, label="Power bought" if vol>0 else "Power Sold")
            plt.axhline(ISP, color="purple", linestyle="dashdot", label="ISP", lw=2)
            for j in range(len(self.domain_prices)):
                plt.axhline(self.domain_prices[j], color="black", alpha=0.3, linestyle=linestyles[j], )#label=f"Boundary Price {j}:{j+1}")

            plt.axvline(0, color='grey')
            plt.ylim(0,180)
            plt.xlim(0)
            plt.xlabel("Day-ahead volume (MW)", fontweight="bold")
            plt.ylabel("€/MWh", fontweight="bold")
            plt.grid(True)
            plt.legend()
            plt.savefig(f'documentation/heuristic_agent/dr_bidding_agent_{self.n_price_domains}_one_hour.png')
            plt.close()
            
            ts=[14,15,16,17]
            fig, axs = plt.subplots(2,2,figsize=(14,10), sharex=True, sharey=True)
            plt.suptitle(f"Decision Rule Bidding Strategy ({self.n_price_domains} domains)", fontweight="bold")
            axs = axs.flatten()
            max_v = 0
            for ix, ax in enumerate(axs):
                t = ts[ix]
                rp = real_prices[t]
                vol = power_traded[t]
                buy_profile = list(buy_volumes[t].items())
                if len(buy_profile) == 0:
                    buy_profile = [(-500,0), (-500,0)]
                else:
                    buy_profile.insert(0, (buy_profile[0][0], 0))
                buy_profile = np.asarray(buy_profile)
                sell_profile = list(sell_volumes[t].items())
                if len(sell_profile) == 0:
                    sell_profile = [(4000,0), (4000,0)]
                else:
                    sell_profile.append((sell_profile[-1][0], 0))
                sell_profile = np.asarray(sell_profile)
                max_v = max(max_v, np.max(-sell_profile[:,1]), np.max(buy_profile[:,1]))
                ax.set_title(f"{t}:00-{t+1}:00", fontweight="normal")
                ax.step(-sell_profile[:,1][::-1], sell_profile[:,0][::-1], label="Selling curve", color="orange", lw=5, alpha=0.5)
                ax.step(buy_profile[:,1], buy_profile[:,0], label="Buying curve", color="blue", lw=5, alpha=0.5)
                ax.axhline(rp, color="red", linestyle="--", label="DA market clearing", lw=2)
                ax.scatter([np.abs(vol)], [rp], color="black", marker='x', s=50, label="Power traded")
                ax.axhline(ISP, color="purple", linestyle="dashdot", label="ISP", lw=2)
                for j in range(len(self.domain_prices)):
                    ax.axhline(self.domain_prices[j], color="black", alpha=0.3, linestyle=linestyles[j], )#label=f"Boundary Price {j}:{j+1}")

                ax.axvline(0, color='grey')
                ax.set_ylim(0,180)
                ax.set_xlabel("Day-ahead volume (MW)", fontweight="normal")
                ax.set_ylabel("€/MWh", fontweight="normal")
                ax.grid(True)
            ax.set_xlim(0, max_v*1.05)
            hl, lb = ax.get_legend_handles_labels()
            fig.legend(handles=hl, labels=lb, bbox_to_anchor=(0.5, 0.07), ncol=5, loc='upper center')
            fig.tight_layout(rect=[0,0.05,1,1])
            plt.savefig(f'documentation/heuristic_agent/dr_bidding_agent_{self.n_price_domains}.png')
            plt.close()

        return power_traded, real_prices

    def _solve_hourly_decisions(self, obs, k, time:pd.Timestamp, info:dict):
        if self.weights is None or type(self.env) == RFPYearEnv:
            return self.env.action_space.sample()
        else:
            self.mode = "eval"
            return super()._solve_hourly_decisions(obs, k, time, info)

    def pi(self, obs, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        time = info["time"]
        if time.day_of_week == 0 and self.guideline == 'production_value': # We do not expect big changes in strike price throughout the year - update two times a month.
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


class AggregateFullHorizonAgent(HierarchicalAgent):
    def __init__(self,
                 env,
                 *args,
                 writer=None,
                 planning_horizon:int = 4*24,
                 hourly_model_class=AggregativeModel,
                 solver='gurobi',
                 documentation=False,
                 objective_logic=None,
                 n_sims=2,
                 **kwargs,
                 ):
        super().__init__(env, *args,
                         writer=writer, planning_horizon=planning_horizon, guideline=None, hourly_model_class=hourly_model_class,
                         solver=solver, documentation=documentation, objective_logic=objective_logic, **kwargs)
        self.n_sims = n_sims

        self.average_price_projection = [ppa.parameters.get("price") for name, ppa in self.env.rfp.get_ppas().items() if not(ppa.parameters.get("simulated"))][0]
        self.average_cf_projection = {name: ppa.parameters.get("annual_cf") for name, ppa in self.env.rfp.get_ppas().items()}
        
        self.longterm_horizon = 0
        self.offtaker_availabilities = None
        self.contract_deadlines = None

        self.logbook = {"average_price_projection": [],
                        "average_cf_projection": [],}
    
    def _estimate_longterm_uncertainties(self, obs, time, info):
        is_backcasting = bool(info.get("forecast_path", False))
        if is_backcasting:
            n_sims = 1
            timestamp_str = time.strftime("%Y-%m-%d")
            year_simulations = [pd.read_csv(info["forecast_path"] + f"long-term-sim_{timestamp_str}.csv", index_col=0)]
        elif self.env.load_data:
            # if n_sims > 5:
            #     print("Only generated 5 simulations year ahead - setting number of sims to 5.")
            n_sims = 5
            timestamp_str = time.strftime("%Y%m%d")
            # Get timestamp_str of the start of the week, since we only have estimates every week:
            # latest_projection_time = time - pd.Timedelta(time.day_of_week, unit="d")
            # updated_timestamp_str = latest_projection_time.strftime("%Y%m%d")
            # hours_extra = (int(timestamp_str) - int(updated_timestamp_str))*24 
            # year_simulations = [pd.read_csv(f"{self.env.scenario_path}year_sim_{updated_timestamp_str}_{ix}.csv").iloc[hours_extra:] for ix in range(n_sims)]
            year_simulations = [pd.read_csv(f"{self.env.scenario_path}year_sim_{timestamp_str}_{ix}.csv") for ix in range(n_sims)]
        else:
            n_sims = self.n_sims
            year_simulations = self.env.forecaster.simulate_year_ahead(start = time, n_sims=n_sims) # Creates a list of n_sims simulated year-ahead forecasts (pd.DataFrame with hourly index and 'price', 'wind', 'solar' columns)
        ts = pd.date_range(start=time, periods=len(year_simulations[0]), freq='h', tz='UTC')
        # self.longterm_horizon = max(((self.env.episode_end - time).days + 1) * 24 - self.planning_horizon, 0)
        # ts = pd.to_datetime(year_simulations[0].index)
        current_year = ts[min(self.planning_horizon,len(ts)-1)].year

        longterm_horizon = max(sum(ts.year<=current_year) - self.planning_horizon, 0)
        average_price_projection = 0
        average_cf_projection = {key: 0 for key in self.env.ppa_names}
        
        if longterm_horizon > 0:
            lth_slice = range(self.planning_horizon, self.planning_horizon+longterm_horizon)
            average_price_projection = np.mean([sim["price"].iloc[lth_slice].values
                                                for sim in year_simulations])
            for name, ppa in self.env.rfp.get_ppas().items():
                cf = 0
                for simulation in year_simulations:
                    if ppa.parameters.get("consumes") == 'wind':
                        cf += np.mean(self.env.wind_mapper(simulation['wind'].iloc[lth_slice]))
                    elif ppa.parameters.get("consumes") == 'solar':
                        cf += np.mean(self.env.solar_mapper(simulation['solar'].iloc[lth_slice]))
                    else:
                        cf += 1 # Assumes full availability of non-variable PPAs.
                average_cf_projection[name] = cf / n_sims
        
        return longterm_horizon, average_price_projection, average_cf_projection
    
    def _solve_hourly_decisions(self, obs, time:pd.Timestamp, info:dict):
        data = self._construct_concrete_data(obs, time)
        
        forecasts, electricity_price_forecast = self._get_forecasts_and_electricity(time, info)
        supplier_cf = self._get_supplier_cf(obs, forecasts[0])
        data[None]["supplier_cf"]       = supplier_cf
        data[None]["electricity_price"] = electricity_price_forecast
        
        ### Aggregative model specific data:
        if self.longterm_horizon > 0:
            data[None]["longterm_horizon"]  = {None: self.longterm_horizon}
            data[None]["longterm_price"]    = {None: self.average_price_projection}
            data[None]["longterm_cf"]       = self.average_cf_projection
            data[None]["offtaker_availabilities"] = dict(self.offtaker_availabilities.iloc[self.planning_horizon:].sum())
            data[None]["contract_deadlines"] = dict(self.contract_deadlines.iloc[self.planning_horizon:].sum())

        # Solve hourly LP model
        return self._run_hourly_model(obs, time, info, data)

    def _update_logbook(self):
        self.logbook["average_price_projection"].append(self.average_price_projection)
        self.logbook["average_cf_projection"].append(self.average_cf_projection)

    def get_schedules(self, time, end_time):
        """ Returns offtaker schedules and contract deadline schedules from time to time + horizon. """        
        hourly_index = pd.to_datetime(pd.date_range(start=time, end=end_time, freq='h'), utc=True)
        
        offtaker_availabilities = {}
        for name, offtaker in self.env.rfp.get_offtakers().items():
            availability_frequency = offtaker.parameters.get("availability")
            offtaker_availabilities[name] = self._get_availability(availability_frequency, hourly_index)
        df_offtakers = pd.DataFrame(offtaker_availabilities, index=hourly_index)

        contract_deadlines = {}
        for name, contract in self.env.rfp.get_contracts().items():
            target_frequency = contract.parameters.get("target_frequency")
            contract_deadlines[name] = self._get_availability(target_frequency, hourly_index)
        df_contracts = pd.DataFrame(contract_deadlines, index=hourly_index)

        return df_offtakers, df_contracts

    def pi(self, obs, k, info:dict):
        time = info["time"]
        is_backcasting = bool(info.get("forecast_path", False))

        if (is_backcasting and k % 7 == 0) ^ (time.day_of_week == 0 and not is_backcasting): # We do not expect big changes in strike price throughout the year - update two times a month.
            self.longterm_horizon, self.average_price_projection, self.average_cf_projection = self._estimate_longterm_uncertainties(obs, time, info)
        else:
            self.longterm_horizon = max(self.longterm_horizon - self.decision_horizon, 0)
        
        end_time = time + pd.Timedelta(self.planning_horizon + self.longterm_horizon - 1, 'h')
        self.offtaker_availabilities, self.contract_deadlines = self.get_schedules(time, end_time) # Get offtaker availabilities for the whole year, since we will use this as a guideline for the whole year.
        
        actions = self._solve_hourly_decisions(obs=obs, time=time, info=info) # Day-ahead solving
        self._update_logbook()

        return np.asarray(actions)


class RecedingHorizonAgent(HierarchicalAgent):
    def __init__(self,
                 env,
                 *args,
                 writer=None,
                 planning_horizon:int = 4*24,
                 guideline:str|None = None,
                 hourly_model_class=HourlyDeterministicLPModel,
                 solver='gurobi',
                 documentation=False,
                 objective_logic=None,
                 **kwargs,
                 ):
        super().__init__(env, *args,
                         writer=writer, planning_horizon=planning_horizon, guideline=guideline,
                         hourly_model_class=hourly_model_class, solver=solver,
                         documentation=documentation, objective_logic=objective_logic, **kwargs)
        self.price_projection = None
        self.cf_projection = None
        self.longterm_horizon = 0
        self.year_sim = None
    
    def _project_longterm_uncertainties(self, obs, k, time, info):
        is_backcasting = bool(info.get("forecast_path", False))
        if is_backcasting and k % 7 == 0:
            timestamp_str = time.strftime("%Y-%m-%d")
            self.year_sim = pd.read_csv(info["forecast_path"] + f"long-term-sim_{timestamp_str}.csv", index_col=0)
        elif self.env.load_data and time.day_of_week == 0:
            # if n_sims > 5:
            #     print("Only generated 5 simulations year ahead - setting number of sims to 5.")
            timestamp_str = time.strftime("%Y%m%d")
            # Get timestamp_str of the start of the week, since we only have estimates every week:
            # latest_projection_time = time - pd.Timedelta(time.day_of_week, unit="d")
            # updated_timestamp_str = latest_projection_time.strftime("%Y%m%d")
            # hours_extra = (int(timestamp_str) - int(updated_timestamp_str))*24 
            # year_simulations = [pd.read_csv(f"{self.env.scenario_path}year_sim_{updated_timestamp_str}_{ix}.csv").iloc[hours_extra:] for ix in range(n_sims)]
            self.year_sim = pd.read_csv(f"{self.env.scenario_path}year_sim_{timestamp_str}_0.csv")
        elif time.day_of_week == 0 or self.year_sim is None:
            self.year_sim = self.env.forecaster.simulate_year_ahead(start = time, n_sims=1)[0] # Creates a list of n_sims simulated year-ahead forecasts (pd.DataFrame with hourly index and 'price', 'wind', 'solar' columns)
        else:
            self.year_sim = self.year_sim.iloc[self.decision_horizon:]
        
        longterm_horizon = max(len(self.year_sim) - self.planning_horizon, 0)
        price_projection = []
        cf_projection = {key: [] for key in self.env.ppa_names}
        
        if longterm_horizon > 0:
            price_projection = self.year_sim["price"].iloc[-longterm_horizon:].values
            for name, ppa in self.env.rfp.get_ppas().items():
                cf = np.ones(len(self.year_sim)) # Assumes full availability of non-variable PPAs.
                if ppa.parameters.get("consumes") == 'wind':
                    cf = self.env.wind_mapper(self.year_sim['wind']).values
                elif ppa.parameters.get("consumes") == 'solar':
                    cf = self.env.solar_mapper(self.year_sim['solar']).values
                cf_projection[name] = cf[-longterm_horizon:]
        
        return longterm_horizon, price_projection, cf_projection

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
            cf_forecast = {(ppa_name, t): forecast_profile[t] for t in range(self.planning_horizon)}
            cf_longterm = {(ppa_name, t + self.planning_horizon): self.cf_projection[ppa_name][t] for t in range(self.longterm_horizon)}
            supplier_cf = {**supplier_cf, **cf_forecast, **cf_longterm}
        return supplier_cf

    def _solve_hourly_decisions(self, obs, time:pd.Timestamp, info:dict):
        full_horizon = self.planning_horizon + self.longterm_horizon

        self.hourly_model.planning_horizon = full_horizon
        self.hourly_model.initialize_model()

        data = self._construct_concrete_data(obs, time)
        time_index = pd.to_datetime(pd.date_range(time, time+pd.Timedelta(full_horizon-1,'h'),freq='h'), utc=True)
        datetime_data = {t: time_index[t] for t in range(full_horizon)}
        data[None]["T_datetime"] = datetime_data

        forecasts, electricity_price_forecast = self._get_forecasts_and_electricity(time, info)
        electricity_price_forecast.update({t + self.planning_horizon: self.price_projection[t] for t in range(self.longterm_horizon)})
        supplier_cf = self._get_supplier_cf(obs, forecasts[0])
        data[None]["supplier_cf"] = supplier_cf
        data[None]["electricity_price"] = electricity_price_forecast

        # Solve hourly LP model
        return self._run_hourly_model(obs, time, info, data)

    def pi(self, obs, k, info:dict):
        time = info["time"]
        # self.longterm_horizon = max(((self.env.episode_end - time).days + 1) * 24 - self.planning_horizon, 0)
        self.longterm_horizon, self.price_projection, self.cf_projection = self._project_longterm_uncertainties(obs, k, time, info)
        actions = self._solve_hourly_decisions(obs=obs, time=time, info=info) # Day-ahead solving
        self._update_logbook()

        return np.asarray(actions)

