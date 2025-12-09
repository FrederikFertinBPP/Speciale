
import gymnasium as gym
import numpy as np
import pandas as pd
from collections import deque
from common_scripts.utils import cache_read
from common_scripts.RFP_initialization import RenewableFuelPlant, create_rfp
from data_scripts.data_generator_v2 import DataForecaster
from sklearn.preprocessing import MinMaxScaler
from time import time as get_unix_time
import copy
from dateutil.relativedelta import relativedelta
import os
from model_scripts.hourly_models import ShieldLPModel, HourlyDeterministicLPModel, HourlyRecourseModel, ShieldRecourseModel

def get_env(env_class:gym.Env, allow_spot_buy=True, balancing_market=False, verbose=False, load_data=False):
    rfp = create_rfp()

    forecaster = DataForecaster(from_pickle=True, cache_id="Anders")
    forecaster = forecaster.unpickle()
    forecaster.t_init = forecaster.test_data.index[0]

    ### Scenario specification: Could be predefined in excel file.
    rfp.get_contract('Ammonia1').parameters['volume'] = rfp.get_component("Haber Bosch Plant").parameters.get('capacity') * 8760 / (2 if allow_spot_buy else 5)  # 50% capacity contracted

    env = env_class(rfp=rfp, forecaster=forecaster, decision_horizon=24,
                    allow_spot_buy = allow_spot_buy, verbose=verbose,
                    balancing_market=balancing_market, load_data=load_data,
                    )

    return env


class VRESystemToAssetMapping:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, *args, **kwds):
        return np.clip(self.model(*args, **kwds), 0, 1)


class RFPShieldEnv(gym.Env):
    """ Environment, which allows for operating a Renewable Fuel Plant in a rolling horizon fashion. """
    
    def __init__(self,
                 rfp:RenewableFuelPlant,
                 forecaster:DataForecaster = None,
                 decision_horizon:int = 24, # Unit: hours
                 allow_spot_buy = True,
                 normalize:bool = False,
                 verbose:bool = False,
                 balancing_market:bool = False,
                 load_data = True,
                 **kwargs,
                 ):
        """
        Initialize the RFP environment.
        This environment simulates a hypothetical path planning problem.
        """
        self.rfp = rfp
        self.original_forecaster = forecaster
        self.decision_horizon = decision_horizon # Decision horizon in hours
        self.allow_spot_buy = allow_spot_buy
        # Whether to normalize the state and action spaces (dependent on the algorithm used for decision-making)
        # Should maybe be set by the agent instead?
        self.normalize_step = normalize
        self.balancing_market = balancing_market
        self.load_data = load_data
        self.scenario_name = "" if (forecaster is None) else forecaster.cache_id
        self.scenario_number = -1
        self.verbose = verbose

        self.time = None # Placeholder for the current time in the environment
        self.state = None # Placeholder for the state of the environment
        self.context = None # Placeholder for the context of the environment (stochastic forecasts)
        self.forecaster = None # Placeholder for the currently used initialization of forecaster.

        """ Retrieve mappers, which produce VRE profiles for single assets, given a system level production profile. """
        cache_path_mappers = os.getcwd() + "/models/plant_models/"
        solar_mapper = cache_read(cache_path_mappers + "solar.pkl")
        self.solar_mapper = VRESystemToAssetMapping(solar_mapper)
        wind_mapper = cache_read(cache_path_mappers + "wind.pkl")
        self.wind_mapper = VRESystemToAssetMapping(wind_mapper)

        # The action space is defined as a Box with lower and upper bounds based on the capacities of the components
        self.action_identity = ['dayahead-ElectricitySpot-out_flow',
                                'link-Electrolyzer-out_flow',
                                "link-Haber Bosch Plant-out_flow",
                                "contract-Hydrogen1-shipment",
                                "contract-Ammonia1-shipment",
                                "contract-AmmoniaSpot-shipment",
                                ]
        if self.balancing_market:
            self.action_identity[0] = 'dayahead-ElectricitySpot-da_buy'
            self.action_identity += ['dayahead-ElectricitySpot-ba_buy', 'dayahead-ElectricitySpot-ba_sell']
        
        self._set_observation_space()

        shield_class = ShieldRecourseModel if self.balancing_market else ShieldLPModel
        self.shield = shield_class(self,
                                    rfp,
                                    decision_horizon=decision_horizon,
                                    solver='gurobi',
                                    allow_spot_buy=allow_spot_buy,
                                    )
        self.shield.initialize_model()
        
        """ Define the action space """
        # The action space consists of four actions:
        # 1. Amount of electricity to buy from the grid (ElectricitySpot)
        self.spot_market_cap    = self.rfp.get_component("ElectricitySpot").parameters.get('capacity') # [MW]
        # 2. Amount of hydrogen to produce (Electrolyzer)
        self.electrolyzer_cap   = self.rfp.get_component("Electrolyzer").parameters.get('capacity') # [tH2/h]
        # 3. Amount of ammonia to produce (Haber Bosch Plant)
        self.hb_cap             = self.rfp.get_component("Haber Bosch Plant").parameters.get('capacity') # [tNH3/h]
        # 4. Amount of hydrogen to ship to hourly offtaker (Hydrogen1)
        self.pipeline_cap       = self.rfp.get_component("Hydrogen Pipeline").parameters.get('capacity') # [tH2/h]
        # 5. Amount of ammonia to ship to yearly contract (Ammonia1)
        # 6. Amount of ammonia to ship to spot contract (AmmoniaSpot)
        self.nh3_ship_cap       = self.rfp.get_component("Ammonia Shipment").parameters.get('capacity') # [tNH3/h]
        lows = [-self.spot_market_cap, 0, 0, 0, 0, 0]
        highs = [self.spot_market_cap * self.allow_spot_buy,
                self.electrolyzer_cap,
                self.hb_cap,
                self.pipeline_cap,
                self.nh3_ship_cap,
                self.nh3_ship_cap,
                ]
        if self.balancing_market:
            lows += [0, 0]
            highs += [self.spot_market_cap * self.allow_spot_buy, self.spot_market_cap]
        self.action_space_low   = np.asarray(lows, dtype = np.float64)
        self.action_space_high  = np.asarray(highs, dtype = np.float64)
        self.action_space = gym.spaces.Box(
                            low  = np.asarray([self.action_space_low]  * self.decision_horizon, dtype = np.float64),
                            high = np.asarray([self.action_space_high] * self.decision_horizon, dtype = np.float64),
                            shape = (self.decision_horizon, len(self.action_identity)), dtype = np.float64)
        
        # Define scaler for normalizing the action space
        self.action_scaler = MinMaxScaler(feature_range=(0, 1))
        self.action_scaler.fit(np.vstack([self.action_space.low, self.action_space.high]))

        # Save metadata to environment because the RFP may change between experiments.
        forecaster_id = str(forecaster.cache_id) if forecaster is not None and forecaster.cache_id is not None else "Unknown"
        self.metadata = {**self.metadata,
                         **self.rfp.to_dict(),
                         **self.observation_space.__dict__,
                         **self.action_space.__dict__,
                         "action_identity": self.action_identity,
                         "normalized": self.normalize_step,
                         "decision_horizon": self.decision_horizon,
                         "forecaster": forecaster_id,
                         "balancing_market": balancing_market,
                         "allow_spot_buy": self.allow_spot_buy,
                        }

    def _set_state_space(self):
        """ Define the state space for RFP storages and contracts. """
        # --- Storages ---
        self.storage_names, s_low, s_high = [], [], []
        for name, storage in self.rfp.get_storages().items():
            self.storage_names.append(name)
            s_high.append(storage.parameters.get("capacity"))
            self.metadata = {**self.metadata, **{str(str(name) + " capacity"): s_high[-1]}}
        self.storage_state_space = gym.spaces.Box(low=0, high = np.asarray(s_high), dtype = np.float64)

        # --- Contracts ---
        self.contract_names, s_high = [], []
        for name, contract in self.rfp.get_contracts().items():
            is_spot_contract = bool(contract.parameters.get("spot_contract", 0))
            if is_spot_contract == False:
                self.contract_names.append(name)
                s_high.append(contract.parameters.get('volume'))
                self.metadata = {**self.metadata, **{str(str(name) + " volume"): s_high[-1]}}
        # State space for actual shipped status:
        self.contract_state_space = gym.spaces.Box(low=0, high = np.asarray(s_high), dtype = np.float64)
        self.state_space = gym.spaces.Dict({"storages": self.storage_state_space, "contracts": self.contract_state_space,})

    def _set_context_space(self):
        """ Define the context of the observation space. """
        # --- Time information ---
        self.time_context_space = gym.spaces.Box(low=0, high=1, shape=(self.decision_horizon, len(self.rfp.frequency_options)))

        # --- Power-Purchase-Agreements ---
        self.ppa_names, c_high = [], []
        for name, ppa in self.rfp.get_ppas().items():
            self.ppa_names.append(name)
            c_high.append(ppa.parameters.get("capacity"))
            self.metadata = {**self.metadata, **{str(name.replace(" ", "_").lower() + "_capacity"): c_high[-1]}}
        self.ppa_context_space = gym.spaces.Box(low=0, high = np.asarray([c_high] * self.decision_horizon), dtype = np.float64)
        
        self.price_context_space = gym.spaces.Box(low=-500, high = np.asarray([4000] * self.decision_horizon), dtype = np.float64)

        self.offtaker_names = []
        for name, offtaker in self.rfp.get_offtakers().items():
            self.offtaker_names.append(name)
        self.offtaker_context_space = gym.spaces.MultiBinary(n=(self.decision_horizon*14, len(self.offtaker_names)))

        self.context_space = gym.spaces.Dict({"time": self.time_context_space,
                                              "ppas": self.ppa_context_space,
                                              "offtakers": self.offtaker_context_space,
                                              "prices": self.price_context_space,
                                              })

    def _set_observation_space(self):
        """ We define the observation space as the union of the space and context. """
        # The observation space consists of a state and a context.
        # State is a relevant term for the status of storages and contracts. (Could also be more detailed information about operational states of links).
        # Context is relevant for information about PPAs, the spot market conditions, and offtaker availabilities.
        self._set_state_space()
        self._set_context_space()
        # Use :class:`gymnasium.wrappers.FlattenObservation` wrapper to work with this later.
        self.observation_space = gym.spaces.Dict({"state": self.state_space, "context": self.context_space})

    def _set_time_context(self):
        """ How far are we in the frequency cycle? """
        T = pd.date_range(self.time, self.time+pd.Timedelta(self.decision_horizon-1, 'h'), freq='h')
        self.realized_time += list(T)

        C = []
        for t in T:
            c = []
            for freq in self.rfp.frequency_options:
                if freq=='hourly':
                    c.append(t.minute/60) # If we want to go to sub-hourly operations.
                if freq=='daily':
                    c.append(t.hour/24)
                if freq=='monthly':
                    c.append(t.day/t.days_in_month)
                if freq=='yearly':
                    days_in_year = 366 if t.is_leap_year else 365
                    c.append(t.day_of_year/days_in_year)
            C.append(c)
        self.time_context = np.asarray(C)

    def _set_ppa_context(self):
        # Realize VRE PPA availability for the next decision horizon (typically 24 hours)
        if self.load_data:
            timestamp_str = self.time.strftime("%Y%m%d")
            system_solar_realization = pd.read_csv(f"{self.scenario_path}solar_{timestamp_str}.csv")
            system_wind_realization = pd.read_csv(f"{self.scenario_path}wind_{timestamp_str}.csv")
        else:
            system_solar_realization, system_wind_realization = self.forecaster.realize_vre(start=self.time, end=self.time + pd.Timedelta(self.decision_horizon-1, 'h')) # DF
        system_solar_realization, system_wind_realization = system_solar_realization['solar'].values, system_wind_realization['wind'].values
        profiles = []
        for name, ppa in self.rfp.get_ppas().items():
            if ppa.parameters.get("consumes") == 'wind':
                cf = self.wind_mapper(system_wind_realization)
            elif ppa.parameters.get("consumes") == 'solar':
                cf = self.solar_mapper(system_solar_realization)
            else:
                cf = np.ones(len(system_solar_realization)) # Assumes full availability of non-variable PPAs.
            profiles.append(cf)
        self.ppa_context = np.transpose(np.asarray(profiles))

    def _set_offtaker_context(self):
        # Binary context space for offtaker availability:
        availabilities = []
        offtaker_schedule_horizon = self.decision_horizon*14
        time_stamps = pd.to_datetime(pd.date_range(start=self.time, end=self.time+pd.Timedelta(offtaker_schedule_horizon-1, 'h'), freq='h'), utc=True)
        for name, offtaker in self.rfp.get_offtakers().items():
            availability_frequency = offtaker.parameters.get("availability")
            a = np.zeros(offtaker_schedule_horizon)
            for t in range(offtaker_schedule_horizon):
                time = time_stamps[t]
                if availability_frequency=='hourly':
                    a[t] = 1
                if availability_frequency=='daily':
                    a[t] = int(time.hour == 23)
                if availability_frequency=='monthly':
                    a[t] = int(time.is_month_end and time.hour == 23)
                if availability_frequency=='yearly':
                    a[t] = int(time.is_year_end and time.hour == 23)
            availabilities.append(a)
        
        self.offtaker_context = np.transpose(np.asarray(availabilities))

    def _set_price_context(self):
        # Realize prices - updates the forecaster object by including the new realizations.
        if self.load_data:
            timestamp_str = self.time.strftime("%Y%m%d")
            price_forecast = pd.read_csv(f"{self.scenario_path}forecast_{timestamp_str}_0.csv", usecols=["price"], nrows=24)['price'].values
            price_realization = pd.read_csv(f"{self.scenario_path}prices_{timestamp_str}.csv")
        else:
            price_forecast = self.forecaster.forecast(start=self.time, end=self.time+pd.Timedelta(self.decision_horizon-1, 'h'), n_forecasts=1)[0]['price'].values
            price_realization = self.forecaster.realize_prices(start=self.time, end=self.time+pd.Timedelta(self.decision_horizon-1, 'h'))
        real_prices = price_realization['price'].values
        self.realized_prices += list(real_prices)
        
        # Naive forecasts:
        # df_prices = pd.DataFrame(index=self.realized_time, data={"price": self.realized_prices})
        # def _get_shifted_ts(ts, lag):
        #     shifted = ts.shift(lag)
        #     shifted.loc[shifted.index[:lag],'price'] = ts.loc[ts.index[-lag:],'price'].values
        #     return shifted
        # lags = [24, 48, 72]
        # df = [_get_shifted_ts(df_prices, lag) for lag in lags]
        # ma = df[0].rolling(window=168).mean()
        # ma.loc[ma.index[:168],'price'] = df[0].loc[df[0].index[:168],'price'].values
        # df += [ma]
        # price_forecasts = np.asarray(df)
        # The price context then includes "price forecasts" - lag of 24, 48, 72, and weekly moving average.
        self.price_context = price_forecast

    def _set_context(self):
        self._set_time_context()
        self._set_ppa_context()
        self._set_offtaker_context()
        self._set_price_context()

    def _get_obs(self):
        """ Convert internal state to observation format.
        Returns:
            dict: Observation of state and context
        """
        self.state   = {"storages": self.storage_state, "contracts": self.contract_state}
        self.context = {"time": self.time_context, "ppas": self.ppa_context, "offtakers": self.offtaker_context, "prices": self.price_context}
        return {"state": self.state, "context": self.context}

    def reset(self, *, seed: int | None = None, options = None):
        """
        Reset the environment to its initial state.
        Returns:
            tuple: The initial state of the environment, and additional info.
        """
        # Reset core and stochastic components of environment:
        super().reset(seed=seed)
        self.forecaster = copy.deepcopy(self.original_forecaster) # self.forecaster.set_seed(np.random.randint(low=0,high=2**30))
        self.time = self.forecaster.t_init
        self.episode_end = self.forecaster.t_init + relativedelta(years=+1) - pd.Timedelta(1, 'hour') # Episodic implementation
        self.scenario_number += 1
        self.scenario_path = f"scenario_data/{self.scenario_name}_scenario_{self.scenario_number}/"
        self.realized_prices = []
        self.realized_time = []

        # --- Reset state ---
        self.storage_state = self.storage_state_space.low
        self.contract_state = self.contract_state_space.low

        # --- Reset context ---
        self._set_context()

        obs = self._get_obs()
        info = {"time": self.time,}
        self.episode_unix_start = get_unix_time()

        return obs, info

    def activate_shield(self, action):
        """ Solve flows for the next 24 hours of operation with fixed decisions.
            We are realizing the plant operation and changing setpoints only when current solution is infeasible.
        """
        # Set up dictionaries:
        supplier_cf = {}
        for ix, (name, ppa) in enumerate(self.rfp.get_ppas().items()):
            cf = {(name, t): self.ppa_context[t,ix] for t in range(self.decision_horizon)}
            supplier_cf = {**supplier_cf, **cf}
        time_index = pd.to_datetime(pd.date_range(start=self.time, end=self.time+pd.Timedelta(self.decision_horizon-1, 'h'), freq='h'), utc=True)
        datetime_data = {t: time_index[t] for t in range(self.decision_horizon)}

        # Chosen actions
        chosen_actions = {(name, t): action[t, ix] for ix, name in enumerate(self.action_identity) for t in range(self.decision_horizon)}

        data = { # Set up the necessary data for the LP Concrete Model
            None: {
                'T_datetime': datetime_data,
                'init_soc': dict(zip(self.storage_names, self.storage_state)),
                'supplier_cf': supplier_cf,
                'init_contract_status' : dict(zip(self.contract_names, self.contract_state)),
                'chosen_actions': chosen_actions,
            }
        }

        self.shield.build_concrete_instance(data=data)
        self.shield.run(verbose=False)
        return self.shield.get_actions()

    def _get_reward_and_info(self, prices, shield_penalty, truncated):
        """ Calculate the reward based on the actions taken. """
        spot_bought = self.shield.decision_results.spot_power # np.ndarray
        spot_electricity_cost = np.sum(spot_bought * prices) # Float
        contract_revenues = np.sum(list(self.shield.decision_results.delivered_revenue.values())) # Float
        contract_penalties = np.sum(list(self.shield.decision_results.contract_penalty.values())) # Float

        # Penalty for truncation violations equal to number of hours left in the year
        truncation_penalty = truncated * (self.episode_end-self.time).total_seconds()/3600 # Float

        """ Summarize environment state at the end of the step in the info dict. """
        info = {}

        # Electricity revenues
        info["el_spot_buy"]             = np.sum(spot_bought * (spot_bought > 0)) # Float
        info["el_spot_sell"]            = -np.sum(spot_bought * (spot_bought < 0)) # Float
        info["el_spot_revenue"]         = -np.sum(spot_bought * prices * (spot_bought < 0)) # Float
        info["el_spot_cost"]            = np.sum(spot_bought * prices * (spot_bought > 0)) # Float
        info["el_spot_balance"]         = info["el_spot_revenue"] - info["el_spot_cost"] # Float
        if self.balancing_market:
            info["da_cost"]             = np.sum(self.shield.decision_results.dayahead_buy * prices)
            info["ba_buy"]              = np.asarray(self.shield.decision_results.balancing_buy)
            info["ba_sell"]             = np.asarray(self.shield.decision_results.balancing_sell)

            info["ba_cost"]             = np.sum((np.asarray(self.shield.decision_results.balancing_buy) * 1.3 -
                                                  np.asarray(self.shield.decision_results.balancing_sell) * 0.7) * prices)
            info["el_revenue"]          = -(info["da_cost"] + info["ba_cost"])
            spot_electricity_cost       = -info["el_revenue"]

        # Fuel sale summaries
        info["contract_revenues"]       = self.shield.decision_results.delivered_revenue # Dict
        info["link_productions"]        = self.shield.decision_results.link_production # Dict
        info["shipments"]               = self.shield.decision_results.shipments # Dict

        # Penalties
        info["contract_penalties"]      = self.shield.decision_results.contract_penalty # Dict
        info["truncation_penalty"]      = truncation_penalty # Float
        info["shield_penalty"]          = shield_penalty # Float

        # Electricity/power flows
        info["ppa_power"]               = self.shield.decision_results.ppa_power # Dict
        info["ppa_cost"]                = self.shield.decision_results.ppa_costs # Float
        info["electricity_price"]       = prices # np.ndarray

        # Hourly SOC
        info["storage_soc"]             = self.shield.decision_results.storage_soc # Dict
        info["contract_status"]         = self.shield.decision_results.contract_status # Dict

        # Monetary summaries
        info["real_cash_flow"]          = contract_revenues - spot_electricity_cost - contract_penalties # Float

        # The reward can be custom-defined based on how we want to train the agent.
        reward = contract_revenues - spot_electricity_cost - contract_penalties - shield_penalty - truncation_penalty # Float

        if truncated: # Inform about infeasibility causing truncation
            info["technical_violation_message"] = "Could not handle the flows determined by the agent."

        return reward, info

    def _step(self, action):
        """
        Perform a step in the environment with the given action.
        Args:
            action np.array: The actions to take. shape:(decision_horizon, n_actions)
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """

        # Compute shielded actions and get correction penalty (0 if not needed).
        shielded_action = self.activate_shield(action) # Also evaluates the environment dynamics.
        shield_penalty, truncated = self.shield.get_objective()

        # --- Update state ---
        self.storage_state = np.asarray(list(self.shield.decision_results.final_soc.values()))
        self.contract_state = np.asarray(list(self.shield.decision_results.final_contract_status.values()))

        # Realize prices - already done when context is set.
        prices = np.asarray(self.realized_prices[-self.decision_horizon:])

        reward, info = self._get_reward_and_info(prices, shield_penalty, truncated)
        
        # We also save the current state in the info for possible flexible access.
        # --- Update context ---
        self.time += pd.Timedelta(self.decision_horizon, 'h')
        terminated = self.time >= self.episode_end # Terminate episode after one year of operations
        info["time"] = self.time
        # if not(terminated):
        self._set_context()

        obs = self._get_obs()
        
        if truncated or terminated:
            info["episode_runtime"] = get_unix_time() - self.episode_unix_start
        
        return obs, reward, terminated, truncated, info

    def _normalized_step(self, normalized_action):
        # Denormalize the action to the original action space
        action = self.action_scaler.inverse_transform(normalized_action)
        return self._step(action)

    def step(self, action):
        """
        Perform a step in the environment with the given normalized action.
        Args:
            action (np.array): The action in the range [0, 1].
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """
        if self.normalize_step:
            return self._normalized_step(action)
        else:
            return self._step(action)


class RFPYearEnv(RFPShieldEnv):
    def __init__(self, rfp, forecaster = None, allow_spot_buy=True, normalize = False, verbose = False, load_data=True, **kwargs):
        self.original_forecaster = forecaster
        t = self.original_forecaster.t_init
        t_end = self.original_forecaster.t_init + relativedelta(years=+1) # Episodic implementation
        horizon = (t_end - t).days * 24 # Number of hours in the year.
        super().__init__(rfp, forecaster, decision_horizon=horizon, allow_spot_buy=allow_spot_buy, normalize=normalize, verbose=verbose, load_data=load_data)
        self.pfm = HourlyDeterministicLPModel(rfp, decision_horizon=horizon, solver='gurobi', allow_spot_buy=allow_spot_buy)
        self.pfm.initialize_model()

    def _set_ppa_context(self):
        # Realize VRE PPA availability for the next decision horizon (typically 24 hours)
        if self.load_data:
            system_solar_realization, system_wind_realization = [],[]
            for day in range((self.episode_end - self.time).days+1):
                timestamp_str = (self.time+pd.Timedelta(day,'days')).strftime("%Y%m%d")
                system_solar_realization += list(pd.read_csv(f"{self.scenario_path}solar_{timestamp_str}.csv")['solar'].values)
                system_wind_realization += list(pd.read_csv(f"{self.scenario_path}wind_{timestamp_str}.csv")['wind'].values)
            system_solar_realization = np.asarray(system_solar_realization)
            system_wind_realization = np.asarray(system_wind_realization)
        else:
            system_solar_realization, system_wind_realization = self.forecaster.realize_vre(start=self.time, end=self.time + pd.Timedelta(self.decision_horizon-1, 'h')) # DF
            system_solar_realization, system_wind_realization = system_solar_realization['solar'].values, system_wind_realization['wind'].values
        profiles = []
        for name, ppa in self.rfp.get_ppas().items():
            if ppa.parameters.get("consumes") == 'wind':
                cf = self.wind_mapper(system_wind_realization)
            elif ppa.parameters.get("consumes") == 'solar':
                cf = self.solar_mapper(system_solar_realization)
            else:
                cf = np.ones(len(system_solar_realization)) # Assumes full availability of non-variable PPAs.
            profiles.append(cf)
        self.ppa_context = np.transpose(np.asarray(profiles))

    def _set_price_context(self):
        # Realize prices - updates the forecaster object by including the new realizations.
        if self.load_data:
            price_forecast, price_realization = [], []
            for day in range((self.episode_end - self.time).days+1):
                timestamp_str = (self.time+pd.Timedelta(day,'days')).strftime("%Y%m%d")
                price_forecast += list(pd.read_csv(f"{self.scenario_path}forecast_{timestamp_str}_0.csv", usecols=["price"], nrows=24)['price'].values)
                price_realization += list(pd.read_csv(f"{self.scenario_path}prices_{timestamp_str}.csv")['price'].values)
            price_forecast = np.asarray(price_forecast)
        else:
            price_forecast = self.forecaster.forecast(start=self.time, end=self.time+pd.Timedelta(self.decision_horizon-1, 'h'), n_forecasts=1)[0]['price'].values
            price_realization = list(self.forecaster.realize_prices(start=self.time, end=self.time+pd.Timedelta(self.decision_horizon-1, 'h'))['price'].values)
        self.realized_prices += price_realization
        self.price_context = price_forecast

    def _get_reward_and_info(self, prices, shield_penalty, truncated):
        reward, info = super()._get_reward_and_info(prices, shield_penalty, truncated)

        time_index = pd.to_datetime(pd.date_range(self.time, self.time+pd.Timedelta(self.decision_horizon-1, 'h'), freq='h'), utc=True)#prices.index
        if self.load_data:
            electricity_price = prices
        else:
            electricity_price = prices.values

        wind_profile = info['ppa_power']['WindPower'] / self.rfp.get_ppa("WindPower").parameters.get("capacity")
        solar_profile = info['ppa_power']['SolarPower'] / self.rfp.get_ppa("SolarPower").parameters.get("capacity")
        
        wind_cf = {('WindPower', t): wind_profile[t] for t in range(self.decision_horizon)}
        solar_cf = {('SolarPower', t): solar_profile[t] for t in range(self.decision_horizon)}
        nuclear_cf = {('NuclearPower', t): 1.0 for t in range(self.decision_horizon)}
        supplier_cf = {**wind_cf, **solar_cf, **nuclear_cf,}
        electricity_price = {t: electricity_price[t] for t in range(self.decision_horizon)}
        datetime_data = {t: time_index[t] for t in range(self.decision_horizon)}
        data = {
            None: {
                'T_datetime' : datetime_data,
                'supplier_cf': supplier_cf,
                'electricity_price': electricity_price,
            }
        }
        self.pfm.build_concrete_instance(data=data)
        self.pfm.run(verbose=True)
        opt_objective, _ = self.pfm.get_objective()
        info["optimal_profit"] = opt_objective
        info["realized_profit"] = info["real_cash_flow"]
        reward = info["realized_profit"] / info["optimal_profit"] # Reward is a standardized measure, not including feasibility penalties or ppa costs.
        
        return reward, info


class RFPEnv(RFPShieldEnv):
    step_with_hourly_model = True
    
    def step(self, action, hourly_model):
        """
        Perform a step in the environment with the given normalized action.
        Args:
            action (np.array): The action in the range [0, 1].
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """
        if self.normalize_step:
            return self._normalized_step(action)
        else:
            return self._step(action, hourly_model)

    def _step(self, action, hourly_model):
        """
        Perform a step in the environment with the given action.
        Args:
            action np.array: The actions to take. shape:(decision_horizon, n_actions)
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """
        # Compute shielded actions and get correction penalty (0 if not needed).
        
        objective, truncated = hourly_model.get_objective()

        # --- Update state ---
        self.storage_state = np.asarray(list(hourly_model.decision_results.final_soc.values()))
        self.contract_state = np.asarray(list(hourly_model.decision_results.final_contract_status.values()))

        # Realize prices - already done when context is set.
        prices = np.asarray(self.realized_prices[-self.decision_horizon:])

        reward, info = self._get_reward_and_info(prices, hourly_model, truncated)
        
        # We also save the current state in the info for possible flexible access.
        # --- Update context ---
        self.time += pd.Timedelta(self.decision_horizon, 'h')
        terminated = self.time >= self.episode_end # Terminate episode after one year of operations
        info["time"] = self.time
        # if not(terminated):
        self._set_context()

        obs = self._get_obs()

        if truncated or terminated:
            info["episode_runtime"] = get_unix_time() - self.episode_unix_start

        return obs, reward, terminated, truncated, info

    def _get_reward_and_info(self, prices, hourly_model, truncated):
        """ Calculate the reward based on the actions taken. """
        horizon = hourly_model.decision_horizon
        spot_bought = hourly_model.decision_results.spot_power # np.ndarray
        spot_electricity_cost = np.sum(spot_bought * prices) # Float
        contract_revenues = np.sum(list(hourly_model.decision_results.delivered_revenue.values())) # Float
        contract_penalties = np.sum(list(hourly_model.decision_results.contract_penalty.values())) # Float
        
        # Penalty for truncation violations equal to number of hours left in the year
        truncation_penalty = truncated * (self.episode_end-self.time).total_seconds()/3600 # Float

        """ Summarize environment state at the end of the step in the info dict. """
        info = {}

        # Electricity revenues
        info["el_spot_bought"] = np.sum(spot_bought * (spot_bought > 0))
        info["el_spot_sold"] = -np.sum(spot_bought * (spot_bought < 0))
        info["el_spot_revenue"]             = -np.sum(spot_bought * prices * (spot_bought < 0)) # Float
        info["el_spot_cost"]                = np.sum(spot_bought * prices * (spot_bought > 0)) # Float
        info["el_spot_balance"]             = info["el_spot_revenue"] - info["el_spot_cost"] # Float
        if self.balancing_market:
            dayahead_buy = np.asarray(hourly_model.decision_results.dayahead_buy)
            info["dayahead_bought"]        = np.sum(dayahead_buy * (dayahead_buy > 0))
            info["dayahead_sold"]          = -np.sum(dayahead_buy * (dayahead_buy < 0))
            info["dayahead_cost"]          = np.sum(dayahead_buy * prices)
            info["balancing_power_bought"] = np.sum(hourly_model.decision_results.balancing_buy)
            info["balancing_power_sold"]   = np.sum(hourly_model.decision_results.balancing_sell)
            info["balancing_buy_cost"]     = sum(hourly_model.decision_results.balancing_buy[t] * prices[t] * (1.3 if prices[t] > 0 else 0.7) for t in range(horizon))
            info["balancing_sell_revenue"] = sum(hourly_model.decision_results.balancing_sell[t] * prices[t] * (1.3 if prices[t] < 0 else 0.7) for t in range(horizon))
            info["balancing_cost"]         = info["balancing_buy_cost"] - info["balancing_sell_revenue"]
            info["el_revenue"]          = -(info["dayahead_cost"] + info["balancing_cost"])
            spot_electricity_cost       = -info["el_revenue"]

        # Fuel sale summaries
        info["contract_revenues"]           = hourly_model.decision_results.delivered_revenue # Dict
        info["link_productions"]            = hourly_model.decision_results.link_production # Dict
        info["shipments"]                   = hourly_model.decision_results.shipments # Dict

        # Penalties
        info["contract_penalties"]          = hourly_model.decision_results.contract_penalty # Dict
        info["truncation_penalty"]          = truncation_penalty # Float

        # Electricity/power flows
        info["ppa_power"]                   = hourly_model.decision_results.ppa_power # Dict
        info["ppa_cost"]                    = hourly_model.decision_results.ppa_costs # Float
        info["electricity_price"]           = prices # np.ndarray

        # Hourly SOC
        info["storage_soc"]                 = hourly_model.decision_results.storage_soc # Dict
        info["contract_status"]             = hourly_model.decision_results.contract_status # Dict
        
        # Monetary summaries
        info["real_cash_flow"]              = contract_revenues - spot_electricity_cost - contract_penalties # Float

        # The reward can be custom-defined based on how we want to train the agent.
        reward = contract_revenues - spot_electricity_cost - contract_penalties - truncation_penalty # Float

        if truncated: # Inform about infeasibility causing truncation
            info["technical_violation_message"] = "Could not handle the flows determined by the agent."
        
        return reward, info


class RFPRecourseEnv(RFPEnv):
    
    def _set_context_space(self):
        """ Define the context of the observation space. """
        # --- Time information ---
        self.time_context_space = gym.spaces.Box(low=0, high=1, shape=(self.decision_horizon+12, len(self.rfp.frequency_options)))

        # --- Power-Purchase-Agreements ---
        self.ppa_names, c_high = [], []
        for name, ppa in self.rfp.get_ppas().items():
            self.ppa_names.append(name)
            c_high.append(ppa.parameters.get("capacity"))
            self.metadata = {**self.metadata, **{str(name.replace(" ", "_").lower() + "_capacity"): c_high[-1]}}
        self.ppa_context_space = gym.spaces.Box(low=0, high = np.asarray([c_high] * (12+self.decision_horizon)), dtype = np.float64)
        
        # The price context contains forecast prices
        self.price_context_space = gym.spaces.Box(low=-500, high = np.asarray([4000] * (self.decision_horizon)), dtype = np.float64)

        self.offtaker_names = []
        for name, offtaker in self.rfp.get_offtakers().items():
            self.offtaker_names.append(name)
        self.offtaker_context_space = gym.spaces.MultiBinary(n=(self.decision_horizon*14, len(self.offtaker_names)))

        self.context_space = gym.spaces.Dict({"time": self.time_context_space,
                                              "ppas": self.ppa_context_space,
                                              "offtakers": self.offtaker_context_space,
                                              "prices": self.price_context_space,
                                              })

    def _set_context(self, reset=False):
        self._set_time_context()
        self._set_ppa_context()
        self._set_offtaker_context()
        self._set_price_context()

    def _set_time_context(self):
        """ How far are we in the frequency cycle? """
        T = pd.date_range(self.time-pd.Timedelta(12, 'h'), self.time+pd.Timedelta(self.decision_horizon-1, 'h'), freq='h')

        C = []
        for t in T:
            c = []
            for freq in self.rfp.frequency_options:
                if freq=='hourly':
                    c.append(t.minute/60) # If we want to go to sub-hourly operations.
                if freq=='daily':
                    c.append(t.hour/24)
                if freq=='monthly':
                    c.append(t.day/t.days_in_month)
                if freq=='yearly':
                    days_in_year = 366 if t.is_leap_year else 365
                    c.append(t.day_of_year/days_in_year)
            C.append(c)
        self.time_context = np.asarray(C)

    def _set_ppa_context(self):
        # Realize VRE PPA availability for the next decision horizon (typically 24 hours)
        if self.load_data:
            timestamp_str = self.time.strftime("%Y%m%d")
            system_solar_realization = pd.read_csv(f"{self.scenario_path}solar_{timestamp_str}.csv")
            system_wind_realization = pd.read_csv(f"{self.scenario_path}wind_{timestamp_str}.csv")
        else:
            system_solar_realization, system_wind_realization = self.forecaster.realize_vre(start=self.time, end=self.time + pd.Timedelta(self.decision_horizon-1, 'h')) # DF
        system_solar_realization, system_wind_realization = system_solar_realization['solar'].values, system_wind_realization['wind'].values

        for ix, (name, ppa) in enumerate(self.rfp.get_ppas().items()):
            if ppa.parameters.get("consumes") == 'wind':
                cf = self.wind_mapper(system_wind_realization)
            elif ppa.parameters.get("consumes") == 'solar':
                cf = self.solar_mapper(system_solar_realization)
            else:
                cf = np.ones(len(system_solar_realization)) # Assumes full availability of non-variable PPAs.
            self.realized_ppa[ix].extend(cf)
        self.ppa_context = np.transpose(np.asarray(self.realized_ppa))

    def _set_offtaker_context(self):
        # Binary context space for offtaker availability:
        availabilities = []
        offtaker_schedule_horizon = self.decision_horizon*14
        time_stamps = pd.to_datetime(pd.date_range(start=self.time, end=self.time+pd.Timedelta(offtaker_schedule_horizon-1, 'h'), freq='h'), utc=True)
        for name, offtaker in self.rfp.get_offtakers().items():
            availability_frequency = offtaker.parameters.get("availability")
            a = np.zeros(offtaker_schedule_horizon)
            for t in range(offtaker_schedule_horizon):
                time = time_stamps[t]
                if availability_frequency=='hourly':
                    a[t] = 1
                if availability_frequency=='daily':
                    a[t] = int(time.hour == 23)
                if availability_frequency=='monthly':
                    a[t] = int(time.is_month_end and time.hour == 23)
                if availability_frequency=='yearly':
                    a[t] = int(time.is_year_end and time.hour == 23)
            availabilities.append(a)
        
        self.offtaker_context = np.transpose(np.asarray(availabilities))

    def _set_price_context(self):
        # Realize prices - updates the forecaster object by including the new realizations.
        if self.load_data:
            timestamp_str = self.time.strftime("%Y%m%d")
            price_forecast = pd.read_csv(f"{self.scenario_path}forecast_{timestamp_str}_0.csv", usecols=["price"], nrows=24)['price'].values
            price_realization = pd.read_csv(f"{self.scenario_path}prices_{timestamp_str}.csv")
        else:
            price_forecast = self.forecaster.forecast(start=self.time, end=self.time+pd.Timedelta(self.decision_horizon-1, 'h'), n_forecasts=1)[0]['price'].values
            price_realization = self.forecaster.realize_prices(start=self.time, end=self.time+pd.Timedelta(self.decision_horizon-1, 'h'))
        real_prices = price_realization['price'].values
        self.realized_prices.extend(real_prices)
        self.forecast_prices.extend(price_forecast)
        self.price_context = price_forecast

    def reset(self, *, seed: int | None = None, options = None):
        """
        Reset the environment to its initial state.
        Returns:
            tuple: The initial state of the environment, and additional info.
        """
        # Reset core and stochastic components of environment:
        gym.Env.reset(self, seed=seed)
        self.forecaster = copy.deepcopy(self.original_forecaster) # self.forecaster.set_seed(np.random.randint(low=0,high=2**30))
        self.time = self.forecaster.t_init
        self.episode_end = self.forecaster.t_init + relativedelta(years=+1) - pd.Timedelta(1, 'hour') # Episodic implementation
        self.scenario_number += 1
        self.scenario_path = f"scenario_data/{self.scenario_name}_scenario_{self.scenario_number}/"
        
        self.realized_prices = deque(np.zeros(36),maxlen=36)
        self.realized_ppa = [deque(np.zeros(36),maxlen=36) for _ in range(self.observation_space["context"]["ppas"].shape[1])]
        self.forecast_prices = deque(np.zeros(12+4*24),maxlen=108)

        # --- Reset state ---
        self.storage_state = self.storage_state_space.low
        self.contract_state = self.contract_state_space.low

        # --- Reset context ---
        self._set_context()

        obs = self._get_obs()
        info = {"time": self.time,}
        self.episode_unix_start = get_unix_time()

        return obs, info

    def _step(self, action, hourly_model):
        """
        Perform a step in the environment with the given action.
        Args:
            action np.array: The actions to take. shape:(decision_horizon, n_actions)
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """
        # Compute shielded actions and get correction penalty (0 if not needed).
        
        objective, truncated = hourly_model.get_objective()

        # --- Update state ---
        self.storage_state = np.asarray(list(hourly_model.decision_results.final_soc.values()))
        self.contract_state = np.asarray(list(hourly_model.decision_results.final_contract_status.values()))

        terminated = self.time + pd.Timedelta(self.decision_horizon, 'h') >= self.episode_end # Terminate episode after one year of operations
        reward, info = self._get_reward_and_info(hourly_model, truncated, terminated)
        
        # We also save the current state in the info for possible flexible access.
        # --- Update context ---
        self.time += pd.Timedelta(self.decision_horizon, 'h')
        info["time"] = self.time
        info["terminates_next"] = self.time + pd.Timedelta(self.decision_horizon, 'h') >= self.episode_end
        
        self._set_context()

        obs = self._get_obs()

        if truncated or terminated:
            info["episode_runtime"] = get_unix_time() - self.episode_unix_start

        return obs, reward, terminated, truncated, info

    def _get_reward_and_info(self, hourly_model, truncated, terminated):
        """ Calculate the reward based on the actions taken. """
        horizon = hourly_model.decision_horizon
        if terminated: # If we terminate then we roll 36 hours instead of 24
            prices = np.asarray(list(self.realized_prices))
        else:
            prices = np.asarray(list(self.realized_prices)[24-horizon:24]) # For the first day we only lock 12 hours of decisions.

        contract_revenues = np.sum(list(hourly_model.decision_results.delivered_revenue.values())) # Float
        contract_penalties = np.sum(list(hourly_model.decision_results.contract_penalty.values())) # Float
        
        # Penalty for truncation violations equal to number of hours left in the year
        truncation_penalty = truncated * (self.episode_end-self.time).total_seconds()/3600 # Float

        """ Summarize environment state at the end of the step in the info dict. """
        info = {}

        info["net_market_import"]   = hourly_model.decision_results.spot_power
        # Electricity revenues
        dayahead_buy = np.asarray(hourly_model.decision_results.dayahead_buy)
        info["dayahead_bought"]        = np.sum(dayahead_buy * (dayahead_buy > 0))
        info["dayahead_sold"]          = -np.sum(dayahead_buy * (dayahead_buy < 0))
        info["dayahead_cost"]          = np.sum(dayahead_buy * prices)
        
        info["balancing_power_bought"] = np.sum(hourly_model.decision_results.balancing_buy)
        info["balancing_power_sold"]   = np.sum(hourly_model.decision_results.balancing_sell)
        info["balancing_buy_cost"]     = sum(hourly_model.decision_results.balancing_buy[t] * prices[t] * (1.3 if prices[t] > 0 else 0.7) for t in range(horizon))
        info["balancing_sell_revenue"] = sum(hourly_model.decision_results.balancing_sell[t] * prices[t] * (1.3 if prices[t] < 0 else 0.7) for t in range(horizon))
        info["balancing_cost"]         = info["balancing_buy_cost"] - info["balancing_sell_revenue"]
        info["el_revenue"]             = -(info["dayahead_cost"] + info["balancing_cost"])
        spot_electricity_cost = -info["el_revenue"]

        # Fuel sale summaries
        info["contract_revenues"]           = hourly_model.decision_results.delivered_revenue # Dict
        info["link_productions"]            = hourly_model.decision_results.link_production # Dict
        info["shipments"]                   = hourly_model.decision_results.shipments # Dict

        # Penalties
        info["contract_penalties"]          = hourly_model.decision_results.contract_penalty # Dict
        info["truncation_penalty"]          = truncation_penalty # Float

        # Electricity/power flows
        info["ppa_power"]                   = hourly_model.decision_results.ppa_power # Dict
        info["ppa_cost"]                    = hourly_model.decision_results.ppa_costs # Float
        info["electricity_price"]           = prices # np.ndarray

        # Hourly SOC
        info["storage_soc"]                 = hourly_model.decision_results.storage_soc # Dict
        info["contract_status"]             = hourly_model.decision_results.contract_status # Dict
        
        # Monetary summaries
        info["real_cash_flow"]              = contract_revenues - spot_electricity_cost - contract_penalties # Float

        # The reward can be custom-defined based on how we want to train the agent.
        reward = contract_revenues - spot_electricity_cost - contract_penalties - truncation_penalty # Float

        if truncated: # Inform about infeasibility causing truncation
            info["technical_violation_message"] = "Could not handle the flows determined by the agent."
        
        return reward, info


