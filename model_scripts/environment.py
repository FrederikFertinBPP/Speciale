
import gymnasium as gym
import numpy as np
import pandas as pd
from collections import deque
from common_scripts.utils import cache_read, expando
from common_scripts.RFP_initialization import RenewableFuelPlant, create_rfp
from data_scripts.data_generator_v2 import DataForecaster
from sklearn.preprocessing import MinMaxScaler
from time import time as get_unix_time
import copy
from dateutil.relativedelta import relativedelta
import os
from model_scripts.hourly_models import ShieldLPModel, HourlyDeterministicLPModel, HourlyRecourseModel, ShieldRecourseModel

def get_env(env_class:gym.Env,
            env_config:dict={},
            scenario_name:str="default",
            data_cache_id:str="Anders",
            layout_file:str="rfp_layout.xlsx",
            ):
    rfp = create_rfp(scenario_name=scenario_name, layout_file=layout_file)

    forecaster = DataForecaster(from_pickle=True, cache_id=data_cache_id)
    forecaster = forecaster.unpickle()
    forecaster.t_init = forecaster.test_data.index[0]
    if env_config.get("load_data", False):
        forecaster_shell = expando()
        forecaster_shell.t_init = forecaster.t_init
        forecaster_shell.cache_id = forecaster.cache_id
        forecaster_shell.database = forecaster.database
        forecaster = forecaster_shell
    
    env = env_class(rfp=rfp, forecaster=forecaster, decision_horizon=24, **env_config)
    
    return env


class EmissionFactorEstimator:
    """  """
    def __init__(self, model):
        self.model = model
    
    def __call__(self, *args, **kwds):
        return np.clip(self.model.predict(*args, **kwds), 0, np.inf)


class VRESystemToAssetMapping:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, *args, **kwds):
        return np.clip(self.model(*args, **kwds), 0, 1)


class RFPBaseEnv(gym.Env):
    """ Environment, which allows for operating a Renewable Fuel Plant in a rolling horizon fashion. """
    realization_memory_size = 24
    
    def __init__(self,
                 rfp:RenewableFuelPlant,
                 forecaster:DataForecaster = None,
                 decision_horizon:int = 24, # Unit: hours
                 allow_spot_buy = True,
                 inflexible:bool = False,
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
        self.inflexible = inflexible
        # Whether to normalize the state and action spaces (dependent on the algorithm used for decision-making)
        # Should maybe be set by the agent instead?
        self.normalize_step = normalize
        self.balancing_market = balancing_market
        self.load_data = load_data
        self.data_cache_id = "" if (forecaster is None) else forecaster.cache_id
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
        # Calculates "Carbon intensity gCO₂eq/kWh (direct)" as a linear function of price [€/MWh], system wind [MW], and system solar [MW].
        cache_path_mappers = os.getcwd() + "/models/plant_models/"
        emissions_mapper = cache_read(cache_path_mappers + "emission_factor.pkl")
        self.emissions_model = EmissionFactorEstimator(emissions_mapper)

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
                         "inflexible": inflexible,
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
        self.emissions_context_space = gym.spaces.Box(low=0, high = np.asarray([1] * self.decision_horizon), dtype = np.float64) # tCO2/MWh

        self.offtaker_names = []
        for name, offtaker in self.rfp.get_offtakers().items():
            self.offtaker_names.append(name)
        self.offtaker_context_space = gym.spaces.MultiBinary(n=(self.decision_horizon*14, len(self.offtaker_names)))

        self.context_space = gym.spaces.Dict({"time": self.time_context_space,
                                              "ppas": self.ppa_context_space,
                                              "offtakers": self.offtaker_context_space,
                                              "prices": self.price_context_space,
                                              "emissions": self.emissions_context_space,
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
        self.system_solar_realization, self.system_wind_realization = system_solar_realization['solar'].values, system_wind_realization['wind'].values

        for ix, (name, ppa) in enumerate(self.rfp.get_ppas().items()):
            if ppa.parameters.get("consumes") == 'wind':
                cf = self.wind_mapper(self.system_wind_realization)
            elif ppa.parameters.get("consumes") == 'solar':
                cf = self.solar_mapper(self.system_solar_realization)
            else:
                cf = np.ones(len(self.system_solar_realization)) # Assumes full availability of non-variable PPAs.
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
            price_forecast = pd.read_csv(f"{self.scenario_path}forecast_{timestamp_str}_0.csv", usecols=["price"], nrows=24)
            price_realization = pd.read_csv(f"{self.scenario_path}prices_{timestamp_str}.csv")
        else:
            price_forecast = self.forecaster.forecast(start=self.time, end=self.time+pd.Timedelta(self.decision_horizon-1, 'h'), n_forecasts=1)[0]
            price_realization = self.forecaster.realize_prices(start=self.time, end=self.time+pd.Timedelta(self.decision_horizon-1, 'h'))
        self.realized_prices.extend(price_realization['price'].values)
        
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
        self.price_context = price_forecast['price'].values

    def _set_emissions_context(self):
        # This where we map from simulated wind, solar, and prices or we draw from pregenerated scenario file.
        hourly_index = pd.to_datetime(pd.date_range(self.time, self.time + pd.Timedelta(self.decision_horizon-1, 'hour'), freq='h'), utc=True)
        year_month_index = hourly_index.tz_localize(None).to_period('M')
        solar_capacities = year_month_index.map(self.forecaster.database.caps['solar'])
        wind_capacities = year_month_index.map(self.forecaster.database.caps['wind'])
        solar = self.system_solar_realization * solar_capacities
        wind = self.system_wind_realization * wind_capacities
        real_prices = np.asarray(list(self.realized_prices)[-self.decision_horizon:])
        forecast_prices = self.price_context
        X_forecast = pd.DataFrame(data={"price":forecast_prices, "wind":wind, "solar":solar})
        X_real = pd.DataFrame(data={"price":real_prices, "wind":wind, "solar":solar})
        forecast_emissions = self.emissions_model(X_forecast) / 1000 # Convert to unit tCO2/MWh.
        real_emissions = self.emissions_model(X_real) / 1000 # Convert to unit tCO2/MWh.

        self.realized_emissions.extend(real_emissions)
        self.emissions_context = forecast_emissions

    def _set_context(self, terminated=False):
        if not(terminated):
            self._set_time_context()
            self._set_ppa_context()
            self._set_offtaker_context()
            self._set_price_context()
            self._set_emissions_context()

    def _get_obs(self):
        """ Convert internal state to observation format.
        Returns:
            dict: Observation of state and context
        """
        self.state   = {"storages": self.storage_state, "contracts": self.contract_state}
        self.context = {"time": self.time_context, "ppas": self.ppa_context, "offtakers": self.offtaker_context, "prices": self.price_context, "emissions": self.emissions_context}
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
        if options is not None:
            self.scenario_number = options.get("scenario_number", self.scenario_number)
        
        self.scenario_path = f"scenario_data/{self.data_cache_id}_scenario_{self.scenario_number}/"
        
        self.realized_prices    = deque(np.zeros(self.realization_memory_size), maxlen=self.realization_memory_size)
        self.realized_ppa       = [deque(np.zeros(self.realization_memory_size), maxlen=self.realization_memory_size) for _ in range(self.observation_space["context"]["ppas"].shape[1])]
        self.realized_emissions = deque(np.zeros(self.realization_memory_size), maxlen=self.realization_memory_size)

        # --- Reset state ---
        self.storage_state = self.storage_state_space.low
        self.contract_state = self.contract_state_space.low

        # --- Reset context ---
        self._set_context()

        obs = self._get_obs()
        info = {"time": self.time,}
        self.episode_unix_start = get_unix_time()

        return obs, info

    def _get_prices_and_emissions_for_step(self, horizon, terminated):
        # Realize prices - already done when context is set.
        prices = np.asarray(list(self.realized_prices)[-horizon:])
        grid_emissions = np.asarray(list(self.realized_emissions)[-horizon:])
        return prices, grid_emissions

    def _get_reward_and_info(self, hourly_model, truncated, terminated, shield_penalty=0):
        """ Calculate the reward based on the actions taken. """
        res = hourly_model.decision_results
        horizon = hourly_model.decision_horizon

        prices, grid_emissions = self._get_prices_and_emissions_for_step(horizon, terminated)

        contract_revenues = np.sum(list(res.delivered_revenue.values())) # Float
        contract_penalties = np.sum(list(res.contract_penalty.values())) # Float
        
        # Penalty for truncation violations equal to number of hours left in the year
        truncation_penalty = truncated * (self.episode_end-self.time).total_seconds()/3600 # Float

        """ Summarize environment state at the end of the step in the info dict. """
        info = {}

        # Electricity revenues
        info["net_market_import"] = res.spot_power # np.ndarray

        info["dayahead_buy_profile"]   = res.da_buy * (res.da_buy > 0) # np.ndarray
        info["dayahead_bought"]        = np.sum(info["dayahead_buy_profile"]) # Float
        info["dayahead_sell_profile"]  = -res.da_buy * (res.da_buy < 0) # np.ndarray
        info["dayahead_sold"]          = np.sum(info["dayahead_sell_profile"]) # Float
        info["dayahead_cost"]          = np.sum(res.da_buy * prices) # Float
        info["el_revenue"]             = -info["dayahead_cost"]
        if self.balancing_market:
            info["balancing_buy_profile"] = res.balancing_buy # np.ndarray
            info["balancing_bought"] = np.sum(res.balancing_buy) # Float
            info["balancing_buy_prices"] = np.asarray([prices[t] * (1.3 if prices[t] > 0 else 0.7) for t in range(horizon)]) # np.ndarray
            info["balancing_sell_profile"] = res.balancing_sell # np.ndarray
            info["balancing_sold"]   = np.sum(res.balancing_sell) # Float
            info["balancing_sell_prices"] = np.asarray([prices[t] * (1.3 if prices[t] < 0 else 0.7) for t in range(horizon)]) # np.ndarray
            info["balancing_buy_cost"]     = sum(res.balancing_buy[t] * info["balancing_buy_prices"][t] for t in range(horizon)) # Float
            info["balancing_sell_revenue"] = sum(res.balancing_sell[t] * info["balancing_sell_prices"][t] for t in range(horizon)) # Float
            
            info["balancing_cost"] = info["balancing_buy_cost"] - info["balancing_sell_revenue"] # Float
            info["el_revenue"] -= info["balancing_cost"] # Float

        # Fuel sale summaries
        info["contract_revenues"]       = res.delivered_revenue # Dict
        info["link_productions"]        = res.link_production # Dict
        info["shipments"]               = res.shipments # Dict

        # Penalties
        info["contract_penalties"]      = res.contract_penalty # Dict
        info["truncation_penalty"]      = truncation_penalty # Float

        # Electricity/power flows
        info["ppa_power"]               = res.ppa_power # Dict
        info["ppa_cost"]                = res.ppa_costs # Float
        info["electricity_price"]       = prices # np.ndarray
        info["electricity_emissions"]   = grid_emissions
        info["power_consumption"]       = res.power_consumption

        # Hourly SOC
        info["storage_soc"]             = res.storage_soc # Dict
        info["contract_status"]         = res.contract_status # Dict
        
        # Monetary summaries
        info["real_cash_flow"]          = contract_revenues + info["el_revenue"] - contract_penalties - res.ppa_costs # Float

        # The reward can be custom-defined based on how we want to train the agent.
        reward = contract_revenues + info["el_revenue"] - contract_penalties - res.ppa_costs - truncation_penalty - shield_penalty # Float

        if truncated: # Inform about infeasibility causing truncation
            info["technical_violation_message"] = "Could not handle the flows determined by the agent."
        
        return reward, info


class RFPShieldEnv(RFPBaseEnv):
    def __init__(self,
                 rfp:RenewableFuelPlant,
                 forecaster:DataForecaster = None,
                 decision_horizon:int = 24, # Unit: hours
                 allow_spot_buy = True,
                 inflexible:bool = False,
                 normalize:bool = False,
                 verbose:bool = False,
                 balancing_market:bool = False,
                 load_data = True,
                 **kwargs,
                 ):
        super().__init__(rfp, forecaster, decision_horizon, allow_spot_buy, inflexible, normalize, verbose, balancing_market, load_data, **kwargs)
        shield_class = ShieldRecourseModel if self.balancing_market else ShieldLPModel
        self.shield = shield_class(self,
                                    rfp,
                                    decision_horizon=decision_horizon,
                                    solver='gurobi',
                                    allow_spot_buy=allow_spot_buy,
                                    inflexible=inflexible,
                                    )
        self.shield.initialize_model()

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

        terminated = self.time + pd.Timedelta(self.decision_horizon, 'h') >= self.episode_end # Terminate episode after one year of operations
        reward, info = self._get_reward_and_info(self.shield, truncated, terminated, shield_penalty)
        
        # We also save the current state in the info for possible flexible access.
        # --- Update context ---
        self.time += pd.Timedelta(self.decision_horizon, 'h')
        info["time"] = self.time
        
        self._set_context(terminated=terminated)

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
    def __init__(self, rfp, forecaster = None, allow_spot_buy=True, inflexible=False, normalize = False, verbose = False, load_data=True, **kwargs):
        self.original_forecaster = forecaster
        t = self.original_forecaster.t_init
        t_end = self.original_forecaster.t_init + relativedelta(years=+1) # Episodic implementation
        horizon = (t_end - t).days * 24 # Number of hours in the year.
        self.realization_memory_size = horizon
        super().__init__(rfp, forecaster, decision_horizon=horizon, allow_spot_buy=allow_spot_buy, inflexible=inflexible, normalize=normalize, verbose=verbose, load_data=load_data)
        self.pfm = HourlyDeterministicLPModel(rfp, decision_horizon=horizon, solver='gurobi', allow_spot_buy=allow_spot_buy, inflexible=inflexible)
        self.pfm.initialize_model()

    def _set_ppa_context(self):
        # Realize VRE PPA availability for the next decision horizon (typically 24 hours)
        if self.load_data:
            system_solar_realization, system_wind_realization = [],[]
            for day in range((self.episode_end - self.time).days+1):
                timestamp_str = (self.time+pd.Timedelta(day,'days')).strftime("%Y%m%d")
                system_solar_realization += list(pd.read_csv(f"{self.scenario_path}solar_{timestamp_str}.csv")['solar'].values)
                system_wind_realization += list(pd.read_csv(f"{self.scenario_path}wind_{timestamp_str}.csv")['wind'].values)
            self.system_solar_realization = np.asarray(system_solar_realization)
            self.system_wind_realization = np.asarray(system_wind_realization)
        else:
            system_solar_realization, system_wind_realization = self.forecaster.realize_vre(start=self.time, end=self.time + pd.Timedelta(self.decision_horizon-1, 'h')) # DF
            self.system_solar_realization, self.system_wind_realization = system_solar_realization['solar'].values, system_wind_realization['wind'].values
        profiles = []
        for name, ppa in self.rfp.get_ppas().items():
            if ppa.parameters.get("consumes") == 'wind':
                cf = self.wind_mapper(self.system_wind_realization)
            elif ppa.parameters.get("consumes") == 'solar':
                cf = self.solar_mapper(self.system_solar_realization)
            else:
                cf = np.ones(len(self.system_solar_realization)) # Assumes full availability of non-variable PPAs.
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
        self.realized_prices.extend(price_realization)
        self.price_context = price_forecast

    def _get_reward_and_info(self, hourly_model, truncated, terminated, shield_penalty=0):
        """ Calculate the reward based on the actions taken. """
        reward, info = super()._get_reward_and_info(hourly_model, truncated, terminated, shield_penalty)

        prices = info["electricity_price"]

        time_index = pd.to_datetime(pd.date_range(self.time, self.time+pd.Timedelta(self.decision_horizon-1, 'h'), freq='h'), utc=True)#prices.index
        
        supplier_cf = {}
        for ix, ppa_name in enumerate(self.ppa_names):
            ppa = self.rfp.get_ppa(ppa_name)
            ppa_profile = self.ppa_context[:,ix]
            if ppa.parameters.get("consumes") == 'wind':
                ppa_profile = self.wind_mapper(ppa_profile)
            elif ppa.parameters.get("consumes") == 'solar':
                ppa_profile = self.solar_mapper(ppa_profile)
            supplier_cf = {**supplier_cf, **{(ppa_name, t): ppa_profile[t] for t in range(self.decision_horizon)}}

        electricity_price = {t: prices[t] for t in range(self.decision_horizon)}
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


class RFPModelActionsEnv(RFPBaseEnv):
    step_with_hourly_model = True

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

        self._set_context(terminated=terminated)

        obs = self._get_obs()

        if truncated or terminated:
            info["episode_runtime"] = get_unix_time() - self.episode_unix_start

        return obs, reward, terminated, truncated, info

    def _normalized_step(self, normalized_action, hourly_model):
        # Denormalize the action to the original action space
        action = self.action_scaler.inverse_transform(normalized_action)
        return self._step(action, hourly_model)

    def step(self, action, hourly_model):
        """
        Perform a step in the environment with the given normalized action.
        Args:
            action (np.array): The action in the range [0, 1].
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """
        if self.normalize_step:
            return self._normalized_step(action, hourly_model)
        else:
            return self._step(action, hourly_model)


class RFPRecourseEnv(RFPModelActionsEnv):
    balancing_market = True
    realization_memory_size = 36
    
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
        self.emissions_context_space = gym.spaces.Box(low=0, high = np.asarray([1] * self.decision_horizon), dtype = np.float64) # tCO2/MWh

        self.offtaker_names = []
        for name, offtaker in self.rfp.get_offtakers().items():
            self.offtaker_names.append(name)
        self.offtaker_context_space = gym.spaces.MultiBinary(n=(self.decision_horizon*14, len(self.offtaker_names)))

        self.context_space = gym.spaces.Dict({"time": self.time_context_space,
                                              "ppas": self.ppa_context_space,
                                              "offtakers": self.offtaker_context_space,
                                              "prices": self.price_context_space,
                                              "emissions": self.emissions_context_space,
                                              })

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

    def _get_prices_and_emissions_for_step(self, horizon, terminated):
        if terminated:
            # If we terminate then we roll 36 hours instead of 24
            prices = np.asarray(list(self.realized_prices))
            grid_emissions = np.asarray(list(self.realized_emissions))
        else:
            # For the first day we only lock 12 hours of decisions.
            prices = np.asarray(list(self.realized_prices)[24-horizon:24])
            grid_emissions = np.asarray(list(self.realized_emissions)[24-horizon:24])
        return prices, grid_emissions

    def _step(self, action, hourly_model):
        obs, reward, terminated, truncated, info = super()._step(action, hourly_model)
        info["terminates_next"] = self.time + pd.Timedelta(self.decision_horizon, 'h') >= self.episode_end
        return obs, reward, terminated, truncated, info


class RFPBackcastEnv(RFPModelActionsEnv):
    def _step(self, action, hourly_model):
        obs, reward, terminated, truncated, info = super()._step(action, hourly_model)
        info["forecast_path"] = self.forecast_path
        truncated = self.truncated
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options = None):
        """
        Reset the environment to its initial state.
        Returns:
            tuple: The initial state of the environment, and additional info.
        """
        # Reset core and stochastic components of environment:
        gym.Env.reset(self, seed=seed)

        filename = "historical_data/clean_dataframes/backcasting_timeseries.csv"
        self.time = pd.Timestamp('2017-01-01 00:00:00')
        self.forecaster_type = "forecaster"
        if options is not None:
            filename = options.get("historical_data_path", filename)
            self.time = options.get("episode_start", self.time)
            self.forecaster_type = options.get("forecaster_type", self.forecaster_type) # Options: ("forecaster", "prophet", "persistence")
        self.historical_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        self.historical_data.index = self.historical_data.index.tz_localize(None)
        self.episode_end = self.historical_data.index[-1]

        self.forecast_path = f"scenario_data/Historicals/{self.forecaster_type}/"
        
        self.realized_prices    = deque(np.zeros(self.realization_memory_size), maxlen=self.realization_memory_size)
        self.realized_ppa       = [deque(np.zeros(self.realization_memory_size), maxlen=self.realization_memory_size)
                                   for _ in range(self.observation_space["context"]["ppas"].shape[1])]
        self.realized_emissions = deque(np.zeros(self.realization_memory_size), maxlen=self.realization_memory_size)

        # --- Reset state ---
        self.storage_state = self.storage_state_space.low
        self.contract_state = self.contract_state_space.low

        # --- Reset context ---
        self.truncated = False
        self._set_context()

        obs = self._get_obs()
        info = {"time": self.time, "forecast_path": self.forecast_path,}
        self.episode_unix_start = get_unix_time()

        return obs, info

    def _set_ppa_context(self):
        # Realize VRE PPA availability for the next decision horizon (typically 24 hours)
        hist_slice = self.historical_data.loc[self.time:self.time+pd.Timedelta(self.decision_horizon-1,'h')]
        self.system_solar_realization, self.system_wind_realization = hist_slice['solar'].values, hist_slice['wind'].values

        for ix, (name, ppa) in enumerate(self.rfp.get_ppas().items()):
            if ppa.parameters.get("consumes") == 'wind':
                cf = self.wind_mapper(self.system_wind_realization)
            elif ppa.parameters.get("consumes") == 'solar':
                cf = self.solar_mapper(self.system_solar_realization)
            else:
                cf = np.ones(len(self.system_solar_realization)) # Assumes full availability of non-variable PPAs.
            self.realized_ppa[ix].extend(cf)
        self.ppa_context = np.transpose(np.asarray(self.realized_ppa))

    def _set_price_context(self):
        # Realize prices - updates the forecaster object by including the new realizations.
        try:
            timestamp_str = self.time.strftime("%Y-%m-%d")
            price_forecast = pd.read_csv(f"scenario_data/Historicals/{self.forecaster_type}/forecast_{timestamp_str}.csv", usecols=["price"], nrows=24)
            self.price_context = price_forecast['price'].values
        except FileNotFoundError:
            self.truncated = True
        hist_slice = self.historical_data.loc[self.time:self.time+pd.Timedelta(self.decision_horizon-1,'h')]
        self.realized_prices.extend(hist_slice['price'].values)

    def _set_emissions_context(self):
        # This where we map from simulated wind, solar, and prices or we draw from pregenerated scenario file.
        # hourly_index = pd.to_datetime(pd.date_range(self.time, self.time + pd.Timedelta(self.decision_horizon-1, 'hour'), freq='h'), utc=True)
        # year_month_index = hourly_index.tz_localize(None).to_period('M')
        # solar_capacities = year_month_index.map(self.forecaster.database.caps['solar'])
        # wind_capacities = year_month_index.map(self.forecaster.database.caps['wind'])
        # solar = self.system_solar_realization * solar_capacities
        # wind = self.system_wind_realization * wind_capacities
        # forecast_prices = self.price_context
        # X_forecast = pd.DataFrame(data={"price":forecast_prices, "wind":wind, "solar":solar})
        # forecast_emissions = self.emissions_model(X_forecast) / 1000 # Convert to unit tCO2/MWh.
        real_emissions = self.historical_data.loc[self.time:self.time+pd.Timedelta(self.decision_horizon-1,'h'),"emissions"].values / 1000 # Convert to unit tCO2/MWh.

        self.realized_emissions.extend(real_emissions)
        self.emissions_context = self.context_space["emissions"].low


class RFPBackcastRecourseEnv(RFPBackcastEnv,RFPRecourseEnv):
    pass

