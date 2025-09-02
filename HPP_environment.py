import gymnasium as gym
import numpy as np

class Component():
    def __init__(self, name, parameters=None, notes=""):
        """
        Initialize a component of the hybrid power plant.
        Args:
            name (str): The name of the component.
            parameters (dict, optional): Parameters for the component.
        """
        self.name = name
        self.parameters = parameters if parameters is not None else {}
        self.notes = notes

class Contract():
    def __init__(self, name, parameters=None, notes=""):
        """
        Initialize a contract for the hybrid power plant.
        Args:
            name (str): The name of the contract.
            parameters (dict, optional): Parameters for the contract.
        """
        self.name = name
        self.parameters = parameters if parameters is not None else {}
        self.notes = notes

class HybridPowerPlant():
    def __init__(self):
        """
        Initialize the Hybrid Power Plant (HPP) system.
        This class represents a hybrid power plant system with various components.
        """
        self.components = {}
        self.contracts = {}

    def add_contract(self, contract):
        """
        Add a contract to the hybrid power plant system.
        Args:
            contract (object): The contract to add to the system.
        """
        self.contracts[contract.name] = contract
    
    def get_contract(self, name):
        """
        Get a contract by its name.
        Args:
            name (str): The name of the contract to retrieve.
        Returns:
            object: The contract with the specified name, or None if not found.
        """
        return self.contracts.get(name, None)

    def get_contracts(self):
        """
        Get the list of contracts in the hybrid power plant system.
        Returns:
            list: A list of contracts in the system.
        """
        return self.contracts

    def add_component(self, component):
        """
        Add a component to the hybrid power plant system.
        Args:
            component (object): The component to add to the system.
        """
        self.components[component.name] = component

    def get_component(self, name):
        """
        Get a component by its name.
        Args:
            name (str): The name of the component to retrieve.
        Returns:
            object: The component with the specified name, or None if not found.
        """
        return self.components.get(name, None)

    def get_components(self):
        """
        Get the list of components in the hybrid power plant system.
        Returns:
            list: A list of components in the system.
        """
        return self.components

class HPPEnvironment(gym.Env):

    def __init__(self, hpp:HybridPowerPlant):
        """
        Initialize the HPP environment.
        This environment simulates a hypothetical path planning problem.
        """
        self.dt = 1  # Time step in hours
        self.hpp = hpp
        self.state = None  # Placeholder for the state of the environment
        self.time = None  # Placeholder for the current time in the environment
        self.technical_violation_cost = 1.0
        # Define the action space
        # The action space consists of three actions:
        # 1. Amount of electricity to sell to the grid (GCP)
        # 2. Amount of hydrogen to sell through the H2 pipeline
        # 3. Amount of ammonia to store in the ammonia storage
        # The action space is defined as a Box with lower and upper bounds based on the capacities of the components
        self.action_names = ['Grid Power Sale [MW]', 'Electrolyzer Power [MW]', "H2 Pipeline [tH2/h]", "NH3 Production [tNH3]"]
        self.action_space = gym.spaces.Box(low = np.asarray([-self.hpp.get_component("GCP").parameters['capacity'],
                                                             0,
                                                             0,
                                                             0],
                                                             dtype = np.float64),
                                           high = np.asarray([self.hpp.get_component("GCP").parameters['capacity'],
                                                              self.hpp.get_component("Electrolyzer").parameters['capacity'],
                                                              self.hpp.get_component("Hydrogen Pipeline").parameters['capacity'],
                                                              self.hpp.get_component("Haber-Bosch Plant").parameters['capacity']],
                                                              dtype = np.float64),
                                                              dtype = np.float64)
        # Define the observation space
        self.state_names, s_low, s_high = [], [], []
        for name, comp in self.hpp.get_components().items():
            if comp.parameters.get("storage") is not None:
                self.state_names.append(name)
                s_high.append(comp.parameters["storage"]["max"] * comp.parameters["capacity"])
                s_low.append(comp.parameters["storage"]["min"] * comp.parameters["capacity"])
        for name, cont in self.hpp.get_contracts().items():
            if cont.parameters.get("frequency") != "hourly": # Non-hourly contracts are a part of the observation space as this info is needed for the agent to make decisions
                self.state_names.append(name)
                s_high.append(np.inf) # Set large upper bound for non-hourly contracts
                s_low.append(0)
        self.observation_space = gym.spaces.Box(low = np.asarray(s_low),
                                                high = np.asarray(s_high),
                                                dtype = np.float64)

    def step(self, action):
        """
        Perform a step in the environment with the given action.
        Args:
            action np.array: The actions to take.
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """
        truncated = False
        terminated = False
        technical_violations = []
        def _clip(a, a_min, a_max):
            clipped = np.clip(a, a_min=a_min, a_max=a_max)
            violations = clipped - a  # Calculate the penalty for deviating from the action space bounds
            if type(violations) is not np.ndarray:
                technical_violations.extend(np.array([violations]))
            else:
                technical_violations.extend(violations)
            return clipped, violations
        # Ensure the action is within the action space bounds
        action, action_technical_violations = _clip(action, self.action_space.low, self.action_space.high)
        # Get the stochastic inputs for the environment
        p_wind, p_solar, price_el = self.realize_stochasticity()
        # Current state of the environment
        soc_h2, soc_nh3, ytd_nh3, = self.state
        # Actions taken by the agent
        # (1) Amount of electricity to sell to the grid (2) Power flowing to the electrolyzer
        # (3) Amount of hydrogen to sell through the H2 pipeline (4) Amount of ammonia to store
        p_to_grid, p_electrolyzer, m_h2_to_pipeline, m_nh3_to_storage = action

        soc_nh3, nh3_flow_adjustment = _clip(m_nh3_to_storage + soc_nh3, self.observation_space.low[1], self.observation_space.high[1])  # Update the state of charge of ammonia
        m_nh3_to_storage += nh3_flow_adjustment  # Adjust the flow of ammonia to storage based on the action taken
        
        kappa_hb = self.hpp.get_component("Haber-Bosch Plant").parameters["rate"]
        m_h2_to_hb = m_nh3_to_storage * kappa_hb  # Amount of hydrogen to send to the Haber-Bosch plant
        
        eta_electrolyzer = self.hpp.get_component("Electrolyzer").parameters["rate"]  # Efficiency of the electrolyzer [t H2/MWh]
        m_h2_from_electrolyzer = p_electrolyzer * eta_electrolyzer  # Amount of hydrogen produced by the electrolyzer
        m_h2_to_storage = m_h2_from_electrolyzer - m_h2_to_pipeline - m_h2_to_hb # Amount of hydrogen to store in the H2 storage
        soc_h2, h2_flow_adjustment = _clip(m_h2_to_storage + soc_h2, self.observation_space.low[0], self.observation_space.high[0])  # Update the state of charge of hydrogen
        m_h2_to_storage += h2_flow_adjustment  # Adjust the flow of hydrogen to storage if needed
        # If the hydrogen storage could not handle the flows determined, we first adjust the electrolyzer, then the pipeline, and finally the Haber-Bosch plant
        if np.abs(h2_flow_adjustment) > 0:
            p_electrolyzer, pipeline_adjustment = _clip((m_h2_from_electrolyzer + h2_flow_adjustment) / eta_electrolyzer, self.action_space.low[1], self.action_space.high[1])
            if np.abs(pipeline_adjustment) > 0:
                m_h2_to_pipeline, hb_adjustment = _clip(m_h2_to_pipeline + pipeline_adjustment * eta_electrolyzer, self.action_space.low[2], self.action_space.high[2])
                if np.abs(hb_adjustment) > 0:
                    m_nh3_to_storage, warn1 = _clip((m_h2_to_hb - hb_adjustment) / kappa_hb, self.action_space.low[3], self.action_space.high[3])
                    soc_nh3, warn2 = _clip(soc_nh3 - hb_adjustment / kappa_hb, self.observation_space.low[1], self.observation_space.high[1])
                    truncated = np.abs(warn1) > 0 or np.abs(warn2) > 0

        eta_hb = self.hpp.get_component("Haber-Bosch Plant").parameters["rate2"]  # Efficiency of the Haber-Bosch plant [MWh/t NH3]
        p_to_hb = m_nh3_to_storage * eta_hb  # Amount of electricity to send to the Haber-Bosch plant
        p_compressor = (m_h2_to_storage > 0) * m_h2_to_storage * self.hpp.get_component("Hydrogen Storage").parameters["storage"]["charge"]["rate"]  # Power consumed by the compressor to fill the H2 storage. No power used if the storage is discharged.
        p_balancing = p_wind + p_solar - p_to_grid - p_to_hb - p_compressor
        
        # Update the state of the environment based on the actions taken
        if self.time % (30*24) == 0 and self.time > 0:  # Every 30 days, reset the state of the environment
            m_nh3_to_ship = min(self.hpp.get_component("Ammonia Shipment").parameters["capacity"], soc_nh3)  # Amount of ammonia to ship every 30 days
            soc_nh3 -= m_nh3_to_ship  # Update the state of charge of ammonia
            ytd_nh3 += m_nh3_to_ship  # Update the year-to-date ammonia delivery

        self.state = np.asarray([soc_h2, soc_nh3, ytd_nh3], dtype=np.float64)

        # Calculate the reward based on the actions taken
        if p_balancing < 0: # If we are buying unplanned electricity from the grid
            balancing_reward = p_balancing * price_el * 1.5 # Added cost for buying electricity
        else: # If we are selling unplanned electricity to the grid
            balancing_reward = p_balancing * price_el * 0.5 # Reduced reward for selling electricity to the grid
        violation_cost = np.sum(e**2 for e in technical_violations) * self.technical_violation_cost  # Penalty for technical violations

        h2_contract_penalty = max(0, # Penalty for not meeting the hydrogen contract
                                    self.hpp.get_contract("Hydrogen Offtake Agreement").parameters["penalty"] * 
                                    (self.hpp.get_contract("Hydrogen Offtake Agreement").parameters["volume"] - m_h2_to_pipeline))
        # Check if the episode is done
        self.time += self.dt
        terminated = self.time >= 8760  # End after one year (8760 hours)
        nh3_contract_penalty = terminated * max(0, # Penalty for not meeting the ammonia contract
                                    self.hpp.get_contract("Ammonia Offtake Agreement").parameters["penalty"] * 
                                    (self.hpp.get_contract("Ammonia Offtake Agreement").parameters["volume"] - ytd_nh3))
        truncation_penalty = truncated * 50000  # Penalty for truncation violations
        
        info = {}
        info["action"] = action
        info["technical_violation_cost"] = violation_cost
        info["el_spot_revenue"] = p_to_grid * price_el
        info["balancing_revenue"] = balancing_reward
        info["h2_revenue"] = m_h2_to_pipeline * self.hpp.get_contract("Hydrogen Offtake Agreement").parameters["price"]
        info["h2_penalty"] = h2_contract_penalty
        info["nh3_revenue"] = m_nh3_to_storage * self.hpp.get_contract("Ammonia Offtake Agreement").parameters["price"]
        info["nh3_penalty"] = nh3_contract_penalty
        info["truncation_penalty"] = truncation_penalty
        info["balancing"] = p_balancing
        info["wind_power"] = p_wind
        info["solar_power"] = p_solar
        info["electricity_price"] = price_el

        reward_actual = info["el_spot_revenue"] + info["balancing_revenue"] + \
                    info["h2_revenue"] - info["h2_penalty"] + \
                        info["nh3_revenue"] - info["nh3_penalty"] - \
                            (info["technical_violation_cost"] + info["truncation_penalty"])  # Total reward calculation
        info["actual_reward"] = reward_actual

        reward = info["h2_revenue"] + info["nh3_revenue"] - info["technical_violation_cost"]


        if truncated:
            info["technical_violation_message"] = "Could not handle the flows determined by the agent."

        return self.state, reward, terminated, truncated, info

    def realize_stochasticity(self):
        """
        Makes a one-step realization of the stochastic inputs for the environment.
        Returns:
            tuple: A tuple containing the realized wind power, solar power, and electricity price.
        """
        p_wind = np.random.uniform(0, self.hpp.get_component("WindPower").parameters["capacity"])  # Simulated wind power in MW
        p_solar = np.random.uniform(0, self.hpp.get_component("SolarPower").parameters["capacity"])  # Simulated solar power in MW
        price_el = np.random.uniform(0, 100)  # Simulated electricity price in €/MWh
        return p_wind, p_solar, price_el

    def reset(self,
        *,
        seed: int | None = None,):
        """
        Reset the environment to its initial state.
        Returns:
            tuple: The initial state of the environment, and additional info.
        """
        self.state = self.observation_space.low
        self.time = 0
        self.seed = seed

        return self.state, {}


def make_hpp_env():
    """
    Create a hybrid power plant environment with predefined components and contracts.
    """
    hpp = HybridPowerPlant()
    lhv = {"hydrogen": 120, "ammonia": 18.6}  # Lower heating values in GJ/t
    hb_capacity = 50 / lhv["ammonia"] * 3.6  # Capacity in MW
    electrolyzer_capacity = 50 / lhv["hydrogen"] * 3.6  # Capacity in MW
    hpp.add_component(Component(name =      "SolarPower",
                                parameters =    {"capacity": 50, "consumes": "solar", "produces": "electricity", "rate": 1.0},
                                notes =         "Solar power capacity is 50 MWp."
                                ))
    hpp.add_component(Component(name =      "WindPower",
                                parameters =    {"capacity": 100, "consumes": "wind", "produces": "electricity", "rate": 1.0},
                                notes =         "Wind power capacity is 100 MW."
                                ))
    # hpp.add_component(Component("Battery Storage", {"capacity": 100, "consumes": "electricity", "produces": "electricity",
    #                                                 "storage": {"max": 100, "min": 0, "flowcapacity": 100, "charge": {"resource": "electricity", "rate": 0.05}, "discharge": {"resource": "self", "rate": 0.05}}))
    hpp.add_component(Component(name =      "GCP",
                                parameters =    {"capacity": 100, "consumes": "electricity", "produces": "grid_sale", "rate": 1.0, "reversible": True},
                                notes =         "Grid connection point can sell or buy electricity, depending on the action taken. GCP capacity is 100 MW."))
    hpp.add_component(Component(name =      "Electrolyzer",
                                parameters =    {"capacity": 100, "consumes": "electricity", "produces": "hydrogen", "rate": 1/53.5},
                                notes =         "Electrolyzer converts electricity to hydrogen at a rate of 1 t H2 per 53.5 MWh of electricity. The electrolyzer has a capacity of 100 MWe."))
    hpp.add_component(Component(name =      "Hydrogen Pipeline",
                                parameters =    {"capacity": electrolyzer_capacity, "consumes": "hydrogen", "produces": "hydrogen_sale", "rate": 1.0},
                                notes =         "H2 pipeline transports hydrogen to the market/offtaker, which produces a reward.\n" \
                                                "H2 pipeline capacity is 1.5 t/h (50 MW)."))
    hpp.add_component(Component(name =      "Hydrogen Storage",
                                parameters =    {"capacity": 10*electrolyzer_capacity, "consumes": "hydrogen", "produces": "hydrogen",
                                                "storage": {"max": 1.0, "min": 0.0, "flowcapacity": None,
                                                         "charge": {"resource": "electricity", "rate": 0.94},
                                                         "discharge": {"resource": None, "rate": 0.0}}},
                                notes =     "H2 storage costs electricity to fill through compression (0.94 MWh per 1 t of H2).\n" \
                                            "H2 storage capacity is 15 t H2 (500 MWh)."))
    hpp.add_component(Component(name =      "Haber-Bosch Plant",
                                parameters =    {"capacity": hb_capacity, "consumes": "hydrogen", "produces": "ammonia", "rate": 5.29, "consumes2": "electricity", "rate2": 1/0.38},
                                notes =         "Ammonia synthesis rate is 5.29 t NH3 per 1 t of H2 and 1 t of NH3 per 0.38 MWh electricity.\n" \
                                                "Haber-Bosch plant capacity is 19.35 t/h (100 MW)."))
    hpp.add_component(Component(name =      "Ammonia Storage",
                                parameters =    {"capacity": (31*24)*hb_capacity, "consumes": "ammonia", "produces": "ammonia",
                                                "storage": {"max": 1.0, "min": 0.0, "flowcapacity": None,
                                                         "charge": {"resource": None, "rate": 0.0},
                                                         "discharge": {"resource": None, "rate": 0.0}}},
                                notes =         "Ammonia storage is a simple storage component with no losses of flows in or out.\n" \
                                                "Ammonia storage capacity is equal to 31 days of continuous production (big)."))
    hpp.add_component(Component(name =      "Ammonia Shipment",
                                parameters =    {"capacity": 8760*hb_capacity/2/12, "consumes": "ammonia", "produces": "ammonia_sale", "rate": 1.0},
                                notes =         "Ammonia shipment transports ammonia to the market/offtaker, which produces a reward."))
    hpp.add_contract(Contract(name =        "Hydrogen Offtake Agreement",
                              parameters =      {"price": 2000, "volume": electrolyzer_capacity/2, "frequency": "hourly", "penalty": 0.5},
                              notes =           "Contract for selling hydrogen to an off-taker at a fixed price. Unit price is in €/t H2."))
    hpp.add_contract(Contract(name =        "Ammonia Offtake Agreement",
                              parameters =      {"price": 300, "volume": 8760*hb_capacity/2, "frequency": "yearly", "penalty": 0.25},
                              notes =           "Contract for selling ammonia to an off-taker at a fixed price. Unit price is in €/t NH3."))
    
    env = HPPEnvironment(hpp)
    return env
    

