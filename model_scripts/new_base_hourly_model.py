from common_scripts.RFP_initialization import RenewableFuelPlant #, create_rfp
from common_scripts import expando
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
import numpy as np
import pandas as pd


class BaseHourlyModel:
    frequency_options = ("hourly", "daily", "monthly", "yearly", "planning_horizon", None)
    guideline_options = ("production_value", "hourly_target", None)
    model_types = ("recourse DA", "non-recourse DA", "non-recourse flows", "capacity_planning", None)

    def __init__(self,
                 rfp: RenewableFuelPlant,
                 *args,
                 inflexible: bool = False,
                 enforce_rfnbo: bool = False,
                 planning_horizon: int = 4*24,
                 decision_horizon: int = 24,
                 fixed_horizon: int = 12,
                 solver: str = 'gurobi',
                 compute_duals: bool = False,
                 allow_spot_buy: bool = True,
                 guideline: str|None = None,
                 objective_logic: str|None = None,
                 n_scenarios: int = 1, # If 1, then it is identical to the deterministic model.
                 model_type: str|None = None,
                 discount_rate: float = 0.1, 
                 cvar_info: dict|bool|None = None,
                 documentation: bool = False,
                 **kwargs,
                 ):
        # Problem specific parameters:
        self.rfp              = rfp
        self.inflexible       = inflexible
        self.min_load_active  = inflexible
        self.enforce_rfnbo    = enforce_rfnbo
        self.allow_spot_buy   = allow_spot_buy

        self.decision_horizon = decision_horizon
        self.planning_horizon = max(planning_horizon, self.decision_horizon)
        self.fixed_horizon    = min(fixed_horizon, self.decision_horizon)
        self.n_scenarios      = n_scenarios

        assert guideline in self.guideline_options, "f{guideline} not in guideline options: {self.guideline_options}"
        self.guideline        = guideline.lower()
        self.objective_logic  = objective_logic.lower()
        self.model_type       = model_type.lower()
        
        self.cvar_formulation = True if cvar_info is not None else False
        if self.cvar_formulation:
            self.cvar_alpha = cvar_info.get("alpha", 0.9)
            self.cvar_beta = cvar_info.get("beta", 0.5)

        # Capacity Planner Specifics:
        self.discount_rate = discount_rate
        self.capacity_planning = bool(self.model_type == "capacity_planning")
        if self.capacity_planning:
            self.inflexible = False

        self.steering_variables = {} # Placeholder for the parameters, which should be steered.

        # Objects to build the LP around:
        self.model = None
        self.inst  = None
        self.planning_results = expando()
        self.decision_results = expando()
        self.flow_results = expando()
        self.soc_results = expando()

        # Solver setup:
        if solver == 'scip': # 'ipopt'
            self.solver        = SolverFactory(solver, solver_io='nl')
        if solver == 'gurobi_persistent':
            self.solver        = SolverFactory('gurobi_persistent')
        else:
            self.solver        = SolverFactory(solver)
        self.uses_persistent_solver = issubclass(self.solver.__class__, PersistentSolver)
        self.compute_duals = compute_duals
        self.documentation = documentation # If true, the model will store extra information useful for documentation of results.

    def initialize_model(self):
        # Initialize the optimization model
        self.model = pyo.AbstractModel()
        self._build_abstract_model()
        if self.uses_persistent_solver:
            self._build_concrete_instance() # Creates self.inst
            self.solver.set_instance(self.inst)

    def _build_abstract_model(self):
        # Stochastic scenarios:
        self.model.S    = pyo.RangeSet(0, self.n_scenarios-1) # Set of price scenarios
        self.model.weights = pyo.Param(self.model.S, within=pyo.NonNegativeReals, default=1/self.n_scenarios, mutable=True)
        
        # Model Time Sets:
        self.model.T    = pyo.RangeSet(0, self.fixed_horizon + self.planning_horizon - 1)  # Time steps
        self.model.T_fix_dayahead = pyo.RangeSet(self.fixed_horizon, self.fixed_horizon + self.decision_horizon - 1) # Used for results processing
        self.model.T_fix_recourse = pyo.RangeSet(0, self.decision_horizon - 1) # Used for results processing

        # Model Set definitions
        self.model.carriers     = pyo.Set(initialize=[name for name in self.rfp.get_carriers().keys()])
        self.model.storages     = pyo.Set(initialize=[name for name in self.rfp.get_storages().keys()])
        self.model.ppas         = pyo.Set(initialize=[name for name in self.rfp.get_ppas().keys()])
        self.model.dayaheads    = pyo.Set(initialize=[name for name in self.rfp.get_dayaheads().keys()])
        self.model.links        = pyo.Set(initialize=[name for name in self.rfp.get_links().keys()])
        self.model.offtakers    = pyo.Set(initialize=[name for name in self.rfp.get_offtakers().keys()])
        self.model.contracts    = pyo.Set(initialize=[name for name in self.rfp.get_contracts().keys()])

        ### Mutable model parameters ###

        # Datetime info:
        self.model.T_datetime = pyo.Param(self.model.S, self.model.T, within=pyo.Any, initialize=pd.date_range(start=0, end=self.planning_horizon - 1, freq='h'), mutable=True)
        self.model.timedelta  = pyo.Param(self.model.T, within=pyo.NonNegativeReals, default=1, mutable=True) # Time step duration in hours. Used for scaling of flows.

        # Initial setpoints:
        self.model.init_contract_status = pyo.Param(self.model.contracts, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.init_soc             = pyo.Param(self.model.storages, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.prev_link_setpoints  = pyo.Param(self.model.links, within=pyo.NonNegativeReals, default=0.5, mutable=True)

        # Time series:
        self.model.supplier_cf              = pyo.Param(self.model.ppas, self.model.T, within=pyo.NonNegativeReals, default=1, mutable=True)
        self.model.electricity_price        = pyo.Param(self.model.T, within=pyo.Reals, default=50, mutable=True)
        self.model.grid_emissions_intensity = pyo.Param(self.model.T, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.ets_price                = pyo.Param(self.model.T, within=pyo.Reals, default=0, mutable=True)

        # Recourse intraday stage:
        self.model.cleared_power = pyo.Param(self.model.T, within=pyo.Reals, default=0, mutable=True) # Cleared DA market power. Only enforced if fixed_da is True.
        self.model.fixed_da = pyo.Param(self.model.T, within=pyo.Binary, default=0, mutable=True) # Whether we should fix the DA decision to cleared_power parameter.

        # If extra spot deal shipment is needed:
        self.model.spot_shipment = pyo.Param(within=pyo.Binary, default=0, mutable=True)

        # Guideline related mutable parameters:
        if self.guideline == "production_value": # Specified value of outflow of links.
            self.model.production_value = pyo.Param(self.model.links, within=pyo.Reals, default=0, mutable=True)
            self.model.shipment_value = pyo.Param(self.model.contracts, within=pyo.Reals, default=0, mutable=True)
            self.steering_variables[self.guideline] = {key:0 for key in self.rfp.get_links().keys()}
        elif self.guideline == "hourly_target": # Hourly target for ammonia production.
            self.model.hourly_target = pyo.Param(within=pyo.NonNegativeReals, default=0, mutable=True)
            self.steering_variables[self.guideline] = {None:0}
        if self.objective_logic == "value_maximization":
            self.model.storage_value = pyo.Param(self.model.storages, within=pyo.Reals, default=0, mutable=True)
            self.model.contract_value = pyo.Param(self.model.contracts, within=pyo.Reals, default=0, mutable=True)
            self.steering_variables[self.objective_logic] = {}
            self.steering_variables[self.objective_logic]["storage_value"] = {key:0 for key in self.rfp.get_storages().keys()}
            self.steering_variables[self.objective_logic]["contract_value"] = {key:0 for key, contract in self.rfp.get_contracts().items() if not(contract.parameters.get("spot_contract", 0))}

        def carrierBlock_rule(b, carr):
            """ Create a block for each energy carrier to enable nodal carrier balance enforcement. """
            carrier = self.rfp.get_carrier(carr)
            b.type = carrier.name
            b.carrier_in = b.type
            b._in = {(s,t): [] for s in self.model.S for t in self.model.T}
            b._out = {(s,t): [] for s in self.model.S for t in self.model.T}
        self.model.carrierBlocks = pyo.Block(self.model.carriers, rule=carrierBlock_rule)

        def storageBlock_rule(b, stor): # Create a block for each storage to handle charge/discharge and state of charge
            storage         = self.rfp.get_component(stor)
            b._name         = storage.name
            b.capacity      = storage.parameters["capacity"]
            b.ec            = storage.parameters.get("electricity_consumption", 0) # Electricity consumption rate
            b.carrier_in    = str(storage.parameters["consumes"])
            b.carrier_out   = str(storage.parameters["produces"])

            b.soc       = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            b.in_flow   = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            b.out_flow  = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            if b.ec > 0:
                b.elec_cons = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity*b.ec))
                def ec_rule(m, s, t):
                    return b.elec_cons[s,t] == b.in_flow[s,t] * b.ec
                b.ec_constraint = pyo.Constraint(self.model.S, self.model.T, rule=ec_rule)
            if self.model_type == "non_recourse flows":
                b.main_soc       = pyo.Var(self.model.T_fix_recourse, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
                def main_soc_rule(m, s, t):
                    return b.main_soc[t] == b.soc[s,t]
                b.main_soc_constraint = pyo.Constraint(self.model.S, self.model.T_fix_recourse, rule=main_soc_rule)
        self.model.storageBlocks = pyo.Block(self.model.storages, rule=storageBlock_rule)

        def ppaBlock_rule(b, ppa): # Create a block for each ppa to handle production.
            # A bit too complexly implemented, but allows for potential other structure than PPAs.
            ppa_        = self.rfp.get_ppa(ppa)
            b._name         = ppa
            b.carrier_in    = str(ppa_.parameters["consumes"])
            b.carrier_out   = str(ppa_.parameters["produces"])
            b.capacity      = ppa_.parameters.get('capacity')
            b.price         = ppa_.parameters.get('price')
            b.out_flow      = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
        self.model.ppaBlocks = pyo.Block(self.model.ppas, rule=ppaBlock_rule)

        def dayaheadBlock_rule(b, da):
            dayahead        = self.rfp.get_component(da)
            b._name         = da
            b.carrier_in    = str(dayahead.parameters["consumes"])
            b.carrier_out   = str(dayahead.parameters["produces"])
            b.capacity      = dayahead.parameters.get('capacity')
            # Power bought from the day-ahead market; negative if power is sold.
            b.out_flow = pyo.Var(self.model.S, self.model.T, domain=pyo.Reals, bounds=(-b.capacity, b.capacity * self.allow_spot_buy))
            if self.model_type == "recourse DA": # ! Not otherwise implemented.
                b.da_buy = pyo.Var(self.model.S, self.model.T, domain=pyo.Reals, bounds=(-b.capacity, b.capacity * self.allow_spot_buy)) # * Only non-recourse decision.
            else:
                b.da_buy = pyo.Var(self.model.T, domain=pyo.Reals, bounds=(-b.capacity, b.capacity * self.allow_spot_buy)) # * Only non-recourse decision.
            b.ba_buy = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            b.ba_sell = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            def balancing_rule(m, s, t):
                if self.model_type == "recourse DA": # ! Not otherwise implemented.
                    return b.out_flow[s,t] == b.da_buy[s,t] + b.ba_buy[s,t] - b.ba_sell[s,t]
                else:
                    return b.out_flow[s,t] == b.da_buy[t] + b.ba_buy[s,t] - b.ba_sell[s,t]
            b.balancing_constraint = pyo.Constraint(self.model.S, self.model.T, rule=balancing_rule)
            if self.model_type == "non_recourse flows":
                b.main_ba_buy = pyo.Var(self.model.T_fix_recourse, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
                def main_ba_buy_rule(m, s, t):
                    return b.main_ba_buy[t] == b.ba_buy[s,t]
                b.main_ba_buy_constraint = pyo.Constraint(self.model.S, self.model.T_fix_recourse, rule=main_ba_buy_rule)
                b.main_ba_sell = pyo.Var(self.model.T_fix_recourse, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
                def main_ba_sell_rule(m, s, t):
                    return b.main_ba_sell[t] == b.ba_sell[s,t]
                b.main_ba_sell_constraint = pyo.Constraint(self.model.S, self.model.T_fix_recourse, rule=main_ba_sell_rule)
        self.model.dayaheadBlocks = pyo.Block(self.model.dayaheads, rule=dayaheadBlock_rule)

        def linkBlock_rule(b, lin): # Create a block for each link to handle conversions between carriers
            link            = self.rfp.get_component(lin)
            b._name         = link.name
            b.rate          = link.parameters.get("rate", 1)
            assert b.rate > 0, f"Link {b._name} has non-positive conversion rate, which is not supported in the current model formulation."
            b.capacity      = link.parameters.get('capacity', np.inf)
            b.ec            = link.parameters.get("electricity_consumption", 0) # Electricity consumption rate
            b.carrier_in    = str(link.parameters["consumes"])
            b.carrier_out   = str(link.parameters["produces"])
            b.in_capacity   = b.capacity / b.rate
            b.max_ramp_up   = link.parameters.get('max_ramp_up', 1) * b.in_capacity
            b.max_ramp_down = link.parameters.get('max_ramp_down', 1) * b.in_capacity

            b.min_load      = link.parameters.get('min_load', 0) * self.inflexible # If inflexible, we set min_load to the given value. If flexible, we set min_load to 0, which means that the flow can be reduced to 0 if desired.
            b.in_flow       = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(b.min_load*b.in_capacity, b.in_capacity))
            b.out_flow      = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(b.min_load*b.capacity, b.capacity))

            def conversion_rule(m, s, t):
                return b.out_flow[s,t] == b.rate * b.in_flow[s,t]
            b.conversion_constraint = pyo.Constraint(self.model.S, self.model.T, rule=conversion_rule)
            if self.inflexible and bool(link.parameters.get('efficiency_curve', 0)):
                ec_rule = lambda m, s, t, segment: pyo.Constraint.Skip
                b.max_electricity_consumption = None
                slope = None
                if b._name == "Electrolyzer":
                    # Implement piecewise linear efficiency curve.
                    data = pd.read_json("setup_files/" + link.parameters['efficiency_curve'], orient="index")
                    slope = np.asarray(data.loc['a'].values[0])
                    intercept = np.asarray(data.loc['b'].values[0])
                    # Slope and intercept relate y = b * Capacity + a * x, where x is power (MW) and y is mass outflow (kg/h)
                    b.max_electricity_consumption = b.capacity / (slope[-1] + intercept[-1]) * 1000
                    def ec_rule(m, s, t, segment):
                        return b.out_flow[s, t] <= (slope[segment] * b.elec_cons[s, t] + intercept[segment] * b.max_electricity_consumption) / 1000
                elif b._name == "Haber Bosch Plant":
                    # Implement piecewise linear efficiency curve.
                    data = pd.read_excel("setup_files/" + link.parameters['efficiency_curve'], sheet_name="Piecewise")
                    data = data.dropna(how="all")
                    slope = np.asarray(data['a'].values)
                    intercept = np.asarray(data['b'].values)
                    b.max_electrical_consumption = b.capacity * (intercept[-1] + slope[-1])
                    def ec_rule(m, s, t, segment):
                        return b.elec_cons[s, t] >= slope[segment] * b.out_flow[s, t] + intercept[segment] * b.capacity
                b.segments = pyo.RangeSet(0, len(slope)-1)
                b.elec_cons = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.max_electricity_consumption))
                b.ec_constraints = pyo.Constraint(self.model.S, self.model.T, b.segments, rule=ec_rule)
            elif b.ec > 0:
                b.elec_cons = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity * b.ec))
                def ec_rule(m, s, t):
                    return b.elec_cons[s,t] == b.ec * b.out_flow[s,t]
                b.ec_constraint = pyo.Constraint(self.model.S, self.model.T, rule=ec_rule)
            if self.model_type == "non_recourse flows":
                b.main_in_flow = pyo.Var(self.model.T_fix_recourse, domain=pyo.NonNegativeReals, bounds=(0, b.in_capacity))
                def main_in_flow_rule(m, s, t):
                    return b.main_in_flow[t] == b.in_flow[s,t]
                b.main_in_flow_constraint = pyo.Constraint(self.model.S, self.model.T_fix_recourse, rule=main_in_flow_rule)
        self.model.linkBlocks = pyo.Block(self.model.links, rule=linkBlock_rule)

        def offtakerBlock_rule(b, offt): # Create a block for each offtaker to handle consumption
            offtaker        = self.rfp.get_component(offt)
            b._name         = offtaker.name
            b.carrier_in    = str(offtaker.parameters["consumes"])
            b.carrier_out   = str(offtaker.parameters["produces"])
            b.ec            = offtaker.parameters.get("electricity_consumption", 0) # Electricity consumption rate
            b.capacity      = offtaker.parameters.get('capacity')
            b.contracts     = pyo.Set(initialize=[cont.name for cont in offtaker.contracts])
            b.in_flow       = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            if b.ec > 0:
                b.elec_cons = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity * b.ec))
                def ec_rule(m, s, t):
                    return b.elec_cons[s,t] == b.ec * b.in_flow[s,t]
                b.ec_constraint = pyo.Constraint(self.model.S, self.model.T, rule=ec_rule)
            if self.model_type == "non_recourse flows":
                b.main_in_flow = pyo.Var(self.model.T_fix_recourse, domain=pyo.NonNegativeReals, bounds=(0, b.capacity/b.rate))
                def main_in_flow_rule(m, s, t):
                    return b.main_in_flow[t] == b.in_flow[s,t]
                b.main_in_flow_constraint = pyo.Constraint(self.model.S, self.model.T_fix_recourse, rule=main_in_flow_rule)
        self.model.offtakerBlocks = pyo.Block(self.model.offtakers, rule=offtakerBlock_rule)

        def contractBlock_rule(b, cont):
            contract        = self.rfp.get_contract(cont)
            b._name         = cont
            b.carrier_in    = contract.parameters.get("resource")
            b.volume        = contract.parameters.get("volume")
            b.price         = contract.parameters.get("price")
            b.penalty       = contract.parameters.get("penalty")
            b.offtaker      = contract.offtaker
            b.offtaker_capacity     = self.rfp.get_component(b.offtaker).parameters.get('capacity', 1e9)
            b.is_spot_contract      = bool(contract.parameters.get("spot_contract", 0))
            b.target_frequency      = contract.parameters.get("target_frequency", None)
            b.shipment_frequency    = contract.parameters.get("shipment_frequency", None)

            """ Physical flow of product to contract: """ 
            b.shipment = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, min(b.volume, b.offtaker_capacity)))
            if b.is_spot_contract == False:
                # Bookkeeping of contract status and whether obligations are met.
                b.contract_status = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.volume))
                b.contract_shortfall = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.volume))
                b.contract_slack = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.volume)) # Slack variable. Excess shipments are not awarded.
            if self.model_type == "non_recourse flows":
                b.main_shipment = pyo.Var(self.model.T_fix_recourse, domain=pyo.NonNegativeReals, bounds=(0, min(b.volume, b.offtaker_capacity)))
                def main_shipment_rule(m, s, t):
                    return b.main_shipment[t] == b.shipment[s,t]
                b.main_shipment_constraint = pyo.Constraint(self.model.S, self.model.T_fix_recourse, rule=main_shipment_rule)
        self.model.contractBlocks = pyo.Block(self.model.contracts, rule=contractBlock_rule)

        self.model.allBlocks = {**self.model.linkBlocks, **self.model.storageBlocks, **self.model.ppaBlocks,
                                **self.model.dayaheadBlocks, **self.model.carrierBlocks, **self.model.contractBlocks,
                                **self.model.offtakerBlocks}
    
    def _build_concrete_instance(self, data=None):
        self.inst = self.model.create_instance(data=data)
        self.updated_constraints = []

        """ If we are guiding the model with planning targets for contracts, this logic should be added to the contractBlocks """
        if self.guideline == 'hourly_target':
            def hourly_target_rule(inst, s, t): # Fix hourly production of ammonia:
                return inst.linkBlocks["Haber Bosch Plant"].out_flow[s,t] == inst.hourly_target
            self.inst.hourly_target_constraint = pyo.Constraint(self.inst.S, self.inst.T, rule=hourly_target_rule)
            self.updated_constraints += ['hourly_target_constraint']

        """ Rules that define the physical reality of the renewable fuel plant. """
        def carrier_balance_rule(inst, carr, s, t): # Ensure balance equations of plant energy carriers.
            b = inst.carrierBlocks[carr]
            for name, comp in self.rfp.get_components().items():
                if comp.parameters.get("produces") == b.type:
                    if comp.is_link or comp.is_storage or comp.is_ppa or comp.is_dayahead:
                        b._in[s,t].append(inst.allBlocks[name].out_flow[s,t])
                if comp.parameters.get("consumes") == b.type:
                    if comp.is_link or comp.is_storage or comp.is_offtaker:
                        b._out[s,t].append(inst.allBlocks[name].in_flow[s,t])
                if b.type == "plant_electricity":
                    if comp.parameters.get("electricity_consumption", 0) > 0:
                        if comp.is_link or comp.is_storage or comp.is_offtaker:
                            b._out[s,t].append(inst.allBlocks[name].elec_cons[s,t]) # Using a link can consume electricity.
            return sum(b._in[s,t]) == sum(b._out[s,t]) # Carrier balance arcs
        self.inst.carrier_balance_constraint = pyo.Constraint(self.inst.carriers, self.inst.S, self.inst.T, rule=carrier_balance_rule)

        def ppa_procurement_rule(inst, ppa, s, t): # Constrain supply flows from PPAs.
            b = inst.ppaBlocks[ppa]
            return b.out_flow[s,t] == inst.supplier_cf[ppa, s, t] * b.capacity
        self.inst.ppa_procurement_constraint = pyo.Constraint(self.inst.ppas, self.inst.S, self.inst.T, rule=ppa_procurement_rule)

        def soc_rule(inst, stor, s, t): # Define intertemporal SOC logic
            b = inst.storageBlocks[stor]
            if t == 0: # The initial SOC is externally given.
                return b.soc[s,0] == inst.init_soc[stor] + b.in_flow[s,0] - b.out_flow[s,0]
            else:
                return b.soc[s,t] == b.soc[s,t-1] + b.in_flow[s,t] - b.out_flow[s,t]
        self.inst.soc_constraint = pyo.Constraint(self.inst.storages, self.inst.S, self.inst.T, rule=soc_rule)

        def offtake_rule(inst, offt, s, t): # Ensure that offtake is matched to sale flows and that offtake stream does not violate capacity.
            b = inst.offtakerBlocks[offt]
            return b.in_flow[s,t] == sum(inst.contractBlocks[cont].shipment[s,t] for cont in b.contracts)
        self.inst.offtake_constraint = pyo.Constraint(self.inst.offtakers, self.inst.S, self.inst.T, rule=offtake_rule)

        if self.inflexible: # Link ramping constraints.
            self.inst.ramping_constraints = []
            def ramp_up_rule(inst, link, s, t):
                b = inst.linkBlocks[link]
                if b.max_ramp_up >= b.in_capacity: # If max ramp up is larger than capacity, then we can ramp up from zero to full in one step, so we don't need to add a constraint.
                    return pyo.Constraint.Skip
                else:
                    prev_setpoint = b.in_flow[s,t-1] if t > 0 else self.inst.prev_link_setpoints[link] * b.in_capacity
                    return b.in_flow[s,t] - prev_setpoint <= b.max_ramp_up
            def ramp_down_rule(inst, link, s, t):
                b = inst.linkBlocks[link]
                if b.max_ramp_down >= b.in_capacity: # If max ramp down is larger than capacity, then we can ramp down from full to zero in one step, so we don't need to add a constraint.
                    return pyo.Constraint.Skip
                else:
                    prev_setpoint = b.in_flow[s,t-1] if t > 0 else self.inst.prev_link_setpoints[link] * b.in_capacity
                    return prev_setpoint - b.in_flow[s,t] <= b.max_ramp_down
            self.inst.ramp_up_constraint = pyo.Constraint(self.inst.links, self.inst.S, self.inst.T, rule=ramp_up_rule)
            self.inst.ramp_down_constraint = pyo.Constraint(self.inst.links, self.inst.S, self.inst.T, rule=ramp_down_rule)
            self.updated_constraints += ["ramp_up_constraint", "ramp_down_constraint"]

        """ Rules that keeps track of delivery to contracts. """
        def contract_shipment_rule(inst, cont, s, t): # Ensure that the shipments happen only at shipment time.
            b = inst.contractBlocks[cont]
            if self._is_shipment_time(inst, b, t):
                return pyo.Constraint.Skip # Then don't constrain the shipment more than its existing bounds.
            else:
                return b.shipment[s,t] == 0 # Otherwise there cannot be any sales for this contract for this hour.
        self.inst.shipment_constraint = pyo.Constraint(self.inst.contracts, self.inst.S, self.inst.T, rule=contract_shipment_rule)

        def contract_status_rule(inst, cont, s, t): # The shipments we gets added to the contract_status.
            b = inst.contractBlocks[cont]
            if b.is_spot_contract:
                return pyo.Constraint.Skip
            else:
                if self._is_target_time(inst, b, t-1):
                    return b.contract_status[s,t] == b.shipment[s,t] # If we had deadline in the previous hour, then the status is reset.
                else:
                    prev_status = inst.init_contract_status[cont] if t == 0 else b.contract_status[s,t-1]
                    return b.contract_status[s,t] == prev_status + b.shipment[s,t] # Otherwise we increment by shipment size.
        self.inst.status_constraint = pyo.Constraint(self.inst.contracts, self.inst.S, self.inst.T, rule=contract_status_rule)

        def contract_shortfall_rule(inst, cont, s, t): # At contract delivery time, calculate shortfall.
            b = inst.contractBlocks[cont]
            if b.is_spot_contract:
                return pyo.Constraint.Skip
            else:
                if self._is_target_time(inst, b, t):
                    return b.contract_shortfall[s,t] - b.contract_slack[s,t] == b.volume - b.contract_status[s,t] # At contract deadline we can have non-zero contract_shortfall.
                else:
                    return b.contract_shortfall[s,t] + b.contract_slack[s,t] == 0 # Otherwise there cannot be any shortfall.
        self.inst.shortfall_constraint = pyo.Constraint(self.inst.contracts, self.inst.S, self.inst.T, rule=contract_shortfall_rule)

        if self.model_type == "recourse DA":
            # & Ensure that the buy volumes are non-increasing with price.
            # Precompute sorted scenario order for each time t
            sorted_scenarios = { 
                t: [s for price, s in sorted(
                    [(pyo.value(self.inst.electricity_price[s, t]), s) for s in self.inst.S]
                )]
                for t in self.inst.T if pyo.value(self.inst.fixed_da[t]) == 0
            }

            ordering_pairs = [] # Contains all relevant sets of (scenario[n], scenario[n+1], t)
            for t, scenario_order_t in sorted_scenarios.items():
                for n in range(len(scenario_order_t) - 1):
                    ordering_pairs.append((scenario_order_t[n], scenario_order_t[n+1], t))
            # Convert to Pyomo Set
            self.inst.ORDERING_PAIRS = pyo.Set(initialize=ordering_pairs, dimen=3)
            
            def bid_ordering_rule(inst, prev_s, s, t):
                return inst.dayaheadBlocks["ElectricitySpot"].da_buy[prev_s, t] >= inst.dayaheadBlocks["ElectricitySpot"].da_buy[s, t]
            self.inst.bid_ordering_constraint = pyo.Constraint(self.inst.ORDERING_PAIRS, rule=bid_ordering_rule)
            self.updated_constraints += ["bid_ordering_constraint"]

            def fix_da_rule(inst, da, s, t):
                if pyo.value(inst.fixed_da[t]) == 1:
                    return inst.dayaheadBlocks[da].da_buy[s,t] == inst.cleared_power[t]
                else:
                    return pyo.Constraint.Skip
            self.inst.fixed_da_constraint = pyo.Constraint(self.inst.dayaheads, self.inst.S, self.inst.T, rule=fix_da_rule)
        else:
            def fix_da_rule(inst, da, t):
                if pyo.value(inst.fixed_da[t]) == 1:
                    return inst.dayaheadBlocks[da].da_buy[t] == inst.cleared_power[t]
                else:
                    return pyo.Constraint.Skip
            self.inst.fixed_da_constraint = pyo.Constraint(self.inst.dayaheads, self.inst.T, rule=fix_da_rule)
        
        self._set_objective()

        self.inst.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        self.updated_constraints += ["ppa_procurement_constraint", "soc_constraint", "status_constraint", "shipment_constraint", "shortfall_constraint"]
        self.updated_constraints += ["fixed_da_constraint"]

    def _get_electricity_objective_cost(self, inst):
        # When out_flow is positive, we are buying electricity; negative, we are selling.
        # Balancing is negatively rewarded, we do not have actual balancing/intraday prices.
        balancing_cost = sum(sum(inst.dayaheadBlocks[dayahead].ba_buy[s,t] * (1.3 if pyo.value(inst.electricity_price[s,t]) > 0 else 0.7) * inst.electricity_price[s,t] -
                                inst.dayaheadBlocks[dayahead].ba_sell[s,t] * (1.3 if pyo.value(inst.electricity_price[s,t]) < 0 else 0.7) * inst.electricity_price[s,t]
                                for t in inst.T for dayahead in inst.dayaheads) * inst.weights[s] for s in inst.S)
        if self.model_type == "recourse DA":
            dayahead_cost = sum(sum(inst.dayaheadBlocks[dayahead].da_buy[s,t] * inst.electricity_price[s,t] 
                                    for t in inst.T for dayahead in inst.dayaheads) * inst.weights[s] for s in inst.S)
        else:
            dayahead_cost = sum(sum(inst.dayaheadBlocks[dayahead].da_buy[t] * inst.electricity_price[s,t]
                                    for t in inst.T for dayahead in inst.dayaheads) * inst.weights[s] for s in inst.S)
        return balancing_cost + dayahead_cost

    def _cashflow_rule(self, inst):
        """ Revenues of the RFP (contract payments happen when shipments happen) """
        revenue = sum(sum(b.shipment[s,t] * b.price for name, b in inst.contractBlocks.items() for t in inst.T) * inst.weights[s] for s in inst.S)

        """ Costs of the RFP (PPA costs not included as they are exogenously fixed) """
        costs = self._get_electricity_objective_cost(inst)

        for cont in inst.contracts: # Penalties of not meeting contract obligations:
            b = inst.contractBlocks[cont]
            if b.is_spot_contract == False:
                costs += sum(sum(b.contract_shortfall[s,t] * b.penalty for t in inst.T) * inst.weights[s] for s in inst.S)
        
        """ Maximize profits """
        return revenue - costs

    def _set_objective(self):        
        def production_value_rule(inst):
            production_value = sum(sum(inst.linkBlocks[link].out_flow[s,t] * inst.production_value[link] for link in inst.links for t in inst.T) * inst.weights[s] for s in inst.S)
            remove_shipment_revenue_incentive = sum(sum(b.shipment[s,t] * (0.95 * b.price) for name, b in inst.contractBlocks.items() for t in inst.T) * inst.weights[s] for s in inst.S)
            return production_value - remove_shipment_revenue_incentive
        
        def state_value_rule(inst):
            storage_value = sum(sum(b.soc[s,self.decision_horizon-1] * inst.storage_value[name] for name, b in inst.storageBlocks.items()) * inst.weights[s] for s in inst.S)
            contract_value = sum(sum(b.contract_status[s,self.decision_horizon-1] * inst.contract_value[name] for name, b in inst.contractBlocks.items() if b.is_spot_contract == False) * inst.weights[s] for s in inst.S)
            return storage_value + contract_value

        def objective_rule(inst):
            obj = self._cashflow_rule(inst)
            if self.guideline == 'production_value':
                obj += production_value_rule(inst)
            if self.objective_logic == 'value_maximization':
                obj += state_value_rule(inst)
            return obj

        self.inst.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    def _calculate_expected_el_revenue(self):
        inst = self.inst
        self.decision_results.exp_balancing_buy_cost     = sum(sum(pyo.value(inst.dayaheadBlocks[dayahead].ba_buy[s,t]) * (1.3 if pyo.value(inst.electricity_price[s,t]) > 0 else 0.7) * pyo.value(inst.electricity_price[s,t])
                                        for t in inst.T_fix_recourse for dayahead in inst.dayaheads) * pyo.value(inst.weights[s]) for s in inst.S)
        self.decision_results.exp_balancing_sell_revenue = sum(sum(pyo.value(inst.dayaheadBlocks[dayahead].ba_sell[s,t]) * (1.3 if pyo.value(inst.electricity_price[s,t]) < 0 else 0.7) * pyo.value(inst.electricity_price[s,t])
                                        for t in inst.T_fix_recourse for dayahead in inst.dayaheads) * pyo.value(inst.weights[s]) for s in inst.S)
        if self.model_type == "recourse DA":
            self.decision_results.exp_dayahead_cost      = sum(sum(pyo.value(inst.dayaheadBlocks[dayahead].da_buy[s,t]) * pyo.value(inst.electricity_price[s,t])
                                        for t in inst.T_fix_recourse for dayahead in inst.dayaheads) * pyo.value(inst.weights[s]) for s in inst.S)
        else:
            self.decision_results.exp_dayahead_cost      = sum(sum(pyo.value(inst.dayaheadBlocks[dayahead].da_buy[t]) * pyo.value(inst.electricity_price[s,t])
                                        for t in inst.T_fix_recourse for dayahead in inst.dayaheads) * pyo.value(inst.weights[s]) for s in inst.S)
        self.decision_results.exp_el_revenue = self.decision_results.exp_balancing_sell_revenue - self.decision_results.exp_balancing_buy_cost - self.decision_results.exp_dayahead_cost

    def _save_solution(self):
        """ Save decision results, which are non-recourse, and planning results, some of which are recourse decisions. """
        self._calculate_expected_el_revenue()

        if self.n_scenarios == 1 or self.model_type == "non-recourse flows":
            time_slices = [self.inst.T_fix_recourse]
            result_objects = [self.decision_results]

            for t_slice, res_object in zip(time_slices, result_objects):
                # Results needed for rolling horizon:
                res_object.final_soc          = {name : pyo.value(self.inst.storageBlocks[name].soc[0, len(t_slice)-1]) for name in self.inst.storages}
                res_object.final_contract_status = {b._name : pyo.value(b.contract_status[0, len(t_slice)-1]) for _, b in self.inst.contractBlocks.items() if b.is_spot_contract == False}

                # Save flow and soc results as well for plotting purposes:
                res_object.spot_power      = np.asarray([pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].out_flow[0, t]) for t in t_slice])
                res_object.da_buy          = np.asarray([pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].da_buy[t]) for t in t_slice])
                res_object.ba_buy          = np.asarray([pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].ba_buy[0, t]) for t in t_slice])
                res_object.ba_sell         = np.asarray([pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].ba_sell[0, t]) for t in t_slice])
                res_object.ppa_power       = {name : np.asarray([pyo.value(self.inst.ppaBlocks[name].out_flow[0, t]) for t in t_slice]) for name in self.inst.ppas}
                res_object.ppa_costs       = sum(pyo.value(b.out_flow[0, t]) * pyo.value(b.price) for _, b in self.inst.ppaBlocks.items() for t in t_slice)
                res_object.storage_soc     = {name : np.asarray([pyo.value(self.inst.storageBlocks[name].soc[0, t]) for t in t_slice]) for name in self.inst.storages}
                res_object.storage_inflow  = {name : np.asarray([pyo.value(self.inst.storageBlocks[name].in_flow[0, t]) for t in t_slice]) for name in self.inst.storages}
                res_object.storage_outflow = {name : np.asarray([pyo.value(self.inst.storageBlocks[name].out_flow[0, t]) for t in t_slice]) for name in self.inst.storages}
                res_object.link_production = {name : np.asarray([pyo.value(self.inst.linkBlocks[name].out_flow[0, t]) for t in t_slice]) for name in self.inst.links}
                res_object.power_consumption = np.asarray([pyo.value(self.inst.linkBlocks['Grid Connection Point'].in_flow[0, t]) for t in t_slice])

                # Contractually related results:
                res_object.shipments          = {cont : np.asarray([pyo.value(self.inst.contractBlocks[cont].shipment[0, t]) for t in t_slice])
                                                for cont in self.inst.contracts}
                res_object.delivered_revenue  = {cont : sum(pyo.value(b.shipment[0, t]) * pyo.value(b.price) for t in t_slice) for cont, b in self.inst.contractBlocks.items()}
                res_object.contract_status = {b._name : np.asarray([pyo.value(b.contract_status[0, t]) for t in t_slice])
                                            for _, b in self.inst.contractBlocks.items() if b.is_spot_contract == False}
                res_object.contract_penalty = {b._name : np.asarray([pyo.value(b.contract_shortfall[0, t]) * pyo.value(b.penalty) for t in t_slice])
                                            for _, b in self.inst.contractBlocks.items() if b.is_spot_contract == False}
                
                # Objective values for the time slices:
                res_object.cash_flow       = np.sum(list(res_object.delivered_revenue.values())) + res_object.exp_el_revenue - np.sum(list(res_object.contract_penalty.values()))
                res_object.objective_value = res_object.cash_flow
                
                if self.guideline == 'production_value':
                    res_object.objective_value += sum(pyo.value(self.inst.linkBlocks[link].out_flow[0, t]) * pyo.value(self.inst.production_value[link]) for link in self.inst.links for t in t_slice)
                
                if self.objective_logic == 'value_maximization':
                    storage_value = sum(pyo.value(b.soc[0, self.decision_horizon-1]) * pyo.value(self.inst.storage_value[name]) for name, b in self.inst.storageBlocks.items())
                    contract_value = sum(pyo.value(b.contract_status[0, self.decision_horizon-1]) * pyo.value(self.inst.contract_value[name]) for name, b in self.inst.contractBlocks.items() if b.is_spot_contract == False)
                    res_object.objective_value += storage_value + contract_value

    def get_da_volumes(self):
        if self.model_type == "recourse DA":
            dayahead_buy = [[pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].da_buy[s, t]) for t in self.inst.T_fix_dayahead] for s in self.inst.S] 
        else:
            dayahead_buy = [pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].da_buy[t]) for t in self.inst.T_fix_dayahead]
        return dayahead_buy

    def get_actions(self):
        """ Only the decisions made within the decision horizon are non-recourse. """
        if self.model_type == "non-recourse flows":
            if self.status == TerminationCondition.optimal:
                self.decision_results.electricity_purchase = [pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].out_flow[0,t]) for t in self.inst.T_fix_recourse]
                self.decision_results.hydrogen_production  = [pyo.value(self.inst.linkBlocks['Electrolyzer'].out_flow[0,t])         for t in self.inst.T_fix_recourse]
                self.decision_results.ammonia_production   = [pyo.value(self.inst.linkBlocks['Haber Bosch Plant'].out_flow[0,t])    for t in self.inst.T_fix_recourse]
                self.decision_results.hydrogen1_shipment = [pyo.value(self.inst.contractBlocks['Hydrogen1'].shipment[0,t])      for t in self.inst.T_fix_recourse]
                self.decision_results.ammonia1_shipment = [pyo.value(self.inst.contractBlocks['Ammonia1'].shipment[0,t])      for t in self.inst.T_fix_recourse]
                self.decision_results.ammonia_spot_shipment = [pyo.value(self.inst.contractBlocks['AmmoniaSpot'].shipment[0,t])      for t in self.inst.T_fix_recourse]
                df = pd.DataFrame(index=self.inst.T_fix_recourse, data = {
                    'electricity_purchase' : self.decision_results.electricity_purchase,
                    'hydrogen_production'   : self.decision_results.hydrogen_production,
                    'ammonia_production'   : self.decision_results.ammonia_production,
                    'hydrogen1_shipment' : self.decision_results.hydrogen1_shipment,
                    'ammonia1_shipment' : self.decision_results.ammonia1_shipment,
                    'ammonia_spot_shipment' : self.decision_results.ammonia_spot_shipment,
                    })
                self.decision_results.dayahead_buy = [pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].da_buy[t]) for t in self.inst.T_fix_recourse]
                self.decision_results.balancing_buy = [pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].ba_buy[0,t]) for t in self.inst.T_fix_recourse]
                self.decision_results.balancing_sell = [pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].ba_sell[0,t]) for t in self.inst.T_fix_recourse]
                df["electricity_purchase"] = self.decision_results.dayahead_buy
                df['balancing_buy'] = self.decision_results.balancing_buy
                df['balancing_sell'] = self.decision_results.balancing_sell
                return df
            else:
                return None

