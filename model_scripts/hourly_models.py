from common_scripts.RFP_initialization import RenewableFuelPlant #, create_rfp
from common_scripts import expando
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
import numpy as np
import pandas as pd


class HourlyDeterministicLPModel:
    """ --- Hardcoded exceptions to general formulation:
        plant_electricity:      The energy carrier for electricity within the RFP, which is impacted by elec_cons from units.
        'cmethod' get_actions:  Calls a hard-coded subset of decision variable results.
        
        --- Options to avoid hardcoded implementations:
        plant_electricity:      Should construct a more detail-rich carrierBlock information set. Then it can be a bool like spot contracts are.
                                Or allow for multiple in and out streams from units.
        'cmethod' get_actions:  Should maybe be more flexible or done on upper level - to accommodate various agents.

        --- class_variables:
        guideline_options and frequency_options: Good and clear as is.
    """

    frequency_options = ("hourly", "daily", "monthly", "yearly", "planning_horizon", None)
    guideline_options = ("production_value", "hourly_target", None)

    def __init__(self,
                 rfp: RenewableFuelPlant,
                 planning_horizon: int = 4*24,
                 decision_horizon: int = 24,
                 solver: str = 'scip',
                 allow_spot_buy: bool = True,
                 guideline: str|None = None,
                 objective_logic: str|None = None,
                 documentation: bool = False,
                 **kwargs,
                 ):
        
        # Problem specific parameters:
        self.rfp              = rfp
        self.decision_horizon = decision_horizon
        self.planning_horizon = max(planning_horizon, self.decision_horizon)
        self.allow_spot_buy   = allow_spot_buy
        self.objective_logic  = objective_logic
        self.documentation    = documentation # If true, the model will store extra information useful for documentation of results.

        assert guideline in self.guideline_options, "f{guideline} not in guideline options: {self.guideline_options}"
        self.guideline = guideline

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

    def initialize_model(self):
        # Initialize the optimization model
        self.model = pyo.AbstractModel()
        self._build_abstract_model()
        if self.uses_persistent_solver:
            self._build_concrete_instance() # Creates self.inst
            self.solver.set_instance(self.inst)

    def _build_abstract_model(self):
        # Model Time Sets:
        self.model.T    = pyo.RangeSet(0, self.planning_horizon - 1)  # Time steps
        self.model.T_r  = pyo.RangeSet(0, self.decision_horizon - 1) # Used for results processing

        # Model Set definitions
        self.model.carriers     = pyo.Set(initialize=[name for name in self.rfp.get_carriers().keys()])
        self.model.storages     = pyo.Set(initialize=[name for name in self.rfp.get_storages().keys()])
        self.model.ppas         = pyo.Set(initialize=[name for name in self.rfp.get_ppas().keys()])
        self.model.dayaheads    = pyo.Set(initialize=[name for name in self.rfp.get_dayaheads().keys()])
        self.model.links        = pyo.Set(initialize=[name for name in self.rfp.get_links().keys()])
        self.model.offtakers    = pyo.Set(initialize=[name for name in self.rfp.get_offtakers().keys()])
        self.model.contracts    = pyo.Set(initialize=[name for name in self.rfp.get_contracts().keys()])

        # Mutable model parameters:
        self.model.T_datetime   = pyo.Param(self.model.T, within=pyo.Any, initialize=pd.date_range(start=0, end=self.planning_horizon - 1, freq='h'), mutable=True)
        self.model.init_soc           = pyo.Param(self.model.storages, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.supplier_cf        = pyo.Param(self.model.ppas, self.model.T, within=pyo.NonNegativeReals, default=1, mutable=True)
        self.model.init_contract_status         = pyo.Param(self.model.contracts, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.electricity_price            = pyo.Param(self.model.T, within=pyo.Reals, default=50, mutable=True)
        # self.model.offtaker_availability        = pyo.Param(self.model.offtakers, self.model.T, within=pyo.Binary, default=1, mutable=True)

        # Guideline related mutable parameters:
        if self.guideline == "production_value": # Specified value of outflow of links.
            self.model.production_value = pyo.Param(self.model.links, within=pyo.Reals, default=0, mutable=True)
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
            b._in = {t: [] for t in self.model.T}
            b._out = {t: [] for t in self.model.T}
        self.model.carrierBlocks = pyo.Block(self.model.carriers, rule=carrierBlock_rule)

        def storageBlock_rule(b, stor): # Create a block for each storage to handle charge/discharge and state of charge
            storage         = self.rfp.get_component(stor)
            b._name         = storage.name
            b.capacity      = storage.parameters["capacity"]
            b.ec            = storage.parameters.get("electricity_consumption", 0) # Electricity consumption rate
            b.carrier_in    = str(storage.parameters["consumes"])
            b.carrier_out   = str(storage.parameters["produces"])

            b.soc       = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            b.in_flow   = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            b.out_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            if b.ec > 0:
                b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity*b.ec))
                def ec_rule(m, t):
                    return b.elec_cons[t] == b.in_flow[t] * b.ec
                b.ec_constraint = pyo.Constraint(self.model.T, rule=ec_rule)
        self.model.storageBlocks = pyo.Block(self.model.storages, rule=storageBlock_rule)

        def ppaBlock_rule(b, ppa): # Create a block for each ppa to handle production.
            # A bit too complexly implemented, but allows for potential other structure than PPAs.
            ppa_        = self.rfp.get_ppa(ppa)
            b._name         = ppa
            b.carrier_in    = str(ppa_.parameters["consumes"])
            b.carrier_out   = str(ppa_.parameters["produces"])
            b.capacity      = ppa_.parameters.get('capacity')
            b.price         = ppa_.parameters.get('price')
            b.out_flow      = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
        self.model.ppaBlocks = pyo.Block(self.model.ppas, rule=ppaBlock_rule)

        def dayaheadBlock_rule(b, da):
            dayahead        = self.rfp.get_component(da)
            b._name         = da
            b.carrier_in    = str(dayahead.parameters["consumes"])
            b.carrier_out   = str(dayahead.parameters["produces"])
            b.capacity      = dayahead.parameters.get('capacity')
            # Power bought from the day-ahead market; negative if power is sold.
            b.out_flow = pyo.Var(self.model.T, domain=pyo.Reals, bounds=(-b.capacity, b.capacity * self.allow_spot_buy))
        self.model.dayaheadBlocks = pyo.Block(self.model.dayaheads, rule=dayaheadBlock_rule)

        def linkBlock_rule(b, lin): # Create a block for each link to handle conversions between carriers
            link            = self.rfp.get_component(lin)
            b._name         = link.name
            b.rate          = link.parameters.get("rate", 1)
            b.capacity      = link.parameters.get('capacity', np.inf)
            b.ec            = link.parameters.get("electricity_consumption", 0) # Electricity consumption rate
            b.carrier_in    = str(link.parameters["consumes"])
            b.carrier_out   = str(link.parameters["produces"])
            b.in_flow       = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity/b.rate))
            b.out_flow      = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            if b.ec > 0:
                b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity * b.ec))
                def ec_rule(m, t):
                    return b.elec_cons[t] == b.ec * b.out_flow[t]
                b.ec_constraint = pyo.Constraint(self.model.T, rule=ec_rule)
            def conversion_rule(m, t):
                return b.out_flow[t] == b.rate * b.in_flow[t]
            b.conversion_constraint = pyo.Constraint(self.model.T, rule=conversion_rule)
            # if self.guideline == 'hourly_target' and b._name == "Haber Bosch Plant":
            #     b.hourly_slack = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
        self.model.linkBlocks = pyo.Block(self.model.links, rule=linkBlock_rule)

        def offtakerBlock_rule(b, offt): # Create a block for each offtaker to handle consumption
            offtaker        = self.rfp.get_component(offt)
            b._name         = offtaker.name
            b.carrier_in    = str(offtaker.parameters["consumes"])
            b.carrier_out   = str(offtaker.parameters["produces"])
            b.ec            = offtaker.parameters.get("electricity_consumption", 0) # Electricity consumption rate
            b.capacity      = offtaker.parameters.get('capacity')
            b.contracts     = pyo.Set(initialize=[cont.name for cont in offtaker.contracts])
            b.in_flow       = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            if b.ec > 0:
                b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity * b.ec))
                def ec_rule(m, t):
                    return b.elec_cons[t] == b.ec * b.in_flow[t]
                b.ec_constraint = pyo.Constraint(self.model.T, rule=ec_rule)
        self.model.offtakerBlocks = pyo.Block(self.model.offtakers, rule=offtakerBlock_rule)

        def contractBlock_rule(b, cont):
            contract        = self.rfp.get_contract(cont)
            b._name         = cont
            b.carrier_in    = contract.parameters.get("resource")
            b.price         = contract.parameters.get("price")
            b.penalty       = contract.parameters.get("penalty")
            b.offtaker      = contract.offtaker
            b.offtaker_capacity     = self.rfp.get_component(b.offtaker).parameters.get('capacity', 1e9)
            b.is_spot_contract      = bool(contract.parameters.get("spot_contract", 0))
            b.target_frequency      = contract.parameters.get("target_frequency", None)
            b.shipment_frequency    = contract.parameters.get("shipment_frequency", None)

            if not(self.documentation):
                b.volume        = contract.parameters.get("volume")
                """ Physical flow of product to contract: """ 
                b.shipment = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, min(b.volume, b.offtaker_capacity)))
                if b.is_spot_contract == False:
                    # Bookkeeping of contract status and whether obligations are met.
                    b.contract_status = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.volume))
                    b.contract_shortfall = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.volume))
                    b.contract_slack = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.volume)) # Slack variable. Excess shipments are not awarded.
            else:
                b.volume = pyo.Var(domain=pyo.Reals)
                b.volume_constraint = pyo.Constraint(expr= b.volume == contract.parameters.get("volume"))
                """ Physical flow of product to contract: """ 
                b.shipment = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.offtaker_capacity))
                b.shipment_volume_constraint = pyo.Constraint(self.model.T, rule=lambda m, t: b.shipment[t] <= b.volume)
                if b.is_spot_contract == False:
                    # Bookkeeping of contract status and whether obligations are met.
                    b.contract_status = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
                    b.contract_shortfall = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
                    b.contract_slack = pyo.Var(self.model.T, domain=pyo.NonNegativeReals) # Slack variable. Excess shipments are not awarded.
                    b.contract_status_bounds = pyo.Constraint(self.model.T, rule=lambda m, t: b.contract_status[t] <= b.volume)
                    b.contract_shortfall_bounds = pyo.Constraint(self.model.T, rule=lambda m, t: b.contract_shortfall[t] <= b.volume)
                    b.contract_slack_bounds = pyo.Constraint(self.model.T, rule=lambda m, t: b.contract_slack[t] <= b.volume)
        self.model.contractBlocks = pyo.Block(self.model.contracts, rule=contractBlock_rule)

    def _build_concrete_instance(self, data=None):
        self.inst = self.model.create_instance(data=data)
        self.updated_constraints = []

        """ Helper methods to determine whether it is shipment or target time for contracts. """
        def _get_datetime_infos(inst, t):
            if t == -1:
                dt_t = inst.T_datetime[t+1].value - pd.Timedelta(1, 'h')
            else:
                dt_t = inst.T_datetime[t].value
            is_day_end = (dt_t.hour == 23)
            is_month_end = dt_t.is_month_end
            if isinstance(is_month_end, list):
                is_month_end = is_month_end[0]
            is_year_end = dt_t.is_year_end
            if isinstance(is_year_end, list):
                is_year_end = is_year_end[0]
            return is_day_end, is_month_end, is_year_end
        
        def _is_target_time(inst, b, t):
            assert (b.target_frequency in self.frequency_options), f"{b.target_frequency} for {b._name} is not in options.\nOptions are {self.frequency_options}."
            is_day_end, is_month_end, is_year_end = _get_datetime_infos(inst, t)
            is_planning_end = (t == (self.planning_horizon - 1))
            return bool((b.target_frequency == 'hourly') or                                  # If we have an hourly contract.
                        (b.target_frequency == 'daily'   and is_day_end) or                  # If we have a daily contract and it is end-of-day (EOD).
                        (b.target_frequency == 'monthly' and is_month_end and is_day_end) or # If we have a monthly contract and it is end-of-month and EOD.
                        (b.target_frequency == 'yearly'  and is_year_end and is_day_end) or  # If we have a yearly contract and it is end-of-year and EOD.
                        (b.target_frequency == 'planning_horizon' and is_planning_end))      # If we are constraining the problem on the planning horizon.
        
        def _is_shipment_time(inst, b, t):
            assert (b.shipment_frequency in self.frequency_options), f"{b.shipment_frequency} for {b._name} is not in options.\nOptions are {self.frequency_options}."
            is_day_end, is_month_end, is_year_end = _get_datetime_infos(inst, t)
            return bool((b.shipment_frequency == 'hourly') or                                   # If we have an hourly shipment.
                        (b.shipment_frequency == 'daily'   and is_day_end) or                   # If we have a daily shipment and it is end-of-day (EOD).
                        (b.shipment_frequency == 'monthly' and is_month_end and is_day_end) or  # If we have a monthly shipment and it is end-of-month and EOD.
                        (b.shipment_frequency == 'yearly'  and is_year_end and is_day_end))     # If we have a yearly shipment and it is end-of-year and EOD.

        """ If we are guiding the model with planning targets for contracts, this logic should be added to the contractBlocks """
        if self.guideline == 'hourly_target':
            def hourly_target_rule(inst, t): # Fix hourly production of ammonia:
                return inst.linkBlocks["Haber Bosch Plant"].out_flow[t] == inst.hourly_target
            self.inst.hourly_target_constraint = pyo.Constraint(self.inst.T, rule=hourly_target_rule)
            self.updated_constraints += ['hourly_target_constraint']

        """ Rules that define the physical reality of the renewable fuel plant. """
        def carrier_balance_rule(inst, carr, t): # Ensure balance equations of plant energy carriers.
            b = inst.carrierBlocks[carr]
            for name, comp in self.rfp.get_components().items():
                if comp.parameters.get("produces") == b.type:
                    if comp.is_link:
                        b._in[t].append(inst.linkBlocks[name].out_flow[t])
                    elif comp.is_storage:
                        b._in[t].append(inst.storageBlocks[name].out_flow[t])
                    elif comp.is_ppa:
                        b._in[t].append(inst.ppaBlocks[name].out_flow[t])
                    elif comp.is_dayahead:
                        b._in[t].append(inst.dayaheadBlocks[name].out_flow[t])
                if comp.parameters.get("consumes") == b.type:
                    if comp.is_link:
                        b._out[t].append(inst.linkBlocks[name].in_flow[t])
                    elif comp.is_storage:
                        b._out[t].append(inst.storageBlocks[name].in_flow[t])
                    elif comp.is_offtaker:
                        b._out[t].append(inst.offtakerBlocks[name].in_flow[t])
                if b.type == "plant_electricity":
                    if comp.parameters.get("electricity_consumption", 0) > 0:
                        if comp.is_link:
                            b._out[t].append(inst.linkBlocks[name].elec_cons[t]) # Using a link can consume electricity.
                        elif comp.is_storage:
                            b._out[t].append(inst.storageBlocks[name].elec_cons[t]) # Charging a storage can consume electricity.
                        elif comp.is_offtaker:
                            b._out[t].append(inst.offtakerBlocks[name].elec_cons[t]) # Using an offtaker can consume electricity.
            return sum(b._in[t]) == sum(b._out[t]) # Carrier balance arcs
        self.inst.carrier_balance_constraint = pyo.Constraint(self.inst.carriers, self.inst.T, rule=carrier_balance_rule)

        def ppa_procurement_rule(inst, ppa, t): # Constrain supply flows from PPAs.
            b = inst.ppaBlocks[ppa]
            return b.out_flow[t] == inst.supplier_cf[ppa, t] * b.capacity
        self.inst.ppa_procurement_constraint = pyo.Constraint(self.inst.ppas, self.inst.T, rule=ppa_procurement_rule)

        def soc_rule(inst, stor, t): # Define intertemporal SOC logic
            b = inst.storageBlocks[stor]
            if t == 0: # The initial SOC is externally given.
                return b.soc[0] == inst.init_soc[stor] + b.in_flow[0] - b.out_flow[0]
            else:
                return b.soc[t] == b.soc[t-1] + b.in_flow[t] - b.out_flow[t]
        self.inst.soc_constraint = pyo.Constraint(self.inst.storages, self.inst.T, rule=soc_rule)

        def offtake_rule(inst, offt, t): # Ensure that offtake is matched to sale flows and that offtake stream does not violate capacity.
            b = inst.offtakerBlocks[offt]
            return b.in_flow[t] == sum(inst.contractBlocks[cont].shipment[t] for cont in b.contracts)
        self.inst.offtake_constraint = pyo.Constraint(self.inst.offtakers, self.inst.T, rule=offtake_rule)

        """ Rules that keeps track of delivery to contracts. """
        def contract_shipment_rule(inst, cont, t): # Ensure that the shipments happen only at shipment time.
            b = inst.contractBlocks[cont]
            if _is_shipment_time(inst, b, t):
                return pyo.Constraint.Skip # Then don't constrain the shipment more than its existing bounds.
            else:
                return b.shipment[t] == 0 # Otherwise there cannot be any sales for this contract for this hour.
        self.inst.shipment_constraint = pyo.Constraint(self.inst.contracts, self.inst.T, rule=contract_shipment_rule)

        def contract_status_rule(inst, cont, t): # The shipments we gets added to the contract_status.
            b = inst.contractBlocks[cont]
            if b.is_spot_contract:
                return pyo.Constraint.Skip
            else:
                if _is_target_time(inst, b, t-1):
                    return b.contract_status[t] == b.shipment[t] # If we had deadline in the previous hour, then the status is reset.
                else:
                    prev_status = inst.init_contract_status[cont] if t == 0 else b.contract_status[t-1]
                    return b.contract_status[t] == prev_status + b.shipment[t] # Otherwise we increment by shipment size.
        self.inst.status_constraint = pyo.Constraint(self.inst.contracts, self.inst.T, rule=contract_status_rule)

        def contract_shortfall_rule(inst, cont, t): # At contract delivery time, calculate shortfall.
            b = inst.contractBlocks[cont]
            if b.is_spot_contract:
                return pyo.Constraint.Skip
            else:
                if _is_target_time(inst, b, t):
                    return b.contract_shortfall[t] - b.contract_slack[t] == b.volume - b.contract_status[t] # At contract deadline we can have non-zero contract_shortfall.
                else:
                    return b.contract_shortfall[t] + b.contract_slack[t] == 0 # Otherwise there cannot be any shortfall.
        self.inst.shortfall_constraint = pyo.Constraint(self.inst.contracts, self.inst.T, rule=contract_shortfall_rule)

        self._set_objective()

        self.inst.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        self.updated_constraints += ["ppa_procurement_constraint", "soc_constraint", "status_constraint", "shipment_constraint", "shortfall_constraint"]

    def _update_instance(self, data=None):
        if data is not None:
            data = data[None]
            for param_name, param_data in data.items():
                param = self.inst.component(param_name)
                if isinstance(param, pyo.Param):
                    param.store_values(param_data)
                else:
                    raise TypeError(f"{param_name} is not a Param component.")
            self._refresh_constraints()
            self.solver.set_objective(self.inst.objective)

    def _refresh_constraints(self):
        for name in self.updated_constraints:
            con = self.inst.component(name)
            if con.is_indexed():
                for i in con:
                    self.solver.remove_constraint(con[i])
                    self.solver.add_constraint(con[i])
            else:
                self.solver.remove_constraint(con)
                self.solver.add_constraint(con)

    def build_concrete_instance(self, data=None):
        if self.uses_persistent_solver:
            self._update_instance(data)
        else:
            self._build_concrete_instance(data)

    def _get_electricity_objective_cost(self, inst):
        # When out_flow is positive, we are buying electricity; negative, we are selling.
        # We assume now that there is only one price realization of day-ahead markets. Could be generalized if we also want to buy from neighbouring bidding zones.
        return sum(inst.dayaheadBlocks[dayahead].out_flow[t] * inst.electricity_price[t] for t in inst.T for dayahead in inst.dayaheads)

    def _set_objective(self):
        def cashflow_rule(inst):
            """ Revenues of the RFP (contract payments happen when shipments happen) """
            revenue = sum(b.shipment[t] * b.price for name, b in inst.contractBlocks.items() for t in inst.T)

            """ Costs of the RFP (PPA costs not included as they are exogenously fixed) """
            costs = self._get_electricity_objective_cost(inst)

            for cont in inst.contracts: # Penalties of not meeting contract obligations:
                b = inst.contractBlocks[cont]
                if b.is_spot_contract == False:
                    costs += sum(b.contract_shortfall[t] * b.penalty for t in inst.T)
            
            """ Maximize profits """
            return revenue - costs
        
        def production_value_rule(inst):
            production_value = sum(inst.linkBlocks[link].out_flow[t] * inst.production_value[link] for link in inst.links for t in inst.T)
            remove_shipment_revenue_incentive = sum(b.shipment[t] * (0.95 * b.price) for name, b in inst.contractBlocks.items() for t in inst.T)
            return production_value - remove_shipment_revenue_incentive
        
        def state_value_rule(inst):
            storage_value = sum(b.soc[self.decision_horizon-1] * inst.storage_value[name] for name, b in inst.storageBlocks.items())
            contract_value = sum(b.contract_status[self.decision_horizon-1] * inst.contract_value[name] for name, b in inst.contractBlocks.items() if b.is_spot_contract == False)
            return storage_value + contract_value
        
        def objective_rule(inst):
            obj = cashflow_rule(inst)
            if self.guideline == 'production_value':
                obj += production_value_rule(inst)
            if self.objective_logic == 'value_maximization':
                obj += state_value_rule(inst)
            # if self.guideline == 'hourly_target':
            #     obj -= 1e6 * sum(inst.linkBlocks["Haber Bosch Plant"].hourly_slack[t] for t in inst.T) # Penalize slack variables heavily.
            return obj

        self.inst.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    def run(self, verbose=False):
        if self.inst:
            self.solve_message = None
            if self.uses_persistent_solver:
                self.solve_message = self.solver.solve(tee=verbose)
            else:
                self.solve_message = self.solver.solve(self.inst, tee=verbose)
            self.status = self.solve_message['Solver'][0]['Termination condition']
            if self.status == TerminationCondition.optimal:
                self._save_solution()
            elif self.status == TerminationCondition.infeasibleOrUnbounded:
                print(Warning("Could not solve the problem, status: " + str(self.solve_message['Solver'][0]['Termination condition'])))
            else:
                print(Warning("Non-optimal LP, status: " + str(self.status)))
        else:
            raise("Initialize concrete instance of model with data before running.")
    
    def _calculate_expected_el_revenue(self):
        self.decision_results.exp_el_revenue = -sum(
            pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].out_flow[t]) * pyo.value(self.inst.electricity_price[t])
            for t in self.inst.T_r)
        self.planning_results.exp_el_revenue = -sum(
            pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].out_flow[t]) * pyo.value(self.inst.electricity_price[t]) 
            for t in self.inst.T)

    def _save_solution(self):
        """ Save decision results, which are non-recourse, and planning results, some of which are recourse decisions. """
        time_slices = [self.inst.T_r, self.inst.T]
        result_objects = [self.decision_results, self.planning_results]

        self._calculate_expected_el_revenue()

        for t_slice, res_object in zip(time_slices, result_objects):
            # Results needed for rolling horizon:
            res_object.final_soc          = {name : pyo.value(self.inst.storageBlocks[name].soc[len(t_slice)-1]) for name in self.inst.storages}
            res_object.final_contract_status = {b._name : pyo.value(b.contract_status[len(t_slice)-1]) for _, b in self.inst.contractBlocks.items() if b.is_spot_contract == False}

            # Save flow and soc results as well for plotting purposes:
            res_object.spot_power      = np.asarray([pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].out_flow[t]) for t in t_slice])
            res_object.da_buy          = np.asarray([pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].out_flow[t]) for t in t_slice])
            res_object.ppa_power       = {name : np.asarray([pyo.value(self.inst.ppaBlocks[name].out_flow[t]) for t in t_slice]) for name in self.inst.ppas}
            res_object.ppa_costs       = sum(pyo.value(b.out_flow[t]) * pyo.value(b.price) for _, b in self.inst.ppaBlocks.items() for t in t_slice)
            res_object.storage_soc     = {name : np.asarray([pyo.value(self.inst.storageBlocks[name].soc[t]) for t in t_slice]) for name in self.inst.storages}
            res_object.storage_inflow  = {name : np.asarray([pyo.value(self.inst.storageBlocks[name].in_flow[t]) for t in t_slice]) for name in self.inst.storages}
            res_object.storage_outflow = {name : np.asarray([pyo.value(self.inst.storageBlocks[name].out_flow[t]) for t in t_slice]) for name in self.inst.storages}
            res_object.link_production = {name : np.asarray([pyo.value(self.inst.linkBlocks[name].out_flow[t]) for t in t_slice]) for name in self.inst.links}
            res_object.power_consumption = np.asarray([pyo.value(self.inst.linkBlocks['Grid Connection Point'].in_flow[t]) for t in t_slice])

            # Contractually related results:
            res_object.shipments          = {cont : np.asarray([pyo.value(self.inst.contractBlocks[cont].shipment[t]) for t in t_slice])
                                             for cont in self.inst.contracts}
            res_object.delivered_revenue  = {cont : sum(pyo.value(b.shipment[t]) * pyo.value(b.price) for t in t_slice) for cont, b in self.inst.contractBlocks.items()}
            res_object.contract_status = {b._name : np.asarray([pyo.value(b.contract_status[t]) for t in t_slice])
                                          for _, b in self.inst.contractBlocks.items() if b.is_spot_contract == False}
            res_object.contract_penalty = {b._name : np.asarray([pyo.value(b.contract_shortfall[t]) * pyo.value(b.penalty) for t in t_slice])
                                           for _, b in self.inst.contractBlocks.items() if b.is_spot_contract == False}
            
            # Objective values for the time slices:
            res_object.cash_flow       = sum(res_object.delivered_revenue.values()) - res_object.exp_el_revenue - sum(res_object.contract_penalty.values())
            res_object.objective_value = res_object.cash_flow
            if self.guideline == 'production_value':
                res_object.objective_value += sum(pyo.value(self.inst.linkBlocks[link].out_flow[t]) * pyo.value(self.inst.production_value[link]) for link in self.inst.links for t in t_slice)
            if self.objective_logic == 'value_maximization':
                storage_value = sum(pyo.value(b.soc[self.decision_horizon-1]) * pyo.value(self.inst.storage_value[name]) for name, b in self.inst.storageBlocks.items())
                contract_value = sum(pyo.value(b.contract_status[self.decision_horizon-1]) * pyo.value(self.inst.contract_value[name]) for name, b in self.inst.contractBlocks.items() if b.is_spot_contract == False)
                res_object.objective_value += storage_value + contract_value

    def get_objective(self):
        if hasattr(self, 'solve_message') == False:
            Warning("Model has not been solved yet, no objective to return.")
            return 0, True
        else:
            status = self.solve_message['Solver'][0]['Termination condition']
            if status == TerminationCondition.optimal:
                return self.inst.objective(), False # No truncation
            else:
                return 0, True # Truncate episode

    def get_actions(self):
        """ Only the decisions made within the decision horizon are non-recourse. """
        if self.status == TerminationCondition.optimal:
            self.decision_results.electricity_purchase = [pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].out_flow[t]) for t in self.inst.T_r]
            self.decision_results.hydrogen_production  = [pyo.value(self.inst.linkBlocks['Electrolyzer'].out_flow[t])         for t in self.inst.T_r]
            self.decision_results.ammonia_production   = [pyo.value(self.inst.linkBlocks['Haber Bosch Plant'].out_flow[t])    for t in self.inst.T_r]
            self.decision_results.hydrogen1_shipment = [pyo.value(self.inst.contractBlocks['Hydrogen1'].shipment[t])      for t in self.inst.T_r]
            self.decision_results.ammonia1_shipment = [pyo.value(self.inst.contractBlocks['Ammonia1'].shipment[t])      for t in self.inst.T_r]
            self.decision_results.ammonia_spot_shipment = [pyo.value(self.inst.contractBlocks['AmmoniaSpot'].shipment[t])      for t in self.inst.T_r]
            df = pd.DataFrame(index=self.inst.T_r, data = {
                'electricity_purchase' : self.decision_results.electricity_purchase,
                'hydrogen_production'   : self.decision_results.hydrogen_production,
                'ammonia_production'   : self.decision_results.ammonia_production,
                'hydrogen1_shipment' : self.decision_results.hydrogen1_shipment,
                'ammonia1_shipment' : self.decision_results.ammonia1_shipment,
                'ammonia_spot_shipment' : self.decision_results.ammonia_spot_shipment,
                })
            return df
        else:
            return None


class HourlyStochasticLPModel(HourlyDeterministicLPModel):
    """
    Simple stochastic considerations of different price scenarios, deterministic VRE considerations.

    This model differs from the deterministic model by considering different scenarios of VRE and price evolution.
    We still assume that the first 24 hours of VRE are certain. (As they are for the asset, but not for the system).
    Now instead of forecasting 1 scenario of prices, we simulate 'n_scenarios' scenarios of prices, which include potential paradigm
    shifts and stochasticities. """
    def __init__(self,
                 rfp: RenewableFuelPlant,
                 planning_horizon: int,
                 decision_horizon: int = 24,
                 solver: str = 'scip',
                 allow_spot_buy: bool = True,
                 guideline: str|None = None,
                 n_scenarios: int = 3,
                 **kwargs,
                 ):
        super().__init__(rfp, planning_horizon, decision_horizon, solver, allow_spot_buy, guideline)
        self.n_scenarios = n_scenarios
    
    def initialize_model(self):
        # Initialize the optimization model
        self.model = pyo.AbstractModel()
        self.model.S = pyo.Set(initialize=range(self.n_scenarios)) # Set of price scenarios
        self.model.weights = pyo.Param(self.model.S, within=pyo.NonNegativeReals, default=1/self.n_scenarios, mutable=True)
        self._build_abstract_model()
        if hasattr(self.model, 'electricity_price'):
            self.model.del_component('electricity_price')
        self.model.electricity_price = pyo.Param(self.model.S, self.model.T, within=pyo.Reals, default=50, mutable=True)
        if self.uses_persistent_solver:
            self._build_concrete_instance() # Creates self.inst
            self.solver.set_instance(self.inst)
    
    def _calculate_expected_el_revenue(self):
        self.decision_results.exp_el_revenue = -sum(
            pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].out_flow[t]) * pyo.value(self.inst.electricity_price[s,t]) * pyo.value(self.inst.weights[s])
            for t in self.inst.T_r for s in self.inst.S)
        self.planning_results.exp_el_revenue = -sum(
            pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].out_flow[t]) * pyo.value(self.inst.electricity_price[s,t]) * pyo.value(self.inst.weights[s])
            for t in self.inst.T for s in self.inst.S)

    def _get_electricity_objective_cost(self, inst):
        return sum(sum(inst.dayaheadBlocks[dayahead].out_flow[t] * inst.electricity_price[s,t] for t in inst.T for dayahead in inst.dayaheads)
                   * inst.weights[s] for s in inst.S)


class ShieldLPModel(HourlyDeterministicLPModel):
    """ Class which minimizes the L2 distance from exogenously decided actions, respecting the feasible space of the problem. """
    def __init__(self, env, rfp, planning_horizon = 24, decision_horizon = 24, solver = 'scip', allow_spot_buy = True, penalty_type="L1"):
        super().__init__(rfp=rfp,
                         planning_horizon=planning_horizon,
                         decision_horizon=decision_horizon,
                         solver=solver,
                         allow_spot_buy=allow_spot_buy,
                         guideline=None,
                         )
        self.penalty_type = penalty_type
        self.env = env

    def _build_abstract_model(self):
        self.model.actions = pyo.Set(initialize=[(name, t) for name in self.env.action_identity for t in range(self.planning_horizon)])
        self.model.chosen_actions = pyo.Param(self.model.actions, default=0, mutable=True)
        if self.penalty_type == "L1":
            self.model.action_violation = pyo.Var(self.model.actions, domain=pyo.NonNegativeReals)
        super()._build_abstract_model()
    
    def _build_concrete_instance(self, data=None):
        super()._build_concrete_instance(data)

        if self.penalty_type == "L1":
            self.inst.l1_constraint = pyo.ConstraintList()
            for (description, t) in self.inst.actions:
                blocktype, unit_name, var_name = str(description).split("-")
                # Find the block
                block = next(b for _, b in getattr(self.inst, blocktype + 'Blocks').items() if b._name == unit_name)

                expr = block.__getattribute__(var_name)[t] - self.inst.chosen_actions[(description, t)]
                # Add both constraints for L1 penalty
                self.inst.l1_constraint.add(self.inst.action_violation[(description, t)] >= expr)
                self.inst.l1_constraint.add(self.inst.action_violation[(description, t)] >= -expr)

    def _set_objective(self):
        def objective_rule_shield(inst):
            obj = 0
            for (description, t) in inst.actions:
                if self.penalty_type == "L1":
                    obj += inst.action_violation[(description,t)]
                else:
                    blocktype, unit_name, var_name = str(description).split("-")
                    block = None
                    # Identify the referenced block:
                    for _, b in inst.__getattribute__(blocktype + 'Blocks').items():
                        if b._name == unit_name:
                            block = b
                    expr = (block.__getattribute__(var_name)[t] - inst.chosen_actions[(description,t)])
                    obj += expr * expr # L2 penalty for deviating from actions
            return obj
        self.inst.objective = pyo.Objective(rule=objective_rule_shield, sense=pyo.minimize)


class HourlyRecourseModel(HourlyDeterministicLPModel):
    """ If we have cleared power setpoints for our day-ahead market clearing (testing a bidding curve model)
        We can make all the plant flow decisions here as recourse decisions. Balancing electricity is simply just punished at +/- 30%
        """
    def initialize_model(self):
        # Initialize the optimization model
        self.model = pyo.AbstractModel()
        self._build_abstract_model()
        self.model.cleared_power = pyo.Param(self.model.T, within=pyo.Reals, default=0, mutable=True)
        self.model.fixed_da = pyo.Param(self.model.T, within=pyo.Binary, default=0, mutable=True) # Whether we should fix the DA decision to cleared_power parameter.
        if hasattr(self.model, 'dayaheadBlocks'):
            self.model.del_component('dayaheadBlocks')
        def dayaheadBlock_rule(b, da):
            dayahead        = self.rfp.get_component(da)
            b._name         = da
            b.carrier_in    = str(dayahead.parameters["consumes"])
            b.carrier_out   = str(dayahead.parameters["produces"])
            b.capacity      = dayahead.parameters.get('capacity')
            # Power bought from the day-ahead market; negative if power is sold.
            b.out_flow = pyo.Var(self.model.T, domain=pyo.Reals, bounds=(-b.capacity, b.capacity * self.allow_spot_buy))
            b.da_buy = pyo.Var(self.model.T, domain=pyo.Reals, bounds=(-b.capacity, b.capacity * self.allow_spot_buy))
            b.ba_buy = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            b.ba_sell = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            def balancing_rule(m, t):
                return b.out_flow[t] == b.da_buy[t] + b.ba_buy[t] - b.ba_sell[t]
            b.balancing_constraint = pyo.Constraint(self.model.T, rule=balancing_rule)
        self.model.dayaheadBlocks = pyo.Block(self.model.dayaheads, rule=dayaheadBlock_rule)
        if self.uses_persistent_solver:
            self._build_concrete_instance() # Creates self.inst
            self.solver.set_instance(self.inst)
    
    def _build_concrete_instance(self, data=None):
        super()._build_concrete_instance(data)
        def fix_da_rule(inst, da, t):
            if pyo.value(inst.fixed_da[t]) == 1:
                return inst.dayaheadBlocks[da].da_buy[t] == inst.cleared_power[t]
            else:
                return pyo.Constraint.Skip
        self.inst.fixed_da_constraint = pyo.Constraint(self.inst.dayaheads, self.inst.T, rule=fix_da_rule)
        self.updated_constraints += ["fixed_da_constraint"]

    def _get_electricity_objective_cost(self, inst):
        # When out_flow is positive, we are buying electricity; negative, we are selling.
        # Balancing is negatively rewarded, we do not have actual balancing/intraday prices.
        return sum(inst.dayaheadBlocks[dayahead].da_buy[t] * inst.electricity_price[t] +
                   inst.dayaheadBlocks[dayahead].ba_buy[t] * 1.3 * inst.electricity_price[t] -
                   inst.dayaheadBlocks[dayahead].ba_sell[t] * 0.7 * inst.electricity_price[t]
                   for t in inst.T for dayahead in inst.dayaheads)
    
    def _calculate_expected_el_revenue(self):
        self.decision_results.exp_el_revenue = -sum((pyo.value(self.inst.dayaheadBlocks[dayahead].da_buy[t]) +
                                                    pyo.value(self.inst.dayaheadBlocks[dayahead].ba_buy[t]) * 1.3 -
                                                    pyo.value(self.inst.dayaheadBlocks[dayahead].ba_sell[t]) * 0.7) * pyo.value(self.inst.electricity_price[t])
                                                    for t in self.inst.T_r for dayahead in self.inst.dayaheads)
        self.planning_results.exp_el_revenue = -sum((pyo.value(self.inst.dayaheadBlocks[dayahead].da_buy[t]) +
                                                    pyo.value(self.inst.dayaheadBlocks[dayahead].ba_buy[t]) * 1.3 -
                                                    pyo.value(self.inst.dayaheadBlocks[dayahead].ba_sell[t]) * 0.7) * pyo.value(self.inst.electricity_price[t])
                                                    for t in self.inst.T for dayahead in self.inst.dayaheads)

    def get_actions(self):
        """ Only the decisions made within the decision horizon are non-recourse. """
        df = super().get_actions()
        if self.status == TerminationCondition.optimal:
            self.decision_results.dayahead_buy = [pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].da_buy[t]) for t in self.inst.T_r]
            self.decision_results.balancing_buy = [pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].ba_buy[t]) for t in self.inst.T_r]
            self.decision_results.balancing_sell = [pyo.value(self.inst.dayaheadBlocks['ElectricitySpot'].ba_sell[t]) for t in self.inst.T_r]
            df["electricity_purchase"] = self.decision_results.dayahead_buy
            df['balancing_buy'] = self.decision_results.balancing_buy
            df['balancing_sell'] = self.decision_results.balancing_sell
            return df
        else:
            return None


class ShieldRecourseModel(HourlyRecourseModel):
    """ Class which minimizes the L2 distance from exogenously decided actions, respecting the feasible space of the problem. """
    def __init__(self, env, rfp, planning_horizon = 24, decision_horizon = 24, solver = 'scip', allow_spot_buy = True, penalty_type="L1"):
        super().__init__(rfp=rfp,
                         planning_horizon=planning_horizon,
                         decision_horizon=decision_horizon,
                         solver=solver,
                         allow_spot_buy=allow_spot_buy,
                         guideline=None,
                         )
        self.penalty_type = penalty_type
        self.env = env

    def _build_abstract_model(self):
        self.model.actions = pyo.Set(initialize=[(name, t) for name in self.env.action_identity for t in range(self.planning_horizon)])
        self.model.chosen_actions = pyo.Param(self.model.actions, default=0, mutable=True)
        if self.penalty_type == "L1":
            self.model.action_violation = pyo.Var(self.model.actions, domain=pyo.NonNegativeReals)
        super()._build_abstract_model()

    def _build_concrete_instance(self, data=None):
        super()._build_concrete_instance(data)

        if self.penalty_type == "L1":
            self.inst.l1_constraint = pyo.ConstraintList()
            for (description, t) in self.inst.actions:
                blocktype, unit_name, var_name = str(description).split("-")
                # Find the block
                block = next(b for _, b in getattr(self.inst, blocktype + 'Blocks').items() if b._name == unit_name)

                expr = block.__getattribute__(var_name)[t] - self.inst.chosen_actions[(description, t)]
                # Add both constraints for L1 penalty
                self.inst.l1_constraint.add(self.inst.action_violation[(description, t)] >= expr)
                self.inst.l1_constraint.add(self.inst.action_violation[(description, t)] >= -expr)

    def _set_objective(self):
        def objective_rule_shield(inst):
            obj = 0
            for (description, t) in inst.actions:
                if self.penalty_type == "L1":
                    obj += inst.action_violation[(description,t)]
                else:
                    blocktype, unit_name, var_name = str(description).split("-")
                    block = None
                    # Identify the referenced block:
                    for _, b in inst.__getattribute__(blocktype + 'Blocks').items():
                        if b._name == unit_name:
                            block = b
                    expr = (block.__getattribute__(var_name)[t] - inst.chosen_actions[(description,t)])
                    obj += expr * expr # L2 penalty for deviating from actions
                # blocktype, unit_name, var_name = str(description).split("-")
                # block = None
                # # Identify the referenced block:
                # for _, b in inst.__getattribute__(blocktype + 'Blocks').items():
                #     if b._name == unit_name:
                #         block = b
                # expr = (block.__getattribute__(var_name)[t] - inst.chosen_actions[(description,t)])
                # obj += expr * expr # L2 penalty for deviating from actions
            return obj
        self.inst.objective = pyo.Objective(rule=objective_rule_shield, sense=pyo.minimize)


class DecisionRuleModel(HourlyRecourseModel):
    """ LP model to train linear decision rule. We can also extend this by making the weights something that is given, but can be changed (with a penalty) for online learning.
        """
    def __init__(self,
                 rfp,
                 planning_horizon,
                 decision_horizon = 24,
                 solver = 'scip',
                 allow_spot_buy = True,
                 guideline = None,
                 objective_logic = None,
                 n_features=4,
                 n_price_domains=1,
                 domain_prices=[],
                 **kwargs):
        super().__init__(rfp, planning_horizon, decision_horizon, solver, allow_spot_buy, guideline, objective_logic, **kwargs)
        self.n_features = n_features # If 4, then it is: ["Bias", "Forecast Price", "Realized PPA Power", "Realized Price"]
        self.n_rules = 24 # One linear decision rule for each hour.
        self.n_price_domains = n_price_domains
        self.domain_prices = np.asarray(domain_prices)
        assert len(domain_prices)+1 == n_price_domains

    def initialize_model(self):
        # Initialize the optimization model
        self.model = pyo.AbstractModel()
        self._build_abstract_model()
        self.model.features      = pyo.RangeSet(0, self.n_features - 1)
        self.model.feature_hours = pyo.RangeSet(0, self.n_rules - 1)
        self.model.price_domains = pyo.RangeSet(0, self.n_price_domains - 1)
        self.model.feature_data  = pyo.Param(self.model.features, self.model.T, within=pyo.Reals, default=0, mutable=True)
        self.model.cleared_power = pyo.Param(self.model.T, within=pyo.Reals, default=0, mutable=True)
        self.model.fixed_da      = pyo.Param(self.model.T, within=pyo.Binary, default=0, mutable=True) # Whether we should fix the DA decision to cleared_power parameter.
        self.model.domain_prices = pyo.Param(self.model.price_domains, within=pyo.Reals, default=4000, mutable=True)

        if hasattr(self.model, 'dayaheadBlocks'):
            self.model.del_component('dayaheadBlocks')

        def dayaheadBlock_rule(b, da):
            dayahead        = self.rfp.get_component(da)
            b._name         = da
            b.carrier_in    = str(dayahead.parameters["consumes"])
            b.carrier_out   = str(dayahead.parameters["produces"])
            b.capacity      = dayahead.parameters.get('capacity')
            # Power bought from the day-ahead market; negative if power is sold.
            b.out_flow = pyo.Var(self.model.T, domain=pyo.Reals, bounds=(-b.capacity, b.capacity * self.allow_spot_buy))
            b.linear_weights = pyo.Var(self.model.price_domains, self.model.features, self.model.feature_hours, domain=pyo.Reals)
            b.da_buy = pyo.Var(self.model.T, domain=pyo.Reals, bounds=(-b.capacity, b.capacity * self.allow_spot_buy))
            # b.da_positive_exceedance = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
            # b.da_negative_exceedance = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
            b.ba_buy = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            b.ba_sell = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            def balancing_rule(m, t):
                return b.out_flow[t] == b.da_buy[t] + b.ba_buy[t] - b.ba_sell[t]
            b.balancing_constraint = pyo.Constraint(self.model.T, rule=balancing_rule)
            def non_decreasing_bid_rule(m, _pd, h): # We want a buy curve that is decreasing with price. 
                return b.linear_weights[_pd, self.model.features.at(-1), h] <= 0
            b.non_decreasing_bid_constraint = pyo.Constraint(self.model.price_domains, self.model.feature_hours, rule=non_decreasing_bid_rule)
        self.model.dayaheadBlocks = pyo.Block(self.model.dayaheads, rule=dayaheadBlock_rule)
        
        if self.uses_persistent_solver:
            self._build_concrete_instance() # Creates self.inst
            self.solver.set_instance(self.inst)

    def _build_concrete_instance(self, data=None):
        super()._build_concrete_instance(data)
        
        def linear_mapping_rule(inst, da, t): # Fix hourly production of ammonia:
            b = inst.dayaheadBlocks[da]
            h = pyo.value(inst.T_datetime[t]).hour % self.n_rules # Get hour of day (and dynamically select what feature set this one belongs to)
            price = pyo.value(inst.feature_data[inst.features.at(-1), t])
            _pd = sum(pyo.value(val) < price for d,val in inst.domain_prices.items())
            da_buy_decision_rule = sum(b.linear_weights[_pd, f, h] * inst.feature_data[f, t] for f in inst.features)
            return b.da_buy[t] == da_buy_decision_rule # - b.da_positive_exceedance[t] + b.da_negative_exceedance[t]
        self.inst.linear_mapping_constraint = pyo.Constraint(self.inst.dayaheads, self.inst.T, rule=linear_mapping_rule)
        
        def price_domain_rule(inst, da, _pd, t):
            if _pd == 0:
                return pyo.Constraint.Skip
            else:
                b = inst.dayaheadBlocks[da]
                h = pyo.value(inst.T_datetime[t]).hour % self.n_rules
                lambda_price = pyo.value(inst.domain_prices[_pd-1])
                low_price_domain_buy = (lambda_price * b.linear_weights[_pd-1, inst.features.at(-1), h] + 
                                         sum(b.linear_weights[_pd-1, f, h] * inst.feature_data[f, t]
                                             for f in inst.features if f < len(inst.features)-1))
                high_price_domain_buy = (lambda_price * b.linear_weights[_pd, inst.features.at(-1), h] + 
                                         sum(b.linear_weights[_pd, f, h] * inst.feature_data[f, t] 
                                             for f in inst.features if f < len(inst.features)-1))
                return high_price_domain_buy <= low_price_domain_buy # ensure that what we bid to buy in the high price domain is lower than in the low price domain.
        self.inst.price_domain_constraint = pyo.Constraint(self.inst.dayaheads, self.inst.price_domains, self.inst.T, rule=price_domain_rule)
        
        # def positive_exceedance_rule(inst, da, t):
        #     b = inst.dayaheadBlocks[da]
        #     ppa_power = sum(inst.ppaBlocks[ppa].out_flow[t] for ppa in inst.ppas)
        #     gcp_cap = inst.linkBlocks["Grid Connection Point"].capacity/inst.linkBlocks["Grid Connection Point"].rate
        #     power_buy_cap = gcp_cap - ppa_power
        #     return b.da_positive_exceedance[t] >= b.da_buy[t] - power_buy_cap
        # self.inst.positive_exceedance_constraint = pyo.Constraint(self.inst.dayaheads, self.inst.T, rule=positive_exceedance_rule)
        
        # def negative_exceedance_rule(inst, da, t):
        #     b = inst.dayaheadBlocks[da]
        #     ppa_power = sum(inst.ppaBlocks[ppa].out_flow[t] for ppa in inst.ppas)
        #     power_sell_cap = ppa_power
        #     return b.da_negative_exceedance[t] >= -b.da_buy[t] - power_sell_cap
        # self.inst.negative_exceedance_constraint = pyo.Constraint(self.inst.dayaheads, self.inst.T, rule=negative_exceedance_rule)
        
        self.updated_constraints += ["linear_mapping_constraint", "price_domain_constraint"]
    
    def get_weights(self):
        status = self.solve_message['Solver'][0]['Termination condition']
        if status == TerminationCondition.optimal:
            b = self.inst.dayaheadBlocks["ElectricitySpot"]
            weights = np.asarray([[[pyo.value(b.linear_weights[_pd, f, t]) for t in self.inst.feature_hours] for f in self.inst.features] for _pd in self.inst.price_domains])
            return weights, False # No truncation
        else:
            return None, True # Truncate episode


class StochasticRecourseModel(HourlyRecourseModel):
    """
    * Stochastic decision-making model. All decision variables are recourse, except the day-ahead decisions. *

    This model differs from the deterministic model by considering different scenarios of VRE and price evolution.
    We still assume that the first 24 hours of VRE are certain. (As they are for the asset, but not for the system).
    Now instead of forecasting 1 scenario of prices, we simulate 'n_scenarios' scenarios of prices.
    """
    model_types = ("recourse DA", "non-recourse DA", "non-recourse flows")

    def __init__(self,
                 rfp: RenewableFuelPlant,
                 planning_horizon: int,
                 decision_horizon: int = 24,
                 fixed_horizon: int = 12,
                 solver: str = 'scip',
                 allow_spot_buy: bool = True,
                 guideline: str|None = None,
                 n_scenarios: int = 1, # If 1, then it is identical to the deterministic model.
                 model_type = "non-recourse DA",
                 **kwargs,
                 ):
        super().__init__(rfp, planning_horizon, decision_horizon, solver, allow_spot_buy, guideline)
        self.fixed_horizon = fixed_horizon
        self.n_scenarios = n_scenarios
        self.model_type = model_type

    def initialize_model(self):
        # Initialize the optimization model
        self.model = pyo.AbstractModel()
        self._build_abstract_model()
        if self.uses_persistent_solver:
            self._build_concrete_instance() # Creates self.inst
            self.solver.set_instance(self.inst)

    def _build_abstract_model(self):
        # Model Time Sets:
        self.model.T    = pyo.RangeSet(0, self.fixed_horizon + self.planning_horizon - 1)  # Time steps
        self.model.T_fix_dayahead = pyo.RangeSet(self.fixed_horizon, self.fixed_horizon + self.decision_horizon - 1) # Used for results processing
        self.model.T_fix_recourse = pyo.RangeSet(0, self.decision_horizon - 1) # Used for results processing

        # Stochastic scenarios:
        self.model.S    = pyo.RangeSet(0, self.n_scenarios-1) # Set of price scenarios
        self.model.weights = pyo.Param(self.model.S, within=pyo.NonNegativeReals, default=1/self.n_scenarios, mutable=True)

        # Model Set definitions
        self.model.carriers     = pyo.Set(initialize=[name for name in self.rfp.get_carriers().keys()])
        self.model.storages     = pyo.Set(initialize=[name for name in self.rfp.get_storages().keys()])
        self.model.ppas         = pyo.Set(initialize=[name for name in self.rfp.get_ppas().keys()])
        self.model.dayaheads    = pyo.Set(initialize=[name for name in self.rfp.get_dayaheads().keys()])
        self.model.links        = pyo.Set(initialize=[name for name in self.rfp.get_links().keys()])
        self.model.offtakers    = pyo.Set(initialize=[name for name in self.rfp.get_offtakers().keys()])
        self.model.contracts    = pyo.Set(initialize=[name for name in self.rfp.get_contracts().keys()])

        # Mutable model parameters:
        self.model.T_datetime   = pyo.Param(self.model.T, within=pyo.Any, initialize=pd.date_range(start=0, end=self.planning_horizon - 1, freq='h'), mutable=True)
        self.model.init_soc           = pyo.Param(self.model.storages, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.supplier_cf        = pyo.Param(self.model.ppas, self.model.S, self.model.T, within=pyo.NonNegativeReals, default=1, mutable=True)
        self.model.init_contract_status         = pyo.Param(self.model.contracts, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.electricity_price            = pyo.Param(self.model.S, self.model.T, within=pyo.Reals, default=50, mutable=True)

        self.model.cleared_power = pyo.Param(self.model.T, within=pyo.Reals, default=0, mutable=True)
        self.model.fixed_da = pyo.Param(self.model.T, within=pyo.Binary, default=0, mutable=True) # Whether we should fix the DA decision to cleared_power parameter.
        # self.model.offtaker_availability        = pyo.Param(self.model.offtakers, self.model.T, within=pyo.Binary, default=1, mutable=True)

        # Guideline related mutable parameters:
        if self.guideline == "production_value": # Specified value of outflow of links.
            self.model.production_value = pyo.Param(self.model.links, within=pyo.Reals, default=0, mutable=True)
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
            b.capacity      = link.parameters.get('capacity', np.inf)
            b.ec            = link.parameters.get("electricity_consumption", 0) # Electricity consumption rate
            b.carrier_in    = str(link.parameters["consumes"])
            b.carrier_out   = str(link.parameters["produces"])
            b.in_flow       = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity/b.rate))
            b.out_flow      = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            if b.ec > 0:
                b.elec_cons = pyo.Var(self.model.S, self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity * b.ec))
                def ec_rule(m, s, t):
                    return b.elec_cons[s,t] == b.ec * b.out_flow[s,t]
                b.ec_constraint = pyo.Constraint(self.model.S, self.model.T, rule=ec_rule)
            def conversion_rule(m, s, t):
                return b.out_flow[s,t] == b.rate * b.in_flow[s,t]
            b.conversion_constraint = pyo.Constraint(self.model.S, self.model.T, rule=conversion_rule)
            if self.model_type == "non_recourse flows":
                b.main_in_flow = pyo.Var(self.model.T_fix_recourse, domain=pyo.NonNegativeReals, bounds=(0, b.capacity/b.rate))
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
    
    def _build_concrete_instance(self, data=None):
        self.inst = self.model.create_instance(data=data)
        self.updated_constraints = []

        """ Helper methods to determine whether it is shipment or target time for contracts. """
        def _get_datetime_infos(inst, t):
            if t == -1:
                dt_t = inst.T_datetime[t+1].value - pd.Timedelta(1, 'h')
            else:
                dt_t = inst.T_datetime[t].value
            is_day_end = (dt_t.hour == 23)
            is_month_end = dt_t.is_month_end
            if isinstance(is_month_end, list):
                is_month_end = is_month_end[0]
            is_year_end = dt_t.is_year_end
            if isinstance(is_year_end, list):
                is_year_end = is_year_end[0]
            return is_day_end, is_month_end, is_year_end
        
        def _is_target_time(inst, b, t):
            assert (b.target_frequency in self.frequency_options), f"{b.target_frequency} for {b._name} is not in options.\nOptions are {self.frequency_options}."
            is_day_end, is_month_end, is_year_end = _get_datetime_infos(inst, t)
            is_planning_end = (t == (self.planning_horizon - 1))
            return bool((b.target_frequency == 'hourly') or                                  # If we have an hourly contract.
                        (b.target_frequency == 'daily'   and is_day_end) or                  # If we have a daily contract and it is end-of-day (EOD).
                        (b.target_frequency == 'monthly' and is_month_end and is_day_end) or # If we have a monthly contract and it is end-of-month and EOD.
                        (b.target_frequency == 'yearly'  and is_year_end and is_day_end) or  # If we have a yearly contract and it is end-of-year and EOD.
                        (b.target_frequency == 'planning_horizon' and is_planning_end))      # If we are constraining the problem on the planning horizon.
        
        def _is_shipment_time(inst, b, t):
            assert (b.shipment_frequency in self.frequency_options), f"{b.shipment_frequency} for {b._name} is not in options.\nOptions are {self.frequency_options}."
            is_day_end, is_month_end, is_year_end = _get_datetime_infos(inst, t)
            return bool((b.shipment_frequency == 'hourly') or                                   # If we have an hourly shipment.
                        (b.shipment_frequency == 'daily'   and is_day_end) or                   # If we have a daily shipment and it is end-of-day (EOD).
                        (b.shipment_frequency == 'monthly' and is_month_end and is_day_end) or  # If we have a monthly shipment and it is end-of-month and EOD.
                        (b.shipment_frequency == 'yearly'  and is_year_end and is_day_end))     # If we have a yearly shipment and it is end-of-year and EOD.

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
                    if comp.is_link:
                        b._in[s,t].append(inst.linkBlocks[name].out_flow[s,t])
                    elif comp.is_storage:
                        b._in[s,t].append(inst.storageBlocks[name].out_flow[s,t])
                    elif comp.is_ppa:
                        b._in[s,t].append(inst.ppaBlocks[name].out_flow[s,t])
                    elif comp.is_dayahead:
                        b._in[s,t].append(inst.dayaheadBlocks[name].out_flow[s,t])
                if comp.parameters.get("consumes") == b.type:
                    if comp.is_link:
                        b._out[s,t].append(inst.linkBlocks[name].in_flow[s,t])
                    elif comp.is_storage:
                        b._out[s,t].append(inst.storageBlocks[name].in_flow[s,t])
                    elif comp.is_offtaker:
                        b._out[s,t].append(inst.offtakerBlocks[name].in_flow[s,t])
                if b.type == "plant_electricity":
                    if comp.parameters.get("electricity_consumption", 0) > 0:
                        if comp.is_link:
                            b._out[s,t].append(inst.linkBlocks[name].elec_cons[s,t]) # Using a link can consume electricity.
                        elif comp.is_storage:
                            b._out[s,t].append(inst.storageBlocks[name].elec_cons[s,t]) # Charging a storage can consume electricity.
                        elif comp.is_offtaker:
                            b._out[s,t].append(inst.offtakerBlocks[name].elec_cons[s,t]) # Using an offtaker can consume electricity.
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

        """ Rules that keeps track of delivery to contracts. """
        def contract_shipment_rule(inst, cont, s, t): # Ensure that the shipments happen only at shipment time.
            b = inst.contractBlocks[cont]
            if _is_shipment_time(inst, b, t):
                return pyo.Constraint.Skip # Then don't constrain the shipment more than its existing bounds.
            else:
                return b.shipment[s,t] == 0 # Otherwise there cannot be any sales for this contract for this hour.
        self.inst.shipment_constraint = pyo.Constraint(self.inst.contracts, self.inst.S, self.inst.T, rule=contract_shipment_rule)

        def contract_status_rule(inst, cont, s, t): # The shipments we gets added to the contract_status.
            b = inst.contractBlocks[cont]
            if b.is_spot_contract:
                return pyo.Constraint.Skip
            else:
                if _is_target_time(inst, b, t-1):
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
                if _is_target_time(inst, b, t):
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

    def _set_objective(self):
        def cashflow_rule(inst):
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
        
        def production_value_rule(inst):
            production_value = sum(sum(inst.linkBlocks[link].out_flow[s,t] * inst.production_value[link] for link in inst.links for t in inst.T) * inst.weights[s] for s in inst.S)
            remove_shipment_revenue_incentive = sum(sum(b.shipment[s,t] * (0.95 * b.price) for name, b in inst.contractBlocks.items() for t in inst.T) * inst.weights[s] for s in inst.S)
            return production_value - remove_shipment_revenue_incentive
        
        def state_value_rule(inst):
            storage_value = sum(sum(b.soc[s,self.decision_horizon-1] * inst.storage_value[name] for name, b in inst.storageBlocks.items()) * inst.weights[s] for s in inst.S)
            contract_value = sum(sum(b.contract_status[s,self.decision_horizon-1] * inst.contract_value[name] for name, b in inst.contractBlocks.items() if b.is_spot_contract == False) * inst.weights[s] for s in inst.S)
            return storage_value + contract_value

        def objective_rule(inst):
            obj = cashflow_rule(inst)
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


