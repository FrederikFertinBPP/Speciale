from common_scripts.RFP_initialization import RenewableFuelPlant, create_rfp
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import numpy as np
import pandas as pd

class expando(object):
    pass

class HourlyDeterministicLPModel:

    def __init__(self,
                 rfp: RenewableFuelPlant,
                 planning_horizon: int,
                 frequency: str = "hourly",
                 decision_horizon: int = 24,
                 solver: str = 'scip',
                 guideline: str|None = 'strike_price',
                 ):
        self.rfp            = rfp
        self.horizon        = planning_horizon
        self.step_horizon   = min(decision_horizon, self.horizon)
        self.frequency      = frequency
        self.guideline      = guideline
        self.penalty_scaler = 4
        self.model          = None
        self.inst           = None
        if solver == 'scip': # 'ipopt'
            self.opt        = SolverFactory(solver, solver_io='nl')
        else:
            self.opt        = SolverFactory(solver)
        self.results = expando()
        self.flow_results = expando()
        self.soc_results = expando()
        # Initialize the optimization model
        self._build_abstract_model()

    def _build_abstract_model(self):
        # 1. Initialize the model
        self.model      = pyo.AbstractModel()
        self.model.T_r  = pyo.RangeSet(0, self.step_horizon - 1) # Used for results processing
        self.model.T    = pyo.RangeSet(0, self.horizon - 1)  # Time steps

        self.model.carriers     = pyo.Set(initialize=[name for name, carr in self.rfp.get_carriers()])
        self.model.links        = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_link])
        self.model.storages     = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_storage])
        self.model.suppliers    = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_supplier])
        self.model.offtakers    = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_offtaker])
        self.model.contracts    = pyo.Set(initialize=[name for name, cont in self.rfp.get_contracts()])

        self.model.init_soc           = pyo.Param(self.model.storages, within=pyo.NonNegativeReals, default=0)
        self.model.supplier_cf        = pyo.Param(self.model.suppliers, self.model.T, within=pyo.NonNegativeReals, default=1)
        self.model.contract_target    = pyo.Param(self.model.contracts, initialize={name: cont.parameters.get("volume",0) for name, cont in self.rfp.get_contracts()})
        self.model.electricity_price  = pyo.Param(self.model.T, within=pyo.Reals, default=50)
        self.model.ammonia_strike_price = pyo.Param(within=pyo.Reals, default=self.rfp.get_contract('Ammonia1').parameters['price'])

        def storageBlock_rule(b, stor): # Create a block for each storage to handle charge/discharge and state of charge
            storage         = self.rfp.get_component(stor)
            b.capacity      = storage.parameters["capacity"]
            b.ec            = storage.parameters.get("charge_rate", 0) # Electricity consumption rate
            b.carrier_in    = str(storage.parameters["consumes"])
            b.carrier_out   = str(storage.parameters["produces"])

            b.soc       = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            b.in_flow   = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            if b.carrier_in != 'ammonia':
                b.out_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            if b.ec > 0:
                b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity*b.ec))
                def ec_rule(m, t):
                    return b.elec_cons[t] == b.in_flow[t] * b.ec
                b.ec_constraint = pyo.Constraint(self.model.T, rule=ec_rule)
            def soc_rule(m, t):
                if t == 0:
                    return pyo.Constraint.Skip # To be defined in concrete instance
                else:
                    # We do not model outflow of ammonia tank. This is done in upper level model.
                    if b.carrier_in == 'ammonia':
                        return b.soc[t] == b.soc[t-1] + b.in_flow[t]
                    else:
                        return b.soc[t] == b.soc[t-1] + b.in_flow[t] - b.out_flow[t]
            b.soc_constraint = pyo.Constraint(self.model.T, rule=soc_rule)
        self.model.storageBlocks = pyo.Block(self.model.storages, rule=storageBlock_rule)

        def supplierBlock_rule(b, supp): # Create a block for each supplier to handle production
            supplier = self.rfp.get_component(supp)
            b.carrier_in = str(supplier.parameters["consumes"])
            b.carrier_out = str(supplier.parameters["produces"])
            b.capacity = supplier.parameters.get('capacity', np.inf)
            b.price = supplier.ppa.parameters["price"]
            # b.curtailment = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, supplier.parameters.get('capacity', np.inf)))
            b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, supplier.parameters.get('capacity', np.inf)))
        self.model.supplierBlocks = pyo.Block(self.model.suppliers, rule=supplierBlock_rule)

        def linkBlock_rule(b, lin): # Create a block for each link to handle conversions between carriers
            link = self.rfp.get_component(lin)
            b.rate = link.parameters.get("rate", 1)
            b.capacity = link.parameters.get('capacity', np.inf)
            b.ec = link.parameters.get("charge_rate", 0) # Electricity consumption rate
            b.carrier_in = str(link.parameters["consumes"])
            b.carrier_out = str(link.parameters["produces"])
            b.in_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity/b.rate))
            b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            if b.ec > 0:
                b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity * b.ec))
                def ec_rule(m, t):
                    return b.elec_cons[t] == b.out_flow[t] * b.ec
                b.ec_constraint = pyo.Constraint(self.model.T, rule=ec_rule)
            def conversion_rule(m, t):
                return b.out_flow[t] == b.rate * b.in_flow[t]
            b.conversion_constraint = pyo.Constraint(self.model.T, rule=conversion_rule)
        self.model.linkBlocks = pyo.Block(self.model.links, rule=linkBlock_rule)

        def offtakerBlock_rule(b, offt): # Create a block for each offtaker to handle consumption
            offtaker = self.rfp.get_component(offt)
            b.carrier_in    = str(offtaker.parameters["consumes"])
            b.carrier_out   = str(offtaker.parameters["produces"])
            b.contracts     = pyo.Set(initialize=[cont.name for cont in offtaker.contracts])
            b.prices                = {cont.name: cont.parameters.get("price", 1) for cont in offtaker.contracts}
            b.target_frequencies    = {cont.name: cont.parameters.get("target_frequency", "yearly") for cont in offtaker.contracts}
            b.shipment_frequencies  = {cont.name: cont.parameters.get("shipment_frequency", "monthly") for cont in offtaker.contracts}
            b.sales                 = pyo.Var(b.contracts, self.model.T, domain=pyo.NonNegativeReals)
            b.contract_shortfall    = pyo.Var(b.contracts, self.model.T, domain=pyo.NonNegativeReals)
            if b.carrier_in != 'ammonia': # For Ammonia, the actual flow to the offtaker does not happen every hour. The fuel is kept in the storage.
                b.in_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, offtaker.parameters.get('capacity', np.inf)))
            def sales_equality_rule(m, t):
                if b.carrier_in == 'ammonia': # We create the adequate constraint in the concrete model.
                    return pyo.Constraint.Skip # sum(b.sales[c,t] for c in b.contracts) == b.in_flow[t]
                else:
                    return sum(b.sales[c,t] for c in b.contracts) == b.in_flow[t] # We should have an offtaker for everything we produce, basically.
            b.sales_constraint = pyo.Constraint(self.model.T, rule=sales_equality_rule)
        self.model.offtakerBlocks = pyo.Block(self.model.offtakers, rule=offtakerBlock_rule)

        def carrierBlock_rule(b, carr): # Create a block for each energy carrier to enforce nodal carrier balances
            carrier = self.rfp.get_carrier(carr)
            b.type = carrier.name
            b._in = {t: [] for t in self.model.T}
            b._out = {t: [] for t in self.model.T}
            if b.type == "electricity":
                grid_capacity = self.rfp.get_component("GCP").parameters.get("capacity", 1000)
                b.grid_sales = pyo.Var(self.model.T, domain=pyo.Reals, bounds=(0, grid_capacity)) # Grid sale variable, cannot buy from spot market.
        self.model.carrierBlocks = pyo.Block(self.model.carriers, rule=carrierBlock_rule)

    def build_concrete_instance(self, data=None):
        self.inst = self.model.create_instance(data=data)

        def init_soc_rule(inst, stor):
            b = inst.storageBlocks[stor]
            if b.carrier_in == 'ammonia':
                return b.soc[0] == inst.init_soc[stor] + b.in_flow[0]
            else:
                return b.soc[0] == inst.init_soc[stor] + b.in_flow[0] - b.out_flow[0]
        self.inst.init_soc_constraint = pyo.Constraint(self.inst.storages, rule=init_soc_rule)

        def ppa_procurement_rule(inst, supp, t):
            b = inst.supplierBlocks[supp]
            return b.out_flow[t] == inst.supplier_cf[supp, t] * b.capacity
        self.inst.ppa_procurement_constraint = pyo.Constraint(self.inst.suppliers, self.inst.T, rule=ppa_procurement_rule)

        def carrier_balance_rule(inst, carr, t):
            b = inst.carrierBlocks[carr]
            for name, comp in self.rfp.get_components():
                if comp.parameters.get("produces") == b.type:
                    if comp.is_link:
                        b._in[t].append(inst.linkBlocks[name].out_flow[t])
                    elif comp.is_storage and b.type != 'ammonia':
                        b._in[t].append(inst.storageBlocks[name].out_flow[t])
                    elif comp.is_supplier:
                        b._in[t].append(inst.supplierBlocks[name].out_flow[t])
                if comp.parameters.get("consumes") == b.type:
                    if comp.is_link:
                        b._out[t].append(inst.linkBlocks[name].in_flow[t])
                    elif comp.is_storage:
                        b._out[t].append(inst.storageBlocks[name].in_flow[t])
                    elif comp.is_offtaker and b.type != 'ammonia':
                        b._out[t].append(inst.offtakerBlocks[name].in_flow[t])
                if b.type == "electricity":
                    if comp.parameters.get("charge_rate", 0) > 0:
                        if comp.is_link:
                            b._out[t].append(inst.linkBlocks[name].elec_cons[t]) # Using a link consumes electricity (relevant for haber-bosch)
                        elif comp.is_storage:
                            b._out[t].append(inst.storageBlocks[name].elec_cons[t]) # Charging a storage consumes electricity
            if b.type == "electricity":
                b._out[t].append(b.grid_sales[t]) # Positive values indicate selling to the grid, negative values indicate buying from the grid
            return sum(b._in[t]) == sum(b._out[t]) # Carrier balance arcs
        self.inst.carrier_balance_constraint = pyo.Constraint(self.inst.carriers, self.inst.T, rule=carrier_balance_rule)

        # Here we define that ammonia production should be associated with specific offtake.
        def ammonia_offtake_rule(inst, t):
            b = inst.offtakerBlocks['Ammonia Shipment']
            return inst.storageBlocks['Ammonia Storage'].in_flow[t] == sum(b.sales[cont,t] for cont in b.contracts)
        self.inst.ammonia_offtake_constraint = pyo.Constraint(self.inst.T, rule=ammonia_offtake_rule)

        # Ensure that grid capacity is not exceeded by supply imports
        def grid_capacity_rule(inst, t):
            grid_capacity = self.rfp.get_component("GCP").parameters.get('capacity', 1000) # Assume a large capacity if not specified
            return (0, # No world where we are supplying electricity to the grid
                    sum(inst.supplierBlocks[name].out_flow[t] for name, comp in self.rfp.get_components() if comp.is_supplier) - \
                          inst.carrierBlocks["electricity"].grid_sales[t],
                    grid_capacity)
        self.inst.grid_capacity_constraint = pyo.Constraint(self.inst.T, rule=grid_capacity_rule)

        def hourly_target_rule(inst, cont, t):
            contract = self.rfp.get_contract(cont)
            b = inst.offtakerBlocks[contract.offtaker]
            if contract.target_frequency == "hourly":
                return b.sales[cont,t] == inst.contract_target[cont]
            else:
                return pyo.Constraint.Skip
        self.inst.hourly_volume_constraint = pyo.Constraint(self.inst.contracts, self.inst.T, rule=hourly_target_rule)

        if self.guideline == 'strike_price':
            self.inst.offtakerBlocks['Ammonia Shipment'].prices['Ammonia1'] = self.inst.ammonia_strike_price
        elif self.guideline == 'planning_target':
            self.inst.offtakerBlocks['Ammonia Shipment'].prices['Ammonia1'] = self.inst.ammonia_strike_price
            def guideline_planning_target_rule(inst, cont):
                contract = self.rfp.get_contract(cont)
                b = inst.offtakerBlocks[contract.offtaker]
                if contract.target_frequency != "hourly" and contract.target_frequency is not None:
                    return sum(b.sales[cont,t] for t in inst.T) + sum(b.contract_shortfall[cont,t] for t in inst.T) == self.inst.contract_target[cont]
                else:
                    return pyo.Constraint.Skip
            self.inst.planning_target_constraint = pyo.Constraint(self.inst.contracts, rule=guideline_planning_target_rule)

        self._set_objective()

    def _set_objective(self):
        def objective_rule(inst):
            revenue = sum(inst.carrierBlocks["electricity"].grid_sales[t] * inst.electricity_price[t] for t in inst.T)
            revenue += sum(sum(inst.offtakerBlocks[offt].sales[cont,t] * inst.offtakerBlocks[offt].prices[cont] for cont in inst.offtakerBlocks[offt].contracts) for offt in inst.offtakers for t in inst.T)
            costs = sum(inst.supplierBlocks[supp].out_flow[t] * inst.supplierBlocks[supp].price for supp in inst.suppliers for t in inst.T)
            costs += sum(sum(inst.offtakerBlocks[offt].contract_shortfall[cont,t] * inst.offtakerBlocks[offt].prices[cont] * self.penalty_scaler for cont in inst.offtakerBlocks[offt].contracts) for offt in inst.offtakers for t in inst.T)
            return revenue - costs
        self.inst.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    def run(self, verbose=False):
        if self.inst:
            self.solve_message = self.opt.solve(self.inst, tee=verbose)
            status = self.solve_message['Solver'][0]['Termination condition']
            if status == TerminationCondition.optimal:
                self._save_solution()
            elif status == TerminationCondition.infeasibleOrUnbounded:
                print(Warning("Could not solve the problem, status: " + str(self.solve_message['Solver'][0]['Termination condition'])))
            else:
                print(Warning("Non-optimal LP, status: " + str(status)))
        else:
            raise("Initialize concrete instance of model with data before running.")
    
    def _save_solution(self):
        self.results.final_soc          = {name : self.inst.storageBlocks[name].soc[self.step_horizon-1].value
                                           for name in self.inst.storages}
        self.results.fuel_sales         = {offt : {cont :
                                                    [self.inst.offtakerBlocks[offt].sales[cont,t].value for t in self.inst.T_r]
                                                    for cont in self.inst.offtakerBlocks[offt].contracts}
                                                    for offt in self.inst.offtakers}
        self.results.sale_totals        = {offt : {cont : 
                                                    sum(self.results.fuel_sales[offt][cont]) 
                                                    for cont in self.inst.offtakerBlocks[offt].contracts}
                                                    for offt in self.inst.offtakers}
        self.results.fuel_revenue       = sum(self.results.sale_totals[offt][cont] * self.inst.offtakerBlocks[offt].prices[cont]
                                                    for offt in self.inst.offtakers for cont in self.inst.offtakerBlocks[offt].contracts)
        self.results.grid_sale          = [self.inst.carrierBlocks['electricity'].grid_sales[t].value for t in self.inst.T_r]
        self.results.electricity_price  = [self.inst.electricity_price[t] for t in self.inst.T_r]
        self.results.exp_el_revenue     = sum(self.results.grid_sale[t] * self.results.electricity_price[t] for t in self.inst.T_r)
        self.results.costs              = sum(self.inst.supplierBlocks[supp].out_flow[t].value * self.inst.supplierBlocks[supp].price for supp in self.inst.suppliers for t in self.inst.T_r)
        # Save all flow and soc results as well for plotting purposes
        self.results.storage_soc    = {name : [self.inst.storageBlocks[name].soc[t].value for t in self.inst.T_r] for name in self.inst.storages}
        self.results.storage_inflow = {name : [self.inst.storageBlocks[name].in_flow[t].value for t in self.inst.T_r] for name in self.inst.storages}
        self.results.storage_outflow = {name : [self.inst.storageBlocks[name].out_flow[t].value for t in self.inst.T_r] for name in self.inst.storages if name != 'Ammonia Storage'}
        self.results.production     = {name : [self.inst.linkBlocks[name].out_flow[t].value for t in self.inst.T_r] for name in self.inst.links}

    def get_actions(self):
        """ Only the decisions made within the decision horizon are non-recourse. """
        self.results.electrolyzer_power = [self.inst.linkBlocks['Electrolyzer'].in_flow[t].value for t in self.inst.T_r]
        self.results.hydrogen_pipeline = [self.inst.offtakerBlocks['Hydrogen Pipeline'].in_flow[t].value for t in self.inst.T_r]
        self.results.ammonia_production = [self.inst.linkBlocks['Haber-Bosch Plant'].out_flow[t].value for t in self.inst.T_r]
        self.results.ammonia_spot_sale = [self.inst.offtakerBlocks['Ammonia Shipment'].sales["AmmoniaSpot",t].value for t in self.inst.T_r]
        self.results.electricity_sale = [self.inst.carrierBlocks['electricity'].grid_sales[t].value for t in self.inst.T_r]
        return pd.DataFrame(index=self.inst.T_r, data = {
            'electricity_sale'   : self.results.electricity_sale,
            'electrolyzer_power' : self.results.electrolyzer_power,
            'hydrogen_sale'      : self.results.hydrogen_pipeline,
            'ammonia_production' : self.results.ammonia_production,
            'ammonia_spot_sale'  : self.results.ammonia_spot_sale,
            })

def calculate_realized_revenue(results, realized_prices):
    el_revenue = sum(results.grid_sale[t] * realized_prices[t] for t in range(len(realized_prices)))
    return el_revenue - results.fuel_revenue - results.costs

if __name__ == "__main__":
    rfp = create_rfp()
    horizon = 24 * 4
    step_horizon = 24
    pfm = HourlyDeterministicLPModel(rfp, planning_horizon=horizon, decision_horizon=step_horizon, solver='gurobi')
    wind_cf = {('WindPower', t): 0.5 + 0.5 * np.sin(4 * np.pi * t / 24) for t in range(horizon)}
    solar_cf = {('SolarPower', t): max(0, -np.sin(np.pi/2 + 2 * np.pi * t / 24)) for t in range(horizon)}
    nuclear_cf = {('NuclearPower', t): 1.0 for t in range(horizon)}
    electricity_price = {t: 70 + 40 * np.sin(2 * np.pi * (t - 6) / 24) for t in range(horizon)}
    data = {
        None: {
            'init_soc': {
                'Hydrogen Storage': 0,
                'Ammonia Storage': 0,
            },
            'contract_target': {
                'Hydrogen1': 2,
                'Ammonia1': 20*365*24/2,
            },
            'supplier_cf': {
                **wind_cf,
                **solar_cf,
                **nuclear_cf,
            },
            'electricity_price': electricity_price,
            'ammonia_strike_price': {None: 1000},
        }
    }
    pfm.build_concrete_instance(data=data)
    pfm.run()
    pfm.print_solution()