from RFP_initialization import RenewableFuelPlant, make_rfp
import pyomo.environ as pyo
from pyomo.network import Port, Arc
import numpy as np

class PerfectForesightModel:
    def __init__(self, rfp: RenewableFuelPlant, horizon: int):
        self.rfp            = rfp
        self.horizon        = horizon
        self.variables      = {}
        self.constraints    = {}
        self.objective      = None
        self.model          = None
        # Initialize the optimization model
        self._build_abstract_model()

    def _build_abstract_model(self):
        # 1. Initialize the model
        self.model = pyo.AbstractModel()
        self.model.T = pyo.RangeSet(0, self.horizon - 1)  # Time steps
        self.model.carriers = pyo.Set(initialize=[name for name, carr in self.rfp.get_carriers()])

        self.model.links        = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_link])
        self.model.storages     = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_storage])
        self.model.suppliers    = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_supplier])
        self.model.offtakers    = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_offtaker])
        self.model.contracts    = pyo.Set(initialize=[name for name, cont in self.rfp.get_contracts()])

        self.model.init_soc           = pyo.Param(self.model.storages, within=pyo.NonNegativeReals, default=0)
        self.model.supplier_cf        = pyo.Param(self.model.suppliers, self.model.T, within=pyo.NonNegativeReals, default=1)
        self.model.contract_target    = pyo.Param(self.model.contracts, initialize={name: cont.parameters["volume"] for name, cont in self.rfp.get_contracts()})
        self.model.electricity_price  = pyo.Param(self.model.T, within=pyo.Reals, default=50)
        self.model.ammonia_price      = pyo.Param(self.model.T, within=pyo.NonNegativeReals, initialize=self.rfp.get_contract("Ammonia1").parameters.get("price", 1000))
        self.model.hydrogen_price     = pyo.Param(self.model.T, within=pyo.NonNegativeReals, initialize=self.rfp.get_contract("Hydrogen1").parameters.get("price", 3000))

        def storageBlock_rule(b, stor): # Create a block for each storage to handle charge/discharge and state of charge
            storage = self.rfp.get_component(stor)
            b.soc = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, storage.parameters["capacity"]))
            b.charge = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, storage.parameters["capacity"]))
            b.discharge = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, storage.parameters["capacity"]))
            b.in_port = Port()
            b.out_port = Port()
            b.carrier_in = str(storage.parameters["consumes"])
            b.carrier_out = str(storage.parameters["produces"])
            def soc_rule(m, t):
                if t == 0:
                    pass # To be defined in concrete instance
                else:
                    return b.soc[t] == b.soc[t-1] + b.charge[t] - b.discharge[t]
            b.soc_constraint = pyo.Constraint(self.model.T, rule=soc_rule)
            b.in_port.add(b.charge, "in_flow")
            b.out_port.add(b.discharge, "out_flow")
        self.model.storageBlocks = pyo.Block(self.model.storages, rule=storageBlock_rule)

        def supplierBlock_rule(b, supp): # Create a block for each supplier to handle production
            supplier = self.rfp.get_component(supp)
            b.out_port = Port()
            b.production = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, supplier.parameters.get('capacity', np.inf)))
            b.carrier_in = str(supplier.parameters["consumes"])
            b.carrier_out = str(supplier.parameters["produces"])
            b.capacity = supplier.parameters.get('capacity', np.inf)
            b.out_port.add(b.production, "out_flow")
        self.model.supplierBlocks = pyo.Block(self.model.suppliers, rule=supplierBlock_rule)

        def linkBlock_rule(b, lin): # Create a block for each link to handle conversions between carriers
            link = self.rfp.get_component(lin)
            b.in_port = Port()
            b.out_port = Port()
            b.rate = link.parameters.get("rate", 1)
            b.capacity = link.parameters.get('capacity', np.inf)
            b.in_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity*b.rate))
            b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
            def conversion_rule(m, t):
                return b.out_flow[t] == b.rate * b.in_flow[t]
            b.conversion_constraint = pyo.Constraint(self.model.T, rule=conversion_rule)
            b.carrier_in = str(link.parameters["consumes"])
            b.carrier_out = str(link.parameters["produces"])
            b.ec = link.parameters.get("charge_rate", 0)
            b.in_port.add(b.in_flow, "in_flow")
            b.out_port.add(b.out_flow, "out_flow")
        self.model.linkBlocks = pyo.Block(self.model.links, rule=linkBlock_rule)

        def offtakerBlock_rule(b, offt): # Create a block for each offtaker to handle consumption
            offtaker = self.rfp.get_component(offt)
            b.in_port = Port()
            b.consumption = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, offtaker.parameters.get('capacity', np.inf)))
            b.carrier_in = str(offtaker.parameters["consumes"])
            b.carrier_out = str(offtaker.parameters["produces"])
            b.in_port.add(b.consumption, "in_flow")
        self.model.offtakerBlocks = pyo.Block(self.model.offtakers, rule=offtakerBlock_rule)

        def contractOfftake_rule(b, cont):
            contract = self.rfp.get_contract(cont)
            b.volume = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
            b.in_port = Port()
            b.carrier = str(contract.parameters["consumes"])
            b.in_port.add(b.volume, "in_flow")
            b.price = contract.parameters.get("price", 0)
            b.target_frequency = contract.parameters.get("target_frequency", "yearly")
            b.shipment_frequency = contract.parameters.get("shipment_frequency", "monthly")
            # if b.target_frequency == "hourly":    
            #     def total_volume_rule(m):
            #         return sum(b.volume[t] for t in self.model.T) >= self.model.contract_target[cont]
            # b.total_volume_constraint = pyo.Constraint(rule=total_volume_rule)
        self.model.contractBlocks = pyo.Block(self.model.contracts, rule=contractOfftake_rule)

        def carrierBlock_rule(b, carr): # Create a block for each energy carrier to enforce nodal carrier balances
            carrier = self.rfp.get_carrier(carr)
            b.type = carrier.name
            b._in = []
            b._out = []
            for name, comp in self.rfp.get_components():
                if comp.parameters.get("produces") == b.type:
                    if comp.is_link:
                        b._in.append(self.model.linkBlocks[name].out_port)
                    elif comp.is_storage:
                        b._in.append(self.model.storageBlocks[name].out_port)
                    elif comp.is_supplier:
                        b._in.append(self.model.supplierBlocks[name].out_port)
                if comp.parameters.get("consumes") == b.type:
                    if comp.is_link:
                        b._out.append(self.model.linkBlocks[name].in_port)
                    elif comp.is_storage:
                        b._out.append(self.model.storageBlocks[name].in_port)
                    elif comp.is_offtaker:
                        b._out.append(self.model.offtakerBlocks[name].in_port)
                if b.type == "electricity":
                    ec = comp.parameters.get("charge_rate", 0) # Electricity consumption rate
                    if ec > 0:
                        if comp.is_link:
                            b._out.append(self.model.linkBlocks[name].in_port * ec) # Using a link consumes electricity (relevant for haber-bosch)
                        elif comp.is_storage:
                            b._out.append(self.model.storageBlocks[name].in_port * ec) # Charging a storage consumes electricity
            if b.type == "electricity":
                b.grid_sales = pyo.Var(self.model.T, domain=pyo.Reals) # Grid import/export variable
                b._out.append(Port(initialize={'out_flow': b.grid_sales})) # Positive values indicate selling to the grid, negative values indicate buying from the grid

            b.in_ports = Port(ports=b._in)
            b.out_ports = Port(ports=b._out)
            b.carrier_balances = Arc(b.in_ports, b.out_ports, self.model.T) # Carrier balance arcs
        self.model.carrierBlocks = pyo.Block(self.model.carriers, rule=carrierBlock_rule)

        # Ensure that grid capacity is not exceeded by supply imports
        def grid_capacity_rule(m, t):
            grid_capacity = self.rfp.get_component("GCP").parameters.get('capacity', 1000) # Assume a large capacity if not specified
            return (0,
                    sum(self.model.supplierBlocks[name].production[t] for name, comp in self.rfp.get_components() if comp.is_supplier) - \
                          self.model.carrierBlocks["electricity"].grid_sales[t],
                    grid_capacity)
        self.model.grid_capacity_constraint = pyo.Constraint(self.model.T, rule=grid_capacity_rule)

    def build_concrete_instance(self, data=None):
        self.inst = self.model.create_instance(data)

        def soc_rule(b):
            return b.soc[0] == self.model.init_soc + b.charge[0] - b.discharge[0]
        self.inst.init_soc_constraint = pyo.Constraint(self.model.storageBlocks, rule=soc_rule)

        def max_production_rule(supp, t):
            b = self.inst.supplierBlocks[supp]
            return b.production[t] <= self.model.supplier_cf[supp, t] * b.capacity
        self.inst.prodAvailConstraint = pyo.Constraint(self.model.suppliers, self.model.T, rule=max_production_rule)

        def hourly_target_rule(cont, t):
            b = self.inst.contractBlocks[cont]
            if b.target_frequency == "hourly":
                return b.volume[t] >= self.model.contract_target[cont]
        self.inst.hourly_volume_constraints = pyo.Constraint(self.model.contracts, self.model.T, rule=hourly_target_rule)

        def time_horizon_rule(cont):
            b = self.inst.contractBlocks[cont]
            if b.target_frequency != "hourly":
                return sum(b.volume[t] for t in self.model.T) >= self.model.contract_target[cont]
        self.inst.time_horizon_constraints = pyo.Constraint(self.model.contracts, rule=time_horizon_rule)

        self.set_objective()

    def _set_objective(self):
        def objective_rule(m):
            revenue = sum(self.model.carrierBlocks["electricity"].grid_sales[t] * self.model.electricity_price[t] for t in self.model.T)
            revenue += sum(self.model.contractBlocks[cont].volume[t] * self.model.contractBlocks[cont].price for cont in self.model.contracts for t in self.model.T)
            costs = sum(self.model.supplierBlocks[supp].production[t] * self.model.supplierBlocks[supp].parameters.get("price", 0) for supp in self.model.suppliers for t in self.model.T)
            return revenue - costs

    def run(self):
        self.result = self.model.solve()

    def print_solution(self):
        # 6. Inspect results
        print("Solver status:", self.result)
        print("Optimal value (profit):", self.model.objective_value)
        print("x_A:", self.variables)

if __name__ == "__main__":
    rfp = make_rfp()
    pfm = PerfectForesightModel(rfp, horizon=24)
    wind_cf = {t: 0.5 + 0.5 * np.sin(2 * np.pi * t / 24) for t in range(24)}
    solar_cf = {t: max(0, np.sin(2 * np.pi * (t - 6) / 24)) for t in range(24)}
    nuclear_cf = {t: 1.0 for t in range(24)}
    electricity_price = {t: 50 + 20 * np.sin(2 * np.pi * (t - 6) / 24) for t in range(24)}
    ammonia_price = {t: 300 + 50 * np.cos(2 * np.pi * (t - 12) / 24) for t in range(24)}
    hydrogen_price = {t: 100 + 30 * np.cos(2 * np.pi * (t - 12) / 24) for t in range(24)}
    data = {
        None: {
            'init_soc': {
                'Hydrogen Storage': 0,
                'Ammonia Storage': 0,
            },
            'contract_target': {
                'Hydrogen1': rfp.get_contract("Hydrogen1").parameters["volume"],
                'Ammonia1': rfp.get_contract("Ammonia1").parameters["volume"],
            },
            'supplier_cf': {
                'WindPower': wind_cf,
                'SolarPower': solar_cf,
                "NuclearPower": nuclear_cf,
            },
            'electricity_price': electricity_price,
            'ammonia_price': ammonia_price,
            'hydrogen_price': hydrogen_price,
        }
    }
    pfm.build_model(data=None)
    pfm.run()
    pfm.print_solution()