from Speciale.RFP_initialization import RenewableFuelPlant, make_rfp
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
        self.build_model()

    def build_model(self):
        # 1. Initialize the model
        self.model = pyo.AbstractModel()
        self.model.T = pyo.RangeSet(0, self.horizon - 1)  # Time steps
        self.model.carriers = pyo.Set(initialize=[name for name, carr in self.rfp.get_carriers()])

        self.model.links        = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_link])
        self.model.storages     = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_storage])
        self.model.suppliers    = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_supplier])
        self.model.offtakers    = pyo.Set(initialize=[name for name, comp in self.rfp.get_components() if comp.is_offtaker])

        self.model.contracts    = pyo.Set(initialize=[name for name, cont in self.rfp.get_contracts()])

        # self.model.store_cap    = pyo.Param(self.model.storages,  initialize={name: comp.parameters["capacity"] for name, comp in self.rfp.get_components() if comp.is_storage})
        self.model.supplier_cf  = pyo.Param(self.model.suppliers, self.model.T)

        def storageBlock_rule(b, stor):
            storage = self.rfp.get_component(stor)
            b.soc = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, storage.parameters["capacity"]))
            b.charge = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, storage.parameters["capacity"]))
            b.discharge = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, storage.parameters["capacity"]))
            b.in_port = Port()
            b.out_port = Port()
            b.carrier_in = str(storage.parameters["consumes"])
            b.carrier_out = str(storage.parameters["produces"])
            b.ec = storage.parameters.get("charge_rate", 0)
            def soc_rule(m, t):
                if t == 0:
                    return b.soc[t] == storage.parameters.get("initial_soc", 0) + b.charge[t] - b.discharge[t]
                else:
                    return b.soc[t] == b.soc[t-1] + b.charge[t] - b.discharge[t]
            b.soc_constraint = pyo.Constraint(self.model.T, rule=soc_rule)
            b.in_port.add(b.charge, "in_flow")
            b.out_port.add(b.discharge, "out_flow")
        self.model.storageBlocks = pyo.Block(self.model.storages, rule=storageBlock_rule)

        def supplierBlock_rule(b, supp):
            supplier = self.rfp.get_component(supp)
            b.out_port = Port()
            b.production = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, supplier.parameters.get('capacity', np.inf)))
            b.carrier_in = str(supplier.parameters["consumes"])
            b.carrier_out = str(supplier.parameters["produces"])
            def max_production_rule(m, t):
                return b.production[t] <= self.model.supplier_cf[supp, t] * supplier.parameters.get('capacity', np.inf)
            b.CapConstraint = pyo.Constraint(self.model.T, rule=max_production_rule)
            b.out_port.add(b.production, "out_flow")
        self.model.supplierBlocks = pyo.Block(self.model.suppliers, rule=supplierBlock_rule)
        
        def offtakerBlock_rule(b, offt):
            offtaker = self.rfp.get_component(offt)
            b.in_port = Port()
            b.consumption = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, offtaker.parameters.get('capacity', np.inf)))
            b.carrier_in = str(offtaker.parameters["consumes"])
            b.carrier_out = str(offtaker.parameters["produces"])
            b.in_port.add(b.consumption, "in_flow")
        self.model.offtakerBlocks = pyo.Block(self.model.offtakers, rule=offtakerBlock_rule)

        def linkBlock_rule(b, lin):
            link = self.rfp.get_component(lin)
            b.in_port = Port()
            b.out_port = Port()
            b.in_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, link.parameters.get('capacity', np.inf)))
            b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, link.parameters.get('capacity', np.inf)))
            def conversion_rule(m, t):
                return b.out_flow[t] == link.parameters.get("rate", 1) * b.in_flow[t]
            b.conversion_constraint = pyo.Constraint(self.model.T, rule=conversion_rule)
            b.carrier_in = str(link.parameters["consumes"])
            b.carrier_out = str(link.parameters["produces"])
            b.ec = link.parameters.get("charge_rate", 0)
            b.in_port.add(b.flow, "in_flow")
            b.out_port.add(b.flow, "out_flow")
        self.model.linkBlocks = pyo.Block(self.model.links, rule=linkBlock_rule)

        def carrierBlock_rule(b, carr):
            carrier = self.rfp.get_carrier(carr)
            b.inflow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
            b.outflow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
            b.type = carrier.name
            ports_in = [] ### From here we need to work ###
            ports_out = []
            for stor in self.model.storages:
                if self.model.storageBlocks[stor].carrier_in == carr:
                    ports_in.append(self.model.storageBlocks[stor].in_port)
                if self.model.storageBlocks[stor].carrier_out == carr:
                    ports_out.append(self.model.storageBlocks[stor].out_port)
            for supp in self.model.suppliers:
                if self.model.supplierBlocks[supp].carrier_out == carr:
                    ports_out.append(self.model.supplierBlocks[supp].out_port)
            for offt in self.model.offtakers:
                if self.model.offtakerBlocks[offt].carrier_in == carr:
                    ports_in.append(self.model.offtakerBlocks[offt].in_port)
            for lin in self.model.links:
                if self.model.linkBlocks[lin].carrier_in == carr:
                    ports_in.append(self.model.linkBlocks[lin].in_port)
                if self.model.linkBlocks[lin].carrier_out == carr:
                    ports_out.append(self.model.linkBlocks[lin].out_port)
        self.model.carrierBlocks = pyo.Block(self.model.carriers, rule=carrierBlock_rule)

        cm = self.model.create_instance()

        # Create Arcs to connect the ports through their respective carriers
        for carr in self.model.carriers:
            ports_in = []
            ports_out = []
            for stor in self.model.storages:
                if self.model.storageBlocks[stor].carrier_in == carr:
                    ports_in.append(self.model.storageBlocks[stor].in_port)
                if self.model.storageBlocks[stor].carrier_out == carr:
                    ports_out.append(self.model.storageBlocks[stor].out_port)
            for supp in self.model.suppliers:
                if self.model.supplierBlocks[supp].carrier_out == carr:
                    ports_out.append(self.model.supplierBlocks[supp].out_port)
            for offt in self.model.offtakers:
                if self.model.offtakerBlocks[offt].carrier_in == carr:
                    ports_in.append(self.model.offtakerBlocks[offt].in_port)
            for lin in self.model.links:
                if self.model.linkBlocks[lin].carrier_in == carr:
                    ports_in.append(self.model.linkBlocks[lin].in_port)
                if self.model.linkBlocks[lin].carrier_out == carr:
                    ports_out.append(self.model.linkBlocks[lin].out_port)
            for p_in in ports_in:
                for p_out in ports_out:
                    arc = Arc(source=p_out, destination=p_in)
                    setattr(self.model, f"arc_{carr}_{len(self.model.component_objects(Arc))}", arc)



        # 2. Define variables
        self.add_variables()

        # 3. Define constraints
        self.add_constraints()

        # 4. Define objective
        self.set_objective()

    def add_variables(self):
        self.variables['action'] = self.model.add_variables(lower=self.action_space.low, upper=self.action_space.high)
        
        self.variables['state'] = self.model.add_variables(lower=self.observation_space.low, upper=self.observation_space.high)

        """
        self.variables["contract_offtake"] = {}
        for name, contract in self.hpp.get_contracts():
            self.variables["contract_offtake"][name] = self.model.add_variables(lb=0, name=f"contract_offtake_{name}", shape=self.horizon)
        self.variables["production"] = {}
        self.variables["storage"] = {}
        for name, component in self.hpp.get_components():
            if component.is_producer:
                self.variables["production"][name] = self.model.add_variables(lb=0, name=f"production_{name}", shape=self.horizon)
            if component.is_storage:
                self.variables["storage"][name] = self.model.add_variables(lb=0, name=f"storage_{name}", shape=self.horizon)
        x_A = self.model.add_variables(lb=0, name="x_A")  # units of product A
        x_B = self.model.add_variables(lb=0, name="x_B")  # units of product B
        """

    def add_constraints(self):
        # Get the stochastic inputs for the environment
        p_wind, p_solar, price_el = self.env.realize_stochasticity(self.horizon)
        init_state = self.env.reset()[0][:,-1]  # Current state of the environment
        self._update_constraints(init_state, p_wind, p_solar, price_el)

    def _update_constraints(self, init_state, p_wind, p_solar, price_el):
        for t in range(self.horizon):
            if t == 0:
                soc_h2_, soc_nh3_, ytd_nh3_, = init_state  # Current state of the environment
            else:
                # Current state of the environment
                soc_h2_, soc_nh3_, ytd_nh3_, = (self.variables['state'].at[t-1,i] for i in range(len(init_state)))
            # Next state of the environment
            soc_h2, soc_nh3, ytd_nh3 = (self.variables['state'].at[t,i] for i in range(len(init_state)))
            # Actions taken by the agent
            # (1) Amount of electricity to sell to the grid
            # (2) Power flowing to the electrolyzer
            # (3) Amount of hydrogen to sell through the H2 pipeline
            # (4) Amount of ammonia to store
            p_to_grid, p_electrolyzer, m_h2_to_pipeline, m_nh3_to_storage = self.variables['action'][t,:]
            self.model.add_constraints(soc_nh3 == m_nh3_to_storage + soc_nh3_, name=f"nh3_storage{t}") # Update the state of charge of ammonia
            
            kappa_hb = self.hpp.get_component("Haber-Bosch Plant").parameters["rate"]
            m_h2_to_hb = m_nh3_to_storage * kappa_hb  # Amount of hydrogen to send to the Haber-Bosch plant
            
            eta_electrolyzer = self.hpp.get_component("Electrolyzer").parameters["rate"]  # Efficiency of the electrolyzer [t H2/MWh]
            m_h2_from_electrolyzer = p_electrolyzer * eta_electrolyzer  # Amount of hydrogen produced by the electrolyzer
            m_h2_to_storage = m_h2_from_electrolyzer - m_h2_to_pipeline - m_h2_to_hb # Amount of hydrogen to store in the H2 storage

            eta_hb = self.hpp.get_component("Haber-Bosch Plant").parameters["rate2"]  # Efficiency of the Haber-Bosch plant [MWh/t NH3]
            p_to_hb = m_nh3_to_storage * eta_hb  # Amount of electricity to send to the Haber-Bosch plant
            p_compressor = (m_h2_to_storage > 0) * m_h2_to_storage * self.hpp.get_component("Hydrogen Storage").parameters["storage"]["charge"]["rate"]  # Power consumed by the compressor to fill the H2 storage. No power used if the storage is discharged.
            p_balancing = p_wind + p_solar - p_to_grid - p_to_hb - p_compressor
            
            # Update the state of the environment based on the actions taken
            if self.time % (30*24) == 0 and self.time > 0:  # Every 30 days, a ship comes to pick up ammonia
                m_nh3_to_ship = min(self.hpp.get_component("Ammonia Shipment").parameters["capacity"], soc_nh3)  # Amount of ammonia to ship every 30 days
                soc_nh3 -= m_nh3_to_ship  # Update the state of charge of ammonia
                ytd_nh3 += m_nh3_to_ship  # Update the year-to-date ammonia delivery

            self.state = np.asarray([soc_h2, soc_nh3, ytd_nh3], dtype=np.float64)

            # Calculate the reward based on the actions taken
            if p_balancing < 0: # If we are buying unplanned electricity from the grid
                balancing_reward = p_balancing * price_el * 1.5 # Added cost for buying electricity
            else: # If we are selling unplanned electricity to the grid
                balancing_reward = p_balancing * price_el * 0.5 # Reduced reward for selling electricity to the grid

            h2_contract_penalty = max(0, # Penalty for not meeting the hydrogen contract
                                        self.hpp.get_contract("Hydrogen Offtake Agreement").parameters["penalty"] * 
                                        (self.hpp.get_contract("Hydrogen Offtake Agreement").parameters["volume"] - m_h2_to_pipeline))
            # self.model.add_constraints(2 , name="labor")
            # self.model.add_constraints(1 * x_A + 2 * x_B <= 80, name="material")
        pass

    def set_objective(self):
        self.model.add_objective(3, sense="max")

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
    pfm.run()
    pfm.print_solution()