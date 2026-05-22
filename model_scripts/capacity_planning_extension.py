"""
Capacity Planning Extension for HourlyDeterministicLPModel
===========================================================

Summary of changes to convert the operational LP into a combined
operational + capacity planning model:

KEY DESIGN PRINCIPLE
--------------------
Pyomo variable `bounds` are evaluated at *construction time* and cannot
reference other Pyomo Vars. When `b.capacity` becomes a Var, every
`bounds=(0, b.capacity)` on flow variables must be replaced with:
  1. A static upper bound equal to `max_capacity` (the Var's own upper bound), AND
  2. An explicit linking constraint:  flow_var[t] <= b.capacity

This keeps the problem a pure LP (no bilinear terms).

WHAT GETS A CAPACITY VAR
-------------------------
  storageBlock   → b.capacity (bounds soc, in_flow, out_flow, elec_cons)
  ppaBlock       → b.capacity (bounds out_flow; multiplied in ppa_procurement_rule)
  dayaheadBlock  → b.capacity (bounds out_flow)
  linkBlock      → b.capacity (bounds in_flow, out_flow, elec_cons, ramp terms)
  offtakerBlock  → b.capacity (bounds in_flow, elec_cons)

WHAT STAYS FIXED
----------------
  contractBlock  → b.volume stays fixed (commercial term, not physical capacity)
  b.rate, b.ec   → conversion/efficiency coefficients stay fixed

NEW OBJECTIVE TERM
------------------
  Annualised CAPEX = sum over all components of:
    component.parameters.get("annualized_capex", 0) * b.capacity
  (user computes CRF × overnight_cost externally and stores as "annualized_capex")
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
from model_scripts.hourly_models import HourlyDeterministicLPModel

class CapacityPlanningModel(HourlyDeterministicLPModel):
    def __init__(self,
                rfp,
                inflexible: bool = False,
                enforce_rfnbo: bool = False,
                planning_horizon: int = 4 * 24,
                decision_horizon: int = 24,
                solver: str = 'scip',
                compute_duals: bool = False,
                allow_spot_buy: bool = True,
                documentation: bool = False,
                capacity_planning: bool = False,   # ← NEW
                discount_rate: float = 0.1,        # ← NEW (for objective)
                cvar_info: dict|None = None,
                **kwargs):
        super().__init__(rfp=rfp, inflexible=False, enforce_rfnbo=enforce_rfnbo,
                         planning_horizon=planning_horizon, decision_horizon=decision_horizon,
                         solver=solver, compute_duals=compute_duals,
                         allow_spot_buy=allow_spot_buy, guideline=None,
                         objective_logic=None, documentation=documentation,
                         **kwargs)
        if inflexible:
            print("Note: inflexible=True only enforces minimum load ratings.")
            self.min_load_active = True
        else:
            self.min_load_active = False
        self.capacity_planning = capacity_planning   # ← NEW: store flag
        self.discount_rate = discount_rate           # ← NEW: store discount rate for objective
        self.cvar_formulation = True if cvar_info is not None else False
        if self.cvar_formulation:
            self.cvar_alpha = cvar_info.get("alpha", 0.9)
            self.cvar_beta = cvar_info.get("beta", 0.5)

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

        self.model.init_contract_status = pyo.Param(self.model.contracts, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.init_soc             = pyo.Param(self.model.storages, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.prev_link_setpoints  = pyo.Param(self.model.links, within=pyo.NonNegativeReals, default=0.5, mutable=True)

        self.model.supplier_cf              = pyo.Param(self.model.ppas, self.model.T, within=pyo.NonNegativeReals, default=1, mutable=True)
        self.model.electricity_price        = pyo.Param(self.model.T, within=pyo.Reals, default=50, mutable=True)
        self.model.grid_emissions_intensity = pyo.Param(self.model.T, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.model.ets_price                = pyo.Param(self.model.T, within=pyo.Reals, default=0, mutable=True)
        
        # If extra spot deal shipment is needed:
        self.model.spot_shipment = pyo.Param(within=pyo.Binary, default=0, mutable=True)

        def carrierBlock_rule(b, carr):
            """ Create a block for each energy carrier to enable nodal carrier balance enforcement. """
            carrier = self.rfp.get_carrier(carr)
            b.type = carrier.name
            b.carrier_in = b.type
            b._in = {t: [] for t in self.model.T}
            b._out = {t: [] for t in self.model.T}
        self.model.carrierBlocks = pyo.Block(self.model.carriers, rule=carrierBlock_rule)

        def storageBlock_rule(b, stor):
            storage = self.rfp.get_component(stor)
            b._name      = storage.name
            b.ec         = storage.parameters.get("electricity_consumption", 0)
            b.carrier_in  = str(storage.parameters["consumes"])
            b.carrier_out = str(storage.parameters["produces"])
            b.rates         = {"in": storage.parameters.get("rate", 1), "out": storage.parameters.get("out_rate", 1)}

            if self.capacity_planning:
                max_cap  = storage.parameters.get("max_capacity", np.inf)
                b.capacity = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, max_cap))  # ← NEW
                # Flow variable upper bounds set to max_cap (static); linking constraints added later
                b.soc      = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_cap))
                b.in_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_cap))
                b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_cap))
                if b.ec > 0:
                    b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_cap * b.ec))
                    b.ec_constraint = pyo.Constraint(self.model.T, rule=lambda m, t: b.elec_cons[t] == b.in_flow[t] * b.ec)
            else:                                        # ← original behaviour preserved
                b.capacity = storage.parameters["capacity"]
                b.soc      = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
                b.in_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
                b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
                if b.ec > 0:
                    b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity * b.ec))
                    b.ec_constraint = pyo.Constraint(self.model.T, rule=lambda m, t: b.elec_cons[t] == b.in_flow[t] * b.ec)
        self.model.storageBlocks = pyo.Block(self.model.storages, rule=storageBlock_rule)

        def ppaBlock_rule(b, ppa):
            ppa_ = self.rfp.get_ppa(ppa)
            b._name       = ppa
            b.carrier_in  = str(ppa_.parameters["consumes"])
            b.carrier_out = str(ppa_.parameters["produces"])
            b.price       = ppa_.parameters.get('price')

            if self.capacity_planning:
                max_cap    = ppa_.parameters.get("max_capacity", np.inf)
                b.capacity = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, max_cap))  # ← NEW
                b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_cap))
                # ppa_procurement_rule already uses b.capacity multiplicatively → still linear ✓
            else:
                b.capacity = ppa_.parameters.get('capacity')
                b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
        self.model.ppaBlocks = pyo.Block(self.model.ppas, rule=ppaBlock_rule)

        def dayaheadBlock_rule(b, da):
            dayahead = self.rfp.get_component(da)
            b._name       = da
            b.carrier_in  = str(dayahead.parameters["consumes"])
            b.carrier_out = str(dayahead.parameters["produces"])

            if self.capacity_planning:
                max_cap    = dayahead.parameters.get("max_capacity", np.inf)
                b.capacity = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, max_cap))  # ← NEW
                # out_flow in (-capacity, capacity*allow_spot_buy); use (-max_cap, max_cap) statically
                b.out_flow = pyo.Var(self.model.T, domain=pyo.Reals,
                                    bounds=(-max_cap, max_cap * self.allow_spot_buy))
                b.bought_power = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                    bounds=(0, max_cap * self.allow_spot_buy))
                # Linking constraints added in _build_concrete_instance (see Section 6)
            else:
                b.capacity = dayahead.parameters.get('capacity')
                b.out_flow = pyo.Var(self.model.T, domain=pyo.Reals,
                                    bounds=(-b.capacity, b.capacity * self.allow_spot_buy))
                b.bought_power = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                    bounds=(0, b.capacity * self.allow_spot_buy))
            def bought_power_rule(m, t):
                return b.bought_power[t] >= b.out_flow[t]
            b.bought_power_constraint = pyo.Constraint(self.model.T, rule=bought_power_rule)
        self.model.dayaheadBlocks = pyo.Block(self.model.dayaheads, rule=dayaheadBlock_rule)

        def linkBlock_rule(b, lin):
            link = self.rfp.get_component(lin)
            b._name     = link.name
            b.rate      = link.parameters.get("rate", 1)
            assert b.rate > 0, f"Link {b._name} has non-positive conversion rate."
            b.ec        = link.parameters.get("electricity_consumption", 0)
            b.carrier_in  = str(link.parameters["consumes"])
            b.carrier_out = str(link.parameters["produces"])
            b.min_load    = link.parameters.get('min_load', 0) * self.min_load_active

            # Fractional ramp limits (stored as fractions of in_capacity; applied as constraints later)
            b._max_ramp_up_frac   = link.parameters.get('max_ramp_up',   1)
            b._max_ramp_down_frac = link.parameters.get('max_ramp_down', 1)
            b.reversible    = bool(link.parameters.get('reversible', False))

            if self.capacity_planning:
                max_cap    = link.parameters.get("max_capacity", np.inf)
                b.capacity = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, max_cap))   # ← NEW
                max_in_cap = max_cap / b.rate
                # Static upper bound = max_cap / rate; lower bound = 0 (min_load linked constraint added later)
                b.in_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_in_cap))
                b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_cap))
                if b.reversible:
                    b.reverse_in_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                                 bounds=(0, max_cap))
                    b.reverse_out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                                 bounds=(0, max_in_cap))
                if b.ec > 0:
                    b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_cap * b.ec))
                    b.ec_constraint = pyo.Constraint(
                        self.model.T, rule=lambda m, t: b.elec_cons[t] == b.ec * b.out_flow[t])
                # NOTE: efficiency_curve piecewise logic references b.max_electricity_consumption which
                # depends on capacity → that logic is inherently nonlinear when capacity is a Var.
                # For capacity planning mode the piecewise efficiency curve is intentionally disabled;
                # use a fixed average efficiency (b.ec) instead.
            else:
                # ── Original behaviour preserved exactly ──
                b.capacity      = link.parameters.get('capacity', np.inf)
                b.in_capacity   = b.capacity / b.rate
                b.max_ramp_up   = b._max_ramp_up_frac   * b.in_capacity
                b.max_ramp_down = b._max_ramp_down_frac * b.in_capacity
                b.in_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                    bounds=(b.min_load * b.in_capacity, b.in_capacity))
                b.out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                    bounds=(b.min_load * b.capacity, b.capacity))
                if b.reversible:
                    b.reverse_in_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                                 bounds=(b.min_load*b.capacity, b.capacity))
                    b.reverse_out_flow = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                                 bounds=(b.min_load*b.in_capacity, b.in_capacity))
                if b.ec > 0:
                    b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                        bounds=(0, b.capacity * b.ec))
                    b.ec_constraint = pyo.Constraint(
                        self.model.T, rule=lambda m, t: b.elec_cons[t] == b.ec * b.out_flow[t])
            b.conversion_constraint = pyo.Constraint(
                self.model.T, rule=lambda m, t: b.out_flow[t] == b.rate * b.in_flow[t])
            if b.reversible:
                def reverse_conversion_rule(m, t):
                    return b.reverse_in_flow[t] == b.rate * b.reverse_out_flow[t]
                b.reverse_conversion_constraint = pyo.Constraint(self.model.T, rule=reverse_conversion_rule)
        self.model.linkBlocks = pyo.Block(self.model.links, rule=linkBlock_rule)

        def offtakerBlock_rule(b, offt):
            offtaker = self.rfp.get_component(offt)
            b._name       = offtaker.name
            b.carrier_in  = str(offtaker.parameters["consumes"])
            b.carrier_out = str(offtaker.parameters["produces"])
            b.ec          = offtaker.parameters.get("electricity_consumption", 0)
            b.contracts   = pyo.Set(initialize=[cont.name for cont in offtaker.contracts])

            if self.capacity_planning:
                max_cap    = offtaker.parameters.get("max_capacity", np.inf)
                b.capacity = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, max_cap))  # ← NEW
                b.in_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_cap))
                if b.ec > 0:
                    b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, max_cap * b.ec))
                    b.ec_constraint = pyo.Constraint(
                        self.model.T, rule=lambda m, t: b.elec_cons[t] == b.ec * b.in_flow[t])
            else:
                b.capacity = offtaker.parameters.get('capacity')
                b.in_flow  = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.capacity))
                if b.ec > 0:
                    b.elec_cons = pyo.Var(self.model.T, domain=pyo.NonNegativeReals,
                                        bounds=(0, b.capacity * b.ec))
                    b.ec_constraint = pyo.Constraint(
                        self.model.T, rule=lambda m, t: b.elec_cons[t] == b.ec * b.in_flow[t])
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

            if not self.documentation:
                b.volume        = contract.parameters.get("volume")
                b.min_volume    = contract.parameters.get("min_volume", b.volume)
                b.max_volume    = contract.parameters.get("max_volume", b.volume)
                """ Physical flow of product to contract: """
                b.shipment = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, min(b.volume, b.offtaker_capacity)))
                if b.is_spot_contract == False:
                    # Bookkeeping of contract status and whether obligations are met.
                    b.contract_status = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.max_volume))
                    b.contract_shortfall = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.min_volume))
                    b.contract_slack = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, b.max_volume)) # Slack variable. Excess shipments are not awarded.
            else: # Not compatible with min and max volume contracts.
                b.volume = pyo.Var(domain=pyo.Reals)
                b.min_volume    = contract.parameters.get("min_volume", b.volume)
                b.max_volume    = contract.parameters.get("max_volume", b.volume)
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
        if self.cvar_formulation:
            self.inst.datetimes = pd.to_datetime([pyo.value(self.inst.T_datetime[t]) for t in self.inst.T])
            self.inst.years = self.inst.datetimes.year.unique().values
            self.inst.n_years = len(self.inst.datetimes.year.unique())
            self.inst.VaR   = pyo.Var(domain=pyo.Reals)
            self.inst.theta = pyo.Var(self.inst.years, domain=pyo.NonNegativeReals)
            self.inst.cvar_constraints = pyo.ConstraintList()
            for year in self.inst.years:
                T_set = [n for n, is_included in enumerate(self.inst.datetimes.year == year) if is_included]
                scenario_operational_earnings = self._cashflow_rule(self.inst, T=T_set)
                self.inst.cvar_constraints.add(self.inst.theta[year] >= self.inst.VaR - scenario_operational_earnings) # CVaR constraint

        super()._build_concrete_instance(data, already_instantiated=True)
        if self.capacity_planning:
            self._add_capacity_linking_constraints()

        # emissions = sum(self.inst.grid_emissions_intensity[t] * self.inst.dayaheadBlocks[da].bought_power[t] for da in self.inst.dayaheads for t in self.inst.T)
        # consumed_power = sum(self.inst.linkBlocks["Grid Connection Point"].in_flow[t] for t in self.inst.T)
        # emissions_intensity = emissions / consumed_power # tCO2/MWh -> gCO2/MJ
        # self.inst.emissions_intensity_constraint = pyo.Constraint()
        if hasattr(self.inst, 'soc_constraint'):
            self.inst.del_component('soc_constraint')
        def soc_rule(inst, stor, t): # Define intertemporal SOC logic
            b = inst.storageBlocks[stor]
            was_end_of_year = np.asarray(self._get_datetime_infos(inst, t - 1)).all() # If the last time step was the end of a year, reset.
            if t == 0 or was_end_of_year: # The initial SOC is externally given.
                return b.soc[t] == inst.init_soc[stor] + b.in_flow[t] * b.rates["in"] - b.out_flow[t] / b.rates["out"]
            else:
                return b.soc[t] == b.soc[t-1] + b.in_flow[t] * b.rates["in"] - b.out_flow[t] / b.rates["out"]
        self.inst.soc_constraint = pyo.Constraint(self.inst.storages, self.inst.T, rule=soc_rule)

    def _add_capacity_linking_constraints(self):
        """
        When capacity_planning=True, enforce   flow[t] <= capacity   for every
        component whose capacity is now a decision variable.  These are the
        constraints that were previously implicit in the Var bounds.
        """
        inst = self.inst

        # ── Storage ──────────────────────────────────────────────────────────────
        def stor_soc_cap(inst, stor, t):
            b = inst.storageBlocks[stor]
            return b.soc[t] <= b.capacity
        def stor_in_cap(inst, stor, t):
            b = inst.storageBlocks[stor]
            return b.in_flow[t] <= b.capacity
        def stor_out_cap(inst, stor, t):
            b = inst.storageBlocks[stor]
            return b.out_flow[t] <= b.capacity
        inst.stor_soc_cap_con = pyo.Constraint(inst.storages, inst.T, rule=stor_soc_cap)
        inst.stor_in_cap_con = pyo.Constraint(inst.storages, inst.T, rule=stor_in_cap)
        inst.stor_out_cap_con = pyo.Constraint(inst.storages, inst.T, rule=stor_out_cap)

        # ── PPA ──────────────────────────────────────────────────────────────────
        # ppa_procurement_rule already enforces out_flow == cf * capacity (linear); no extra needed.

        # ── Day-ahead ────────────────────────────────────────────────────────────
        def da_buy_cap(inst, da, t):
            b = inst.dayaheadBlocks[da]
            return b.out_flow[t] <= b.capacity * int(self.allow_spot_buy)
        def da_sell_cap(inst, da, t):
            b = inst.dayaheadBlocks[da]
            return b.out_flow[t] >= -b.capacity
        inst.da_buy_cap_con  = pyo.Constraint(inst.dayaheads, inst.T, rule=da_buy_cap)
        inst.da_sell_cap_con = pyo.Constraint(inst.dayaheads, inst.T, rule=da_sell_cap)

        # ── Links ─────────────────────────────────────────────────────────────────
        def link_out_cap(inst, link, t):
            b = inst.linkBlocks[link]
            return b.out_flow[t] <= b.capacity
        def link_in_cap(inst, link, t):
            b = inst.linkBlocks[link]
            return b.in_flow[t] <= b.capacity / b.rate
        inst.link_out_cap_con = pyo.Constraint(inst.links, inst.T, rule=link_out_cap)
        inst.link_in_cap_con  = pyo.Constraint(inst.links, inst.T, rule=link_in_cap)
        def link_reverse_out_cap(inst, link, t):
            b = inst.linkBlocks[link]
            if b.reversible:
                return b.reverse_out_flow[t] <= b.capacity / b.rate
            else:
                return pyo.Constraint.Skip
        def link_reverse_in_cap(inst, link, t):
            b = inst.linkBlocks[link]
            if b.reversible:
                return b.reverse_in_flow[t] <= b.capacity
            else:
                return pyo.Constraint.Skip
        inst.link_reverse_out_cap_con = pyo.Constraint(inst.links, inst.T, rule=link_reverse_out_cap)
        inst.link_reverse_in_cap_con  = pyo.Constraint(inst.links, inst.T, rule=link_reverse_in_cap)

        # Min-load lower bound (replaces lower bound on in_flow / out_flow)
        def link_min_in(inst, link, t):
            b = inst.linkBlocks[link]
            if b.min_load == 0:
                return pyo.Constraint.Skip
            return b.in_flow[t] >= b.min_load * b.capacity / b.rate
        def link_min_out(inst, link, t):
            b = inst.linkBlocks[link]
            if b.min_load == 0:
                return pyo.Constraint.Skip
            return b.out_flow[t] >= b.min_load * b.capacity
        inst.link_min_in_con  = pyo.Constraint(inst.links, inst.T, rule=link_min_in)
        inst.link_min_out_con = pyo.Constraint(inst.links, inst.T, rule=link_min_out)

        # ── Offtakers ─────────────────────────────────────────────────────────────
        def offt_in_cap(inst, offt, t):
            b = inst.offtakerBlocks[offt]
            return b.in_flow[t] <= b.capacity
        inst.offt_in_cap_con = pyo.Constraint(inst.offtakers, inst.T, rule=offt_in_cap)

    def _cvar_cashflow_rule(self, inst):
        exp_obj = (1-self.cvar_beta) * self._cashflow_rule(inst)        
        cvar_obj = self.cvar_beta * (inst.VaR - 
                                     1/(1-self.cvar_alpha) * sum(inst.theta[year] for year in inst.years) / inst.n_years
                                     )
        return exp_obj + cvar_obj

    def _cashflow_rule(self, inst, T=None):
        T = T if T is not None else inst.T
        """ Revenues of the RFP (contract payments happen when shipments happen) """
        revenue = sum(b.shipment[t] * b.price for name, b in inst.contractBlocks.items() for t in T)

        """ Costs of the RFP (PPA costs not included as they are exogenously fixed) """
        costs = self._get_electricity_objective_cost(inst, T)
        if self.capacity_planning:
            costs += self._get_ppa_cost(inst, T)

        for cont in inst.contracts: # Penalties of not meeting contract obligations:
            b = inst.contractBlocks[cont]
            if b.is_spot_contract == False:
                costs += sum(b.contract_shortfall[t] * b.penalty for t in T)
        
        """ Maximize profits """
        annual_earnings = (revenue - costs) / (len(T) / (365.25 * 24))  # Scale to objective time horizon (e.g. 1 year)
        return annual_earnings

    def _set_objective(self):
        def objective_rule(inst):
            if self.cvar_formulation:
                obj = self._cvar_cashflow_rule(inst)
            else:
                obj = self._cashflow_rule(inst)
            obj -= self._capex_cost(inst)          # ← NEW: subtract annualised CAPEX
            return obj

        self.inst.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    def _get_ppa_cost(self, inst, T=None):
        """
        Cost of electricity procured from PPAs.
        In the operational model this was a fixed cost (exogenous) and excluded
        from the objective. In capacity planning mode the capacity is a decision
        variable, so the cost becomes variable and must be included.
        """
        T = T if T is not None else inst.T
        return sum(
            inst.ppaBlocks[ppa].out_flow[t] * inst.ppaBlocks[ppa].price
            for ppa in inst.ppas
            for t in T
        )

    def _capex_cost(self, inst):
        """
        Annualised capital cost for all capacity decision variables.

        Each component should carry an "capital_cost" parameter (currency / MW
        or currency / MWh for storage). 
        The model multiplies this by a capital recovery factor (CRF) to get
            annualized_capex = overnight_capex_per_unit × CRF
        where CRF = r(1+r)^n / ((1+r)^n - 1).

        Components that have no "capital_cost" key contribute zero cost,
        so the model degrades gracefully for components not yet costed.
        """
        if not self.capacity_planning:
            return 0
        project_lifetime = 25 # years; used as default if component lifetime not specified
        def crf(lifetime):
            return self.discount_rate * (1 + self.discount_rate) ** lifetime / ((1 + self.discount_rate) ** lifetime - 1)
        cost = 0
        componentBlocks = {**inst.storageBlocks, **inst.dayaheadBlocks, **inst.linkBlocks, **inst.offtakerBlocks}
        for name, b in componentBlocks.items():
            cc = self.rfp.get_component(name).parameters.get("capital_cost", 0)
            fom = self.rfp.get_component(name).parameters.get("fixed_operational_cost", 0)
            lifetime = self.rfp.get_component(name).parameters.get("lifetime", project_lifetime)
            annualized_capex = crf(lifetime) * cc
            cost += (annualized_capex + fom) * b.capacity
        return cost

    def save_optimal_capacities(self, filename):
        """
        Save optimal capacities to a CSV file for later use in operational model.
        """
        if not self.capacity_planning:
            raise ValueError("Model was not run in capacity planning mode.")
        inst = self.inst
        data = []
        for stor in inst.storages:
            b = inst.storageBlocks[stor]
            data.append({'component': stor, 'type': 'storage', 'optimal_capacity': pyo.value(b.capacity)})
        for da in inst.dayaheads:
            b = inst.dayaheadBlocks[da]
            data.append({'component': da, 'type': 'dayahead', 'optimal_capacity': pyo.value(b.capacity)})
        for link in inst.links:
            b = inst.linkBlocks[link]
            data.append({'component': link, 'type': 'link', 'optimal_capacity': pyo.value(b.capacity)})
        for offt in inst.offtakers:
            b = inst.offtakerBlocks[offt]
            data.append({'component': offt, 'type': 'offtaker', 'optimal_capacity': pyo.value(b.capacity)})
        for ppa in inst.ppas:
            b = inst.ppaBlocks[ppa]
            data.append({'component': ppa, 'type': 'ppa', 'optimal_capacity': pyo.value(b.capacity)})
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def save_capacity_utilization_factors(self, filename):
        actions = self.get_actions()
        average_ammonia_production = actions["ammonia_production"].mean()
        cf_nh3 = average_ammonia_production / self.inst.linkBlocks["Haber Bosch Plant"].capacity.value
        print("Haber Bosch capacity utilization factor: ", cf_nh3)
        average_hydrogen_production = actions["hydrogen_production"].mean()
        cf_h2 = average_hydrogen_production / self.inst.linkBlocks["Electrolyzer"].capacity.value
        print("Electrolyzer capacity utilization factor: ", cf_h2)
        power_flow = [pyo.value(self.inst.linkBlocks['Grid Connection Point'].out_flow[t]) for t in self.inst.T]
        # power_outflow = [pyo.value(self.inst.linkBlocks['Grid Connection Point'].reverse_in_flow[t]) for t in self.inst.T]
        # power_flow = np.asarray(power_inflow) + np.asarray(power_outflow)
        average_power_flow = np.mean(power_flow)
        cf_gcp = average_power_flow / self.inst.linkBlocks['Grid Connection Point'].capacity.value
        print("Transformer capacity utilization factor: ", cf_gcp)
        # power_inflow = [pyo.value(self.inst.linkBlocks['Battery Inverter'].out_flow[t]) for t in self.inst.T]
        # power_outflow = [pyo.value(self.inst.linkBlocks['Battery Inverter'].reverse_in_flow[t]) for t in self.inst.T]
        # power_flow = np.asarray(power_inflow) + np.asarray(power_outflow)
        # average_power_flow = np.mean(power_flow)
        # if self.inst.linkBlocks['Battery Inverter'].capacity.value > 0:
        #     cf_bess = average_power_flow / self.inst.linkBlocks['Battery Inverter'].capacity.value
        # else:
        cf_bess = -1
        print("Battery Inverter capacity utilization factor: ", cf_bess)
        pd.DataFrame(index=[0], data={"Haber Bosch":cf_nh3, "Electrolyzer": cf_h2, "GCP": cf_gcp, "BESS": cf_bess}).to_csv(filename)
