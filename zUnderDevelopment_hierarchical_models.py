from common_scripts.RFP_initialization import RenewableFuelPlant, create_rfp
from data_scripts import DataForecaster

class MonthlyHeuristicPlanner:

    def __init__(self, rfp : RenewableFuelPlant):
        self.rfp = rfp
        self.contracts = [contract for name, contract in self.rfp.get_contracts() if contract.target_frequency == "yearly"]
        self.prices = None
        self.targets = [None] * 12
    
    def set_targets(self, forecast_prices):
        pass
    
    def update_targets(self, ):
        # _update_prices(self)
        pass


class InputDataConstructor:
    """ The constructor of the input data for the unknown parameters of hourly Abstract pyomo model.  """
    def __init__(self, rfp:RenewableFuelPlant, forecaster:DataForecaster, horizon=4*24):
        self.rfp = rfp
        self.forecaster = forecaster
        self.horizon = horizon
    
    def build_next_data_dict(self, previous_results, time):
        self.time = time # Time of year, possibly a datetime object to have full info
        self.r = previous_results
        self._get_capacity_factors()
        self._forecast_capacity_factors()
        self._forecast_electricity_prices()
        self._construct_input_data()
        return self.data

    def _get_contract_targets(self):
        self.contract_targets = {name : cont.parameters.get("volume", 0) for name, cont in self.rfp.get_contracts()}
        pass

    def _get_capacity_factors(self):
        wind = self.forecaster.load_24_wind_realizations()
        solar = self.forecaster.load_24_solar_realizations()
        self.wind_cf = {("WindPower",t) : wind[t] for t in range(24)}
        self.wind_cf = {("SolarPower",t) : solar[t] for t in range(24)}
        self.nuclear_cf = {("NuclearPower",t) : 1.0 for t in range(self.horizon)}

    def _forecast_capacity_factors(self):
        wind = self.forecaster.forecast_wind(self.time, self.horizon)
        solar = self.forecaster.forecast_solar(self.time, self.horizon)
        self.wind_cf = {**self.wind_cf, **{("WindPower",t+24) : wind[t] for t in range(self.horizon-24)}}
        self.solar_cf = {**self.solar_cf, **{("SolarPower",t+24) : solar[t] for t in range(self.horizon-24)}}

    def _forecast_electricity_prices(self):
        self.electricity_prices = self.forecaster.simulate_prices(self.time, self.horizon)
        
    def _construct_input_data(self):
        self.data = {
            None: {
                'init_soc': self.r.final_soc,
                'contract_target': self.contract_targets,
                'supplier_cf': {
                    **self.wind_cf,
                    **self.solar_cf,
                    **self.nuclear_cf,
                },
                'electricity_price': self.electricity_prices,
            }
        }


if __name__ == "__main__":
    rfp = create_rfp()
    rolling_horizon = 4 * 24
    step_horizon = 24

    forecaster = DataForecaster(from_pickle=True, cache_id="test1")
    forecaster = forecaster.unpickle()
    constructor = InputDataConstructor(rfp, forecaster, horizon=rolling_horizon)