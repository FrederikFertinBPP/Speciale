import pandas as pd
import os

class Component():
    def __init__(self, name:str, parameters:dict = {}, notes:str = "") -> None:
        """
        Initialize a component of the hybrid power plant.
        Args:
            name (str): The name of the component.
            parameters (dict, optional): Parameters for the component.
        """
        self.name = name
        self.parameters = parameters
        self.notes = notes
        self.type = parameters.get("type")

class PhysicalUnit(Component):
    type_options = ("grid", "link", "storage", "supplier", "offtaker")

    def __init__(self, name:str, parameters:dict = {}, ppa = None, offtake_contracts = None, notes:str = "") -> None:
        """
        Initialize a unit of the hybrid power plant.
        Args:
            name (str): The name of the unit.
            parameters (dict, optional): Parameters for the unit.
        """
        super().__init__(name, parameters, notes)
        if "type" in parameters and parameters["type"] not in self.type_options:
            raise ValueError(f"Invalid type '{parameters['type']}' for unit '{name}'. Valid options are: {self.type_options}")
        self.is_grid     = self.type == "grid"
        self.is_link     = self.type == "link"
        self.is_storage  = self.type == "storage"
        self.is_supplier = self.type == "supplier"
        self.is_offtaker = self.type == "offtaker"
        if ppa is not None:
            self.ppa = ppa
        if offtake_contracts is not None:
            self.contracts = offtake_contracts

class Contract(Component):
    frequency_options = ("hourly", "daily", "monthly", "yearly")
    frequency_period_map = {"hourly" : 1, "daily" : 24, "monthly" : 30*24, "yearly" : 8760}

    def __init__(self, name:str, parameters:dict = {}, notes:str = "") -> None:
        """
        Initialize a contract for the renewable fuel plant.
        Args:
            name (str): The name of the contract.
            parameters (dict, optional): Parameters for the contract.
        """
        super().__init__(name, parameters, notes)
        _tf = parameters.get("target_frequency")
        self.target_frequency = _tf if _tf in self.frequency_options else None

class PPA(Component):
    types = ["wind", "solar", "nuclear"]

    def __init__(self, name:str, parameters:dict = {}, notes:str = "") -> None:
        """
        Initialize a PPA for the renewable fuel plant.
        Args:
            name (str): The name of the PPA.
            parameters (dict, optional): Parameters for the PPA.
        """
        super().__init__(name, parameters, notes)
        self.simulate_profile = bool(parameters.get("simulated"))

class Carrier():
    def __init__(self, name:str) -> None:
        self.name = name

class RenewableFuelPlant():
    uncertainties = ["wind", "solar", "price"]

    def __init__(self) -> None:
        self.components = {}
        self.contracts = {}
        self.carriers = {}
        self.ppas = {}

    def add_ppa(self, ppa):
        """
        Add a ppa to the hybrid power plant system.
        Args:
            ppa (object): The ppa to add to the system.
        """
        self.ppas[ppa.name] = ppa

    def get_ppa(self, name):
        """
        Get a ppa by its name.
        Args:
            name (str): The name of the ppa to retrieve.
        Returns:
            object: The contract with the specified name, or None if not found.
        """
        return self.ppas.get(name, None)

    def get_ppas(self):
        """
        Get the dict of ppas in the hybrid power plant system.
        Returns:
            dict: A dict of ppas in the system.
        """
        return self.ppas.items()

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
        Get the dict of contracts in the hybrid power plant system.
        Returns:
            dict: A dict of contracts in the system.
        """
        return self.contracts.items()

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
        Get the dict of components in the hybrid power plant system.
        Returns:
            dict: A dict of components in the system.
        """
        return self.components.items()
    
    def add_carrier(self, carrier):
        """
        Add a carrier to the hybrid power plant system.
        Args:
            carrier (object): The carrier to add to the system.
        """
        self.carriers[carrier.name] = carrier
    
    def get_carrier(self, name):
        """
        Get a carrier by its name.
        Args:
            name (str): The name of the carrier to retrieve.
        Returns:
            object: The carrier with the specified name, or None if not found.
        """
        return self.carriers.get(name, None)
    
    def get_carriers(self):
        """
        Get the dict of carriers in the hybrid power plant system.
        Returns:
            dict: A dict of carriers in the system.
        """
        return self.carriers.items()
    
    def get_annual_contracts(self):
        """
        Get a list of contracts with yearly target frequency.
        Returns:
            list: A list of contracts with yearly target frequency.
        """
        return [cont for name, cont in self.get_contracts() if cont.target_frequency == "yearly"]
    
    def get_monthly_contracts(self):
        """
        Get a list of contracts with monthly target frequency.
        Returns:
            list: A list of contracts with monthly target frequency.
        """
        return [cont for name, cont in self.get_contracts() if cont.target_frequency == "monthly"]

def load_input_data():
    xls_path = os.path.abspath('./setup_files/hpp_layout.xlsx')
    df_components = pd.read_excel(xls_path, sheet_name="Components")
    df_contracts = pd.read_excel(xls_path, sheet_name="Contracts")
    df_carriers = pd.read_excel(xls_path, sheet_name="Carriers")
    df_ppas = pd.read_excel(xls_path, sheet_name="PPAs")
    return df_components, df_contracts, df_carriers, df_ppas

def create_rfp():
    """
    Create a renewable fuel plant with defined components (i.e. fixed capacities and such) and contracts.
    """
    rfp = RenewableFuelPlant()
    df_components, df_contracts, df_carriers, df_ppas = load_input_data()
    
    for _, row in df_carriers.iterrows():
        rfp.add_carrier(Carrier(name=row["name"]))
    for _, row in df_contracts.iterrows():
        parameters = {k: v for k, v in row.items() if pd.notna(v) and k not in ["name", "notes"]}
        rfp.add_contract(Contract(name=row["name"], parameters=parameters, notes=row.get("notes", "")))
    for _, row in df_ppas.iterrows():
        parameters = {k: v for k, v in row.items() if pd.notna(v) and k not in ["name", "notes"]}
        rfp.add_ppa(PPA(name=row["name"], parameters=parameters, notes=row.get("notes", "")))
    for _, row in df_components.iterrows():
        parameters = {k: v for k, v in row.items() if pd.notna(v) and k not in ["name", "notes"]}
        ppa = None
        offtake_contracts = None
        if parameters["type"] == "supplier":
            ppa = rfp.get_ppa(row["name"])
        elif parameters["type"] == "offtaker":
            offtake_contracts = [cont for name, cont in rfp.get_contracts() if cont.type == parameters["produces"]]
            for cont in offtake_contracts:
                cont.offtaker = row['name']
        rfp.add_component(PhysicalUnit(name=row["name"], parameters=parameters, ppa=ppa, offtake_contracts=offtake_contracts, notes=row.get("notes", "")))
    
    return rfp

# if __name__ == "__main__":
#     create_rfp()  # Test the function to ensure it runs without errors