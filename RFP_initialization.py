import numpy as np
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

class PhysicalUnit(Component):
    type_options = ("link", "storage", "supplier", "offtaker")

    def __init__(self, name:str, parameters:dict = {}, notes:str = "") -> None:
        """
        Initialize a unit of the hybrid power plant.
        Args:
            name (str): The name of the unit.
            parameters (dict, optional): Parameters for the unit.
        """
        super().__init__(name, parameters, notes)
        if "type" in parameters and parameters["type"] not in self.type_options:
            raise ValueError(f"Invalid type '{parameters['type']}' for unit '{name}'. Valid options are: {self.type_options}")
        self.type = parameters.get("type")
        self.is_link     = self.type == "link"
        self.is_storage  = self.type == "storage"
        self.is_producer = self.type == "producer"
        self.is_supplier = self.type == "supplier"
        self.is_offtaker = self.type == "offtaker"

class Contract(Component):
    frequency_options = ("hourly", "daily", "monthly", "yearly")

    def __init__(self, name:str, parameters:dict = {}, notes:str = "") -> None:
        """
        Initialize a contract for the hybrid power plant.
        Args:
            name (str): The name of the contract.
            parameters (dict, optional): Parameters for the contract.
        """
        super().__init__(name, parameters, notes)
        if "frequency" in parameters and parameters["frequency"] not in self.frequency_options:
            raise ValueError(f"Invalid frequency '{parameters['frequency']}' for contract '{name}'. Valid options are: {self.frequency_options}")
        self.frequency = parameters.get("frequency")

class Carrier():
    def __init__(self, name:str) -> None:
        self.name = name

class RenewableFuelPlant():
    def __init__(self) -> None:
        self.components = {}
        self.contracts = {}
        self.carriers = {}

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

def load_input_data():
    cwd = os.path.dirname(os.path.abspath(__file__))
    xls_path = os.path.join(cwd, 'input files/hpp_layout.xlsx')
    df_components = pd.read_excel(xls_path, sheet_name="Components")
    df_contracts = pd.read_excel(xls_path, sheet_name="Contracts")
    df_carriers = pd.read_excel(xls_path, sheet_name="Carriers")
    return df_components, df_contracts, df_carriers

def make_rfp():
    """
    Create a renewable fuel plant with defined components (i.e. fixed capacities and such) and contracts.
    """
    rfp = RenewableFuelPlant()
    df_components, df_contracts, df_carriers = load_input_data()
    
    for _, row in df_components.iterrows():
        parameters = {k: v for k, v in row.items() if pd.notna(v) and k not in ["name", "notes"]}
        rfp.add_component(PhysicalUnit(name=row["name"], parameters=parameters, notes=row.get("notes", "")))
    for _, row in df_contracts.iterrows():
        parameters = {k: v for k, v in row.items() if pd.notna(v) and k not in ["name", "notes"]}
        rfp.add_contract(Contract(name=row["name"], parameters=parameters, notes=row.get("notes", "")))
    for _, row in df_carriers.iterrows():
        rfp.add_carrier(Carrier(name=row["name"]))
    
    return rfp

if __name__ == "__main__":
    make_rfp()  # Test the function to ensure it runs without errors