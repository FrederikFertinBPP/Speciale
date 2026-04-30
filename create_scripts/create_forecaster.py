""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data_scripts import DataForecaster, HistoricalData

#%% Data retrieval - There is only data from 2015 and forward for Portugal
start   = pd.Timestamp('20150101', tz='UTC')
end     = pd.Timestamp('20251231', tz='UTC')
data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')
data_object.data = data_object.data.loc[(data_object.data.index.minute==0)] # Remove 2025 quarterly data, we only want hourly data for forecasting.

#%% Forecasting setup
forecaster = DataForecaster(data_object,
                            cache_id="Anders",
                            verbose=False, # Takes under 10 minutes to create, set verbose equals True to see progress, but know that verbose=True for the unpickled object.
                            cache_replace=True,
                            )
forecaster.build_simulation_models(to_pickle=True)
print("Data forecaster and models built and pickled.")