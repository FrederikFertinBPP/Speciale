import pandas as pd
from data_scripts import DataForecaster, HistoricalData

#%% Data retrieval - There is only data from 2015 and forward for Portugal
start   = pd.Timestamp('20160101', tz='UTC')
end     = pd.Timestamp('20201231', tz='UTC')
data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')

#%% Forecasting setup
forecaster = DataForecaster(data_object,
                            cache_id="v2",
                            verbose=False, # Takes under 10 minutes to create, set verbose equals True to see progress, but know that verbose=True for the unpickled object.
                            cache_replace=True,
                            )
forecaster.build_simulation_models(to_pickle=True)
print("Data forecaster and models built and pickled.")