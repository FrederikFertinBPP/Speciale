import pandas as pd
from data_scripts import DataForecaster, HistoricalData

#%% Data retrieval - There is only data from 2015 and forward for Portugal
start   = pd.Timestamp('20160101', tz='UTC')
end     = pd.Timestamp('20201231', tz='UTC')
data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')

#%% Forecasting setup
forecaster = DataForecaster(data_object,
                            cache_id="test1",
                            verbose=True,
                            )
forecaster.build_simulation_models(hmm=False, to_pickle=True)