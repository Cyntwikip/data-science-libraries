import numpy as np
import pandas as pd

import datetime
from datetime import timedelta

def rolling_window(series, lookback=1, steps=1):
    """
    Apply rolling window on a Pandas Series input.
    
    Parameters
    ----------
    series : Pandas Series
        time series data
    lookback : int
        number of past values
    steps : int
        number of future values
        
    Returns
    ----------
    new_data : Pandas DataFrame
        windowed data
    """
    new_data = pd.DataFrame()
    for i in range(lookback+steps):
        new_data[i] = series.shift(-i)
    new_data.dropna(inplace=True)
    new_data.columns = (['t-{}'.format(i) for i in range(1,lookback+1)[::-1]] 
                        + ['t+{}'.format(i) for i in range(0,steps)])
    return new_data

def gen_hourly_epochs(start_time, end_time):
    """
    Generate hourly epochs. Assumes timezone is PH.
    
    Parameters
    ----------
    start_time : datetime
        start time
    end_time : datetime
        end time
        
    Returns
    ----------
    hourly_epochs : Pandas DataFrame
        epochs with hourly intervals as index of the dataframe
        
    Examples
    ----------
    start_time = datetime.datetime(2007,1,1)
    end_time = datetime.datetime(2019,4,1)
    gen_hourly_epochs(start_time, end_time)
    """
    dates = pd.date_range(start_time,end_time,freq=pd.tseries.offsets.Hour())
    dates = pd.DataFrame({'date':dates})
    # convert to timestamp
    dates['ts'] = dates['date'].values.astype(np.int64) // 10 ** 9
    # UTC correction
    dates['ts'] = dates['ts']-3600*8
    dates.set_index('ts',inplace=True)
    hourly_epochs = dates.drop('date',axis=1)
    
    return hourly_epochs

def ts_to_dt(ts):
    """
    Converts timestamp to datetime
    
    Parameters
    ----------
    ts : int
        timestamp
        
    Returns
    ----------
    date : datetime
        datetime equivalent of epoch
        
    Examples
    ----------
    ts_to_dt(1554048000)
    """
    return datetime.datetime.fromtimestamp(ts)

def years_to_ts(years):
    """
    Converts an array of years to timestamp/epoch
    
    Parameters
    ----------
    years : numpy_array
        array of years
        
    Returns
    ----------
    ts : numpy_array
        array of epoch equivalent of each year
    """
    f = lambda year: datetime.datetime(year,1,1).timestamp()
    f = np.vectorize(f)
    return f(years)








