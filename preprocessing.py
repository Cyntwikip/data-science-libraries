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

def delayed_rolling_window(series, delay=0, lookback=1, steps=1):
    """
    Apply rolling window with delay on a Pandas Series input.
    
    Parameters
    ----------
    series : Pandas Series
        time series data
    delay : int
        delay
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
        if i<lookback:
            shift = -i
        else:
            shift = -i-delay
        new_data[i] = series.shift(shift)
    new_data.dropna(inplace=True)
    new_data.columns = (['t-{}'.format(i) for i in range(1+delay,lookback+1+delay)[::-1]] 
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


# Running window
# trim outside 1.5X interquartile range in the window
def trim_outliers(df, col, window=30*24):
    """
    Given dataframe, column name, and window
    Return a new dataframe with trimmed outliers
    Outliers are values beyond 1.5 X interquartile range

    Arguments
    =========
    df = dataframe
    col = column name
    window = hours to check for outliers, default is 30 * 24 hours

    Returns
    ========
    Series of values with null outliers for the column
    Outliers returned as null
    """
    
    df_1 = df.copy()
    
    num_windows = len(df) // window
    for i in range(num_windows + 1):
        range_start = i * window
        range_end = np.min([window * (i + 1), len(df)])
        vals = df[col][range_start: range_end].values

        q75, q25 = np.percentile(vals, [75, 25])
        iqr = q75 - q25
        trim_min = q25 - 1.5 * iqr
        trim_max = q75 + 1.5 * iqr

        df_1.loc[range_start:range_end, col] = df.loc[range_start:range_end, col].apply(
            lambda x: x if x < trim_max else np.nan)
        df_1.loc[range_start:range_end, col] = df.loc[range_start:range_end, col].apply(
            lambda x: x if x > trim_min else np.nan)

        
    return df_1[col]

def add_time_features(df):
    """
    Given dataframe with epochs as index
    Return a dataframe additional columns: one-hot encoded day, month, and hour
    """
    from datetime import datetime as dt

    # adding time, day, and month as features
    # convert epochs to datetime
    months = [dt.fromtimestamp(df.index[i]).month for i in range(len(df))]
    days = [dt.fromtimestamp(df.index[i]).day for i in range(len(df))]
    hours = [dt.fromtimestamp(df.index[i]).hour for i in range(len(df))]

    df["month"] = months
    df["day"] = days
    df["hour"] = hours
    
    df = pd.get_dummies(data = df, columns=["month", "day", "hour"])

    return df

# daily from n hours ago to n months ago (same hour)
def get_past_vals(df_input, col, window_start=1, window_end=24, interval=1):
    """
    Given dataframe df and column col, 
    return a new dataframe with past values as columns

    Arguments
    =========
    df - dataframe
    col - column name
    window_start - starting hour reckoned from first value, default = 0
    window_end - ending hour reckoned from first value, default = 24
    interval - interval of hours to make as columns, default = 1

    Return
    ========
    Dataframe with past data as columns
    """
    df = df_input.copy()

    # make a window of features of len(df_train) = all features
    for i in range(window_start, window_end+1, interval):
        df[f"{col}-{i}h"] = df[col].shift(i)

    return df

def train_test_split(df, test_prop):
    """
    Split a pandas dataframe into training and testing
    Given df dataframe and proportion of test data
    """
    train_prop = 1 - test_prop
    starting_epoch = min(df.index)
    df_len = len(df)
    start_test = int(train_prop * df_len)
    df_train = df[:start_test]
    df_test = df[start_test:]
    return df_train, df_test
