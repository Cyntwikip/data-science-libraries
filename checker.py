import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
from datetime import timedelta

def check_imputation_range(df, col, range_ = 168):
    """
    Arguments
    =========
    df = pandas DataFrame
    range_ = range of missing values (hours)
    col = column to check
    
    Returns
    =========
    MAPE of imputated values (% error)
    Plots imputed vs actual
    
    """
    # artificially make missing values
    start0 = np.random.randint(low=1, high=10000)
    end0 = start0+range_
    
    df_ = df[["date", col]]
    df_.date = pd.to_datetime(df_.date)
    df_[col][start0:end0] = np.nan

    dict_null_ranges = {}
    series = df_[col]
    null_ranges = get_null_ranges(series)
    dict_null_ranges[col] = null_ranges

    # FOR 1 h < X <= 168 h (7 days)
    # get indices of 1 hour null values
    indices = list(dict_null_ranges[col].keys())
    ranges = list(dict_null_ranges[col].values())

    inds_1_168 = [indices[i] for i in range(
        len(indices)) if ranges[i] > 1 and ranges[i] <= 168]

    for i in inds_1_168:
        range_ = dict_null_ranges[col][i]

        # forward
        start, end = find_similar_date_range(df_, col, i, forward=True)
        forward_vals = df_[col][start:end]

        # backward
        start, end = find_similar_date_range(df_, col, i, forward=False)
        backward_vals = df_[col][start:end]

        # Fourier transform TO DO LATER
        # For now, average out the forward values and backward values
        mean_vals = np.mean([forward_vals, backward_vals], axis=0)

        # save
        df_[col][i:i+range_] = mean_vals

    plt.plot(df_[col][start0:end0], "-s", label="imputed")
    plt.plot(df[col][start0:end0], "-s", label="truth")
    plt.xlabel("Index (time)")
    plt.ylabel("Demand")
    plt.title("Imputed vs actual values")
    plt.ylim(0,)
    plt.legend()
    plt.show()

    actual = df[col][start0:end0].values
    imputed = df_[col][start0:end0].values
    num_samples = len(actual)
    mape = np.sum(abs(actual-imputed)/actual)/num_samples * 100
    print(f"MAPE = {mape}%")
    return mape

def check_imputation_1h(df, col):
    """
    Arguments
    =========
    df = pandas DataFrame
    col = column to check

    Returns
    =========
    MAPE of imputated values (% error)
    Plots imputed vs actual
    """

    # artificially make missing values
    df_ = df[["date", col]]
    df_.date = pd.to_datetime(df_.date)
    rand_ind = np.random.randint(1, len(df_))
    df_[col][rand_ind] = np.nan

    # Impute
    df_[col] = impute_1h_gaps(df_, col=col)

    plt.plot(df_[col][rand_ind-3:rand_ind+3], "s-", label="imputed")
    plt.plot(df[col][rand_ind-3:rand_ind+3], "s-", label="truth")
    plt.xlabel("Index (time)")
    plt.ylabel("Demand")
    plt.title("Imputed vs actual values")
    plt.ylim(0,)
    plt.legend()
    plt.show()

    actual = df[col][rand_ind]
    imputed = df_[col][rand_ind]
    mape = np.sum(abs(actual-imputed)/actual) * 100
    print(f"MAPE = {mape}%")
    return mape

def find_similar_date_range(df_, col, i, forward=True):
    """
    Given series of date, and initial date i and range from that date
    Say, Jan 1 and range is 5 so January 1 to 5. Return a list of dates
    with similar time and weekend/weekday matching

    Arguments
    =========
    df_ = pandas Dataframe of dates and values
    col = column of dataframe to consider
    i = index of date in Series
    forward = True if find a date range going forwards or False if going backwards

    Returns
    ========
    Returns an index range for date with similar weekend/weekday matching and hours
    """

    # get values df_[col].values
    s = df_[col]

    # get dictionary of null ranges
    dict_null_ranges = {}
    null_ranges = get_null_ranges(s)
    dict_null_ranges[col] = null_ranges

    range_ = dict_null_ranges[col][i]  # get range

    back_i = i

    # TARGET VALUES
    base = df_.date[i]
    target_dates = np.array([base + datetime.timedelta(hours=i)
                             for i in range(range_)])
    target_dates = [np.datetime64(date)
                    for date in target_dates]  # convert to datetime
    target_dates = [date.astype(dtype='datetime64[s]').item()
                    for date in target_dates]  # convert to datetime64[s]

    target_dayofweek = np.array([d.isoweekday()
                                 for d in target_dates])  # day of week
    target_isweekday = np.array(
        [0 if d == 6 or d == 7 else 1 for d in target_dayofweek])
    target_hour = np.array([d.hour for d in target_dates])

    # from index i - range_ (starting of window), move to backwards one hour at a time
    # until window_hour matches will all_hour and window_isweekday matches with all_isweekday

    while True:
        # running window of dates with null values
        base = df_.date[back_i]
        window_date = np.array([base + datetime.timedelta(hours=i)
                                for i in range(range_)])
        window_date = [np.datetime64(date)
                       for date in window_date]  # convert to datetime
        window_date = [date.astype(dtype='datetime64[s]').item()
                       for date in window_date]  # convert to datetime64[s]

        window_dayofweek = np.array([d.isoweekday()
                                     for d in window_date])  # day of week
        window_isweekday = np.array(
            [0 if d == 6 or d == 7 else 1 for d in window_dayofweek])
        window_hour = np.array([d.hour for d in window_date])

        # check conditions == matching
        # matching hours
        condition_1 = np.all([target_hour[i] == window_hour[i]
                              for i in range(len(window_hour))])

        # matching weekends and weekdays
        condition_2 = np.all([target_isweekday[i] == window_isweekday[i]
                              for i in range(len(window_isweekday))])

        # check for nan
        condition_3 = np.isnan(df_[col][back_i: back_i+range_]).any()

        if condition_1 and condition_2 and not condition_3:
            break

        if forward == True:
            back_i += 1
        else:
            back_i -= 1

    # return a list of dates
    final_date_window = (back_i, back_i + range_)

    return final_date_window

def impute_1h_gaps(df_, col):
    """
    Arguments
    =========
    df_ = dataframe
    col = column of interest in the dataframe

    Returns
    =========
    s = series with imputed values for 1 h gaps
    """

    # dict_null_ranges follows {column_1: {null_index_1: range_1, null_index_2: range_2}, ...}
    dict_null_ranges = {}

    # loop over all columns
    series = df_[col]
    null_ranges = get_null_ranges(series)
    dict_null_ranges[col] = null_ranges

    indices = list(dict_null_ranges[col].keys())
    ranges = list(dict_null_ranges[col].values())

    # get values df_[col].values
    s = df_[col].values

    # FOR 1 hour gaps
    # for ranges == 1, just average previous index and next index
    # get indices of 1 hour null values
    if len(indices) == 1:
        inds_1 = indices
    else:
        inds_1 = [indices[i] for i in range(len(indices)) if ranges[i] == 1]

    # interpolate average(inds_1 - 1, inds_1 + 1)
    for i in range(len(inds_1)):
        ind = inds_1[i]
        s[ind] = np.mean([s[ind - 1], s[ind + 1]])

    return s

# make sure all dates are present
# check if all dates are in the dataframe
def check_dates(df_):
    """
    Given pandas Dataframe, check if all dates are present
    Check if hourly
    Return missing dates
    """
    num_dates = len(df_)

    # generate if all datetimes in the dataframe
    base = min(df_.date)
    arr = np.array([base + datetime.timedelta(hours=i) for i in range(num_dates)])

    # indices of date gaps
    a = [i for i in range(len(df_.date)) if df_.date[i] != arr[i]]
    if a == []:
        print("OK")
    else:
        return(a)
    
def get_null_ranges(series):
    """
    Accepts pandas series and returns key:value pair of
    index of first null value in range and corresponding range
    """

    # get null values
    null_vals = series.isnull().values

    # find indices of null values (True = 1)
    null_inds = np.nonzero(null_vals)[0]

    # dictionary of first index and length of consecutive hours
    null_dict = {}

    i = 0
    while i < len(null_inds):

        # special case that there's only one null
        if len(null_inds) == 1:
            key = null_inds[0]
            null_dict[key] = [1]

        try:
            # initialize key
            key = null_inds[i]

            # range_ is the value. initialize
            range_ = 1

            # append to range_ while there's consecutive vals
            while null_inds[i + 1] == null_inds[i] + 1:
                range_ += 1
                i += 1
            i += 1

            # save value to key
            null_dict[key] = range_

        except:
            # save value to key
            null_dict[key] = range_
            break

    return null_dict




