# This file is in charge of all data scraping, loading, cleaning,
# preprocessing and imputing.
# To use this, you just need to do one call:
# generate_consolidated_data_for_all_zones()
# Of course, it needs a few files to work:
# 1. psy chapter 1_15.xls - from PSA and in sheet T1_1, it contains
# population and population density data
# 2. {Place} Daily Data.xlsx - 7 files that constitute different places from
# pagasa. 7 places are NAIA, Tuguegarao, Dagupan, Clark, Tanay, Tayabas and
# Sangley Point
# 3. Luzon Demand per Zone.xlsx - Data from a certain company
# 4. Visayas Demand per Zone.xlsx - Data from a certain company
# The rest are scraped like GRDP and holiday data
# 5. earthquakes_2000_onwards.csv - Data of earthquakes
# 6. Philippine Provincial Shape Data (5 files)


def load_demand_excel(path):
    """
    Input path of excel file, read excel file as pandas dataframe

    Parameters
    ==========
    path - path of excel file <path>/<filename>/xlsx
    First column in Excel file must be the date

    Returns
    ==========
    pandas dataframe of input data
    """
    import numpy as np
    import pandas as pd

    # read excel as is
    demand_df = pd.read_excel(path)

    # read demand
    demand_df.columns = ["date"] + list(demand_df.columns[1:])

    # change the date to datetime format
    demand_df.date = pd.to_datetime(demand_df.date)

    # return
    return demand_df


def get_null_ranges(series):
    """
    Accepts pandas series and returns key:value pair of
    index of first null value in range and corresponding range
    """
    import numpy as np
    import pandas as pd

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
    import numpy as np
    import pandas as pd

    # dict_null_ranges follows {column_1:
    # {null_index_1: range_1, null_index_2: range_2}, ...}
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
    forward = True if find a date range going forwards or False otherwise

    Returns
    ========
    Returns an index range for date with similar weekend/weekday matching
    and hours
    """
    import numpy as np
    import pandas as pd
    
    import datetime
    from datetime import timedelta
    from datetime import datetime as dt

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

    # from index i - range_ (starting of window), move to backwards one hour
    # at a time until window_hour matches will all_hour and window_isweekday
    # matches with all_isweekday

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


def clean_1_168h(df, cols):
    """
    Given dataframe of demand
    And columns to clean

    Return df with imputed values for 1-168 h missing data
    """
    import numpy as np
    import pandas as pd

    for col in cols:
        dict_null_ranges = {}

        series = df[col]
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
            start, end = find_similar_date_range(df, col, i, forward=True)
            forward_vals = df[col][start:end]

            # backward
            start, end = find_similar_date_range(df, col, i, forward=False)
            backward_vals = df[col][start:end]

            # Average out the forward values and backward values
            mean_vals = np.mean([forward_vals, backward_vals], axis=0)

            # save
            df[col][i:i+range_] = mean_vals

    return df


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
    import numpy as np
    import pandas as pd

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

        df_1.loc[
            range_start:range_end,
            col
        ] = df.loc[range_start:range_end, col].apply(
            lambda x: x if x < trim_max else np.nan)
        df_1.loc[
            range_start:range_end,
            col
        ] = df_1.loc[range_start:range_end, col].apply(
            lambda x: x if x > trim_min else np.nan)

    return df_1[col]


def prepare_demand_data():
    '''Prepare the demand data in multiple "Zone X RTX.csv" file'''
    import numpy as np
    import pandas as pd
    from datetime import datetime as dt
    import datetime

    # Load the excel files
    df_luzon_0 = load_demand_excel('Luzon Demand per Zone.xlsx')
    df_visayas_0 = load_demand_excel('Visayas Demand per Zone.xlsx')

    # Ensure that we only consider the right columns
    luzon_cols = ['date', 'ZONE 1 RTX', 'ZONE 2 RTX', 'ZONE 3 RTX',
                  'ZONE 1 RTD', 'ZONE 2 RTD', 'ZONE 3 RTD', '3KAL_P01 RTX',
                  '3KAL_P02 RTX', '3KAL_P03 RTX', '3KAL_P04 RTX',
                  '3KAL_P01 RTD', '3KAL_P02 RTD', '3KAL_P03 RTD',
                  '3KAL_P04 RTD']
    df_luzon_0 = df_luzon_0[luzon_cols]
    visayas_cols = ['date', 'Zone 4.rtx Load', 'Zone 5.rtx Load',
                    'Zone 6.rtx Load', 'Zone 7.rtx Load', 'Zone 8.rtx Load',
                    'Zone 4.rtd Load Forecast', 'Zone 5.rtd Load Forecast',
                    'Zone 6.rtd Load Forecast', 'Zone 7.rtd Load Forecast',
                    'Zone 8.rtd Load Forecast']
    df_visayas_0 = df_visayas_0[visayas_cols]

    # Lists of relevant columns for later
    cols_rtx_luzon = ['ZONE 1 RTX', 'ZONE 2 RTX', 'ZONE 3 RTX']
    cols_rtd_luzon = ['ZONE 1 RTD', 'ZONE 2 RTD', 'ZONE 3 RTD']

    cols_rtx_visayas = ['Zone 4.rtx Load', 'Zone 5.rtx Load',
                        'Zone 6.rtx Load', 'Zone 7.rtx Load',
                        'Zone 8.rtx Load']

    cols_rtd_visayas = ['Zone 4.rtd Load Forecast',
                        'Zone 5.rtd Load Forecast',
                        'Zone 6.rtd Load Forecast',
                        'Zone 7.rtd Load Forecast',
                        'Zone 8.rtd Load Forecast']

    # Round datetime to the nearest hour
    df_luzon_0.date = df_luzon_0.date.dt.round("H")
    df_visayas_0.date = df_visayas_0.date.dt.round("H")

    # Copy to df_luzon_1 and df_visayas_1
    df_luzon_1 = df_luzon_0.copy()
    df_visayas_1 = df_visayas_0.copy()

    # Replace 0 values with NaN as these are most likely errors in data
    df_luzon_1.iloc[:, 1:] = df_luzon_1.iloc[:, 1:].replace(0, np.nan)
    df_visayas_1.iloc[:, 1:] = df_visayas_1.iloc[:, 1:].replace(0, np.nan)

    # Recopy values for Visayas November 3 to 30
    # This is due to the Yolanda incident, it will be imputed
    # but we want to retain the effect
    cond_1 = df_visayas_1.date <= pd.datetime(2013, 11, 30)
    cond_2 = df_visayas_1.date >= pd.datetime(2013, 11, 3)
    recopy = df_visayas_0[[np.all([i, j]) for i, j in zip(cond_1, cond_2)]]
    df_visayas_1[[np.all([i, j]) for i, j in zip(cond_1, cond_2)]] = recopy

    # Fill in <1 hour gaps for all columns
    for col in df_luzon_1.columns.drop('date'):
        df_luzon_1[col] = impute_1h_gaps(df_luzon_1, col)
    for col in df_visayas_1.columns.drop('date'):
        df_visayas_1[col] = impute_1h_gaps(df_visayas_1, col)

    # Fill in 1-168 hour gaps for all columns
    df_luzon_1 = clean_1_168h(df_luzon_1, cols_rtx_luzon)
    df_visayas_1 = clean_1_168h(df_visayas_1, cols_rtx_visayas)

    # Fill in the rest using RTD data for Luzon
    for col_rtx, col_rtd in zip(cols_rtx_luzon, cols_rtd_luzon):
        null_ranges = get_null_ranges(df_luzon_1[col_rtx])

        for index, range_ in null_ranges.items():
            df_luzon_1[
                col_rtx
            ][
                index:index + range_
            ] = df_luzon_1[col_rtd][index:index+range_]

    # Fill in the rest using RTD data for Visayas
    for col_rtx, col_rtd in zip(cols_rtx_visayas, cols_rtd_visayas):
        null_ranges = get_null_ranges(df_visayas_1[col_rtx])

        for index, range_ in null_ranges.items():
            df_visayas_1[
                col_rtx
            ][
                index:index + range_
            ] = df_visayas_1[col_rtd][index:index+range_]

    # Deduct Kalayaan water pump demand from overall zone 3 demand
    kalayaan = ['3KAL_P01 RTX', '3KAL_P02 RTX',
                '3KAL_P03 RTX', '3KAL_P04 RTX']
    to_deduct = df_luzon_1[kalayaan].sum(axis=1)
    df_luzon_1["ZONE 3 RTX"] = df_luzon_1["ZONE 3 RTX"] - to_deduct

    # Copy to another dataframe for safety
    df_luzon_trimmed = df_luzon_1.copy().reset_index(drop=True)
    df_visayas_trimmed = df_visayas_1.copy().reset_index(drop=True)

    # Trimming outliers for every 30 day window
    window = 30 * 24
    for col in cols_rtx_luzon:
        df_luzon_trimmed[col] = trim_outliers(
            df_luzon_trimmed, col, window=window)
    for col in cols_rtx_visayas:
        df_visayas_trimmed[col] = trim_outliers(
            df_visayas_trimmed, col, window=window)

    # Refill in 1 hour gaps
    for col in cols_rtx_luzon:
        df_luzon_trimmed[col] = impute_1h_gaps(df_luzon_trimmed, col)
    for col in cols_rtx_visayas:
        df_visayas_trimmed[col] = impute_1h_gaps(df_visayas_trimmed, col)

    # Refill in 1-168 h gaps
    df_luzon_trimmed = clean_1_168h(df_luzon_trimmed, cols_rtx_luzon)
    df_visayas_trimmed = clean_1_168h(df_visayas_trimmed, cols_rtx_visayas)

    # Reset Index to date column
    df_luzon_trimmed = df_luzon_trimmed.reset_index(drop=True)
    df_visayas_trimmed = df_visayas_trimmed.reset_index(drop=True)

    # Make date into epochs (timestamps)
    df_luzon_trimmed["epoch"] = df_luzon_trimmed.date.apply(
        lambda x: dt.timestamp(x))
    df_visayas_trimmed["epoch"] = df_visayas_trimmed.date.apply(
        lambda x: dt.timestamp(x))

    # Convert epochs to integer
    df_luzon_trimmed.epoch = df_luzon_trimmed.epoch.apply(lambda x: int(x))
    df_visayas_trimmed.epoch = df_visayas_trimmed.epoch.apply(
        lambda x: int(x))

    # Set index to epoch
    df_luzon_trimmed = df_luzon_trimmed.set_index("epoch")
    df_visayas_trimmed = df_visayas_trimmed.set_index("epoch")

    # Load the data for luzon with some preprocessing
    df_luzon_trimmed.drop('date', axis=1, inplace=True)
    df_luzon_trimmed = df_luzon_trimmed.reset_index()
    df_luzon_trimmed = df_luzon_trimmed.rename(columns={'epoch': 'date'})

    # Load the data for visayas with some preprocessing
    df_visayas_trimmed.drop('date', axis=1, inplace=True)
    df_visayas_trimmed = df_visayas_trimmed.reset_index()
    df_visayas_trimmed = df_visayas_trimmed.rename(columns={'epoch': 'date'})

    # Set the timestamp as the index
    df_1 = df_luzon_trimmed.iloc[:, [0, 1]].set_index('date')
    df_2 = df_luzon_trimmed.iloc[:, [0, 2]].set_index('date')
    df_3 = df_luzon_trimmed.iloc[:, [0, 3]].set_index('date')
    df_4 = df_visayas_trimmed.iloc[:, [0, 1]].set_index('date')
    df_5 = df_visayas_trimmed.iloc[:, [0, 2]].set_index('date')
    df_6 = df_visayas_trimmed.iloc[:, [0, 3]].set_index('date')
    df_7 = df_visayas_trimmed.iloc[:, [0, 4]].set_index('date')
    df_8 = df_visayas_trimmed.iloc[:, [0, 5]].set_index('date')

    # Save to file
    df_1.to_csv('Zone 1 RTX.csv')
    df_2.to_csv('Zone 2 RTX.csv')
    df_3.to_csv('Zone 3 RTX.csv')
    df_4.to_csv('Zone 4 RTX.csv')
    df_5.to_csv('Zone 5 RTX.csv')
    df_6.to_csv('Zone 6 RTX.csv')
    df_7.to_csv('Zone 7 RTX.csv')
    df_8.to_csv('Zone 8 RTX.csv')

    return df_1.index.tolist()


def prepare_earthquake_data():
    '''Prepare earthquake data for all zones'''
    from shapely.ops import nearest_points
    from shapely.geometry import Point, MultiPoint
    import pandas as pd
    import geopandas as gpd

    # Mapping of zones to provinces and cities available in data set
    zone_to_province_city_mapping = {
        'Zone 1': [
            'Abra',
            'Apayao',
            'Benguet',
            'Ifugao',
            'Kalinga',
            'Mt. Province',
            'Ilocos Norte',
            'Ilocos Sur',
            'La Union',
            'Pangasinan',
            'Batanes',
            'Cagayan',
            'Isabela',
            'Nueva Vizcaya',
            'Quirino',
            'Aurora',
            'Bataan',
            'Bulacan',
            'Nueva Ecija',
            'Pampanga',
            'Tarlac',
            'Zambales',
        ],
        'Zone 2': [
            'Metropolitan Manila'
        ],
        'Zone 3': [
            'Batangas',
            'Cavite',
            'Laguna',
            'Quezon',
            'Rizal',
            'Albay',
            'Camarines Norte',
            'Camarines Sur',
            'Catanduanes',
            'Sorsogon',
        ],
        'Zone 4': [
            'Eastern Samar',
            'Leyte',
            'Northern Samar',
            'Samar',
            'Southern Leyte',
        ],
        'Zone 5': [
            'Cebu',
        ],
        'Zone 6': [
            'Negros Oriental',
            'Negros Occidental',
        ],
        'Zone 7': [
            'Bohol',
        ],
        'Zone 8': [
            'Aklan',
            'Antique',
            'Capiz',
            'Iloilo',
        ]
    }

    # Read the earthquake data and the philippine shapefiles
    earth = pd.read_csv('earthquakes_2000_onwards.csv').iloc[:, :5]
    ph = gpd.read_file('gadm36_PHL_1.shp')

    # Create the geometry
    geometry = [Point(xy) for xy in zip(earth['longitude'],
                                        earth['latitude'])]

    # Set the crs
    crs = {'init': 'epsg:4326'}

    # Create the GeoDataFrame from the DataFrame
    earth_gdf = gpd.GeoDataFrame(earth, crs=crs, geometry=geometry)
    earth_gdf

    # Assign the centroid of the province to the geometry column
    ph['geometry'] = ph['geometry'].centroid

    # Get the centroid of the province nearest to each of the provinces
    lst = []
    for id, point in earth_gdf.iterrows():
        nearest_geoms = nearest_points(
            point['geometry'], MultiPoint(ph['geometry'].tolist()))
        lst.append(nearest_geoms[1])

    # Create new column for centroid on the earthquake data
    earth_gdf['new'] = lst
    earth_gdf.rename(columns={'new': 'geometry',
                              'geometry': 'new'}, inplace=True)

    # Join together the two columns
    final_df = gpd.sjoin(earth_gdf, ph, op='within')

    # Fix date so that it is consistent with all the other functions
    final_df['date'] = pd.to_datetime(final_df['time']).dt.round('H')
    final_df = final_df.sort_values(['date'], ascending=True)
    final_df['date'] = pd.to_datetime(
        final_df['date'].astype(str)) + pd.DateOffset(hours=8)
    final_df = final_df.sort_values('mag', ascending=False).drop_duplicates(
        'date').sort_values('date', ascending=True)
    final_df.set_index('date').resample('H').pad()
    final_df = final_df.reset_index()
    final_df['date'] = final_df[['date']].apply(
        lambda x: x[0].timestamp(), axis=1).astype(int)
    final_df['date'] = final_df['date'] - (3600*8)
    final_df = final_df.set_index('date')

    # Prune some columns
    final_df = final_df.loc[:, ['depth', 'mag', 'NAME_1']]

    # Map province data into zone data
    for zone, provinces in zone_to_province_city_mapping.items():
        final_df[zone + ' mag'] = [
            value['mag'] if value['NAME_1'] in provinces
            else 0 for index, value in final_df.iterrows()]
        final_df[zone + ' depth'] = [
            value['depth'] if value['NAME_1'] in provinces
            else 0 for index, value in final_df.iterrows()]

    # Save the final dataframe to CSV file
    final_df.to_csv('earthquake_per_zone.csv')
    

def incorporate_typhoon_data():
    '''Include typhoon data in the consolidated dataset'''
    import pandas as pd
    import numpy as np
    from shapely.ops import nearest_points
    from shapely.geometry import Point, MultiPoint
    import pandas as pd
    import geopandas as gpd

    # Mapping of zones to provinces and cities available in data set
    zone_to_province_city_mapping = {
        'Zone 1': [
            'Abra',
            'Apayao',
            'Benguet',
            'Ifugao',
            'Kalinga',
            'Mt. Province',
            'Ilocos Norte',
            'Ilocos Sur',
            'La Union',
            'Pangasinan',
            'Batanes',
            'Cagayan',
            'Isabela',
            'Nueva Vizcaya',
            'Quirino',
            'Aurora',
            'Bataan',
            'Bulacan',
            'Nueva Ecija',
            'Pampanga',
            'Tarlac',
            'Zambales',
        ],
        'Zone 2': [
            'Metropolitan Manila'
        ],
        'Zone 3': [
            'Batangas',
            'Cavite',
            'Laguna',
            'Quezon',
            'Rizal',
            'Albay',
            'Camarines Norte',
            'Camarines Sur',
            'Catanduanes',
            'Sorsogon',
        ],
        'Zone 4': [
            'Eastern Samar',
            'Leyte',
            'Northern Samar',
            'Samar',
            'Southern Leyte',
        ],
        'Zone 5': [
            'Cebu',
        ],
        'Zone 6': [
            'Negros Oriental',
            'Negros Occidental',
        ],
        'Zone 7': [
            'Bohol',
        ],
        'Zone 8': [
            'Aklan',
            'Antique',
            'Capiz',
            'Iloilo',
        ]
    }

    typhoon_df = pd.read_csv('typhoons_cleaned(1).csv')
    typhoon_df['RADIUS'] = np.mean(typhoon_df[['USA_R34_NE', 'USA_R34_NW',
                                               'USA_R34_SE', 'USA_R34_SW']],
                                   axis=1) * 1852
    typhoon_df.drop([
        'NAME',
        'USA_R34_NE',
        'USA_R34_NW',
        'USA_R34_SE',
        'USA_R34_SW'
    ], axis=1, inplace=True)

    # Read the philippine shapefiles
    ph = gpd.read_file('gadm36_PHL_1.shp')
    # Create the geometry
    geometry = [Point(xy) for xy in zip(typhoon_df['LON'],
                                        typhoon_df['LAT'])]

    # Set the crs
    crs = {'init': 'epsg:4326'}

    # Create the GeoDataFrame from the DataFrame
    typhoon_gdf = gpd.GeoDataFrame(typhoon_df, crs=crs, geometry=geometry)
    typhoon_gdf = typhoon_gdf.to_crs({'init': 'epsg:3857'})

    # Assign the centroid of the province to the geometry column
    ph['geometry'] = ph['geometry'].centroid
    ph = ph.to_crs({'init': 'epsg:3857'})

    for i in range(len(typhoon_gdf)):
        typhoon_gdf.loc[
            i,
            'geometry'
        ] = typhoon_gdf.loc[i,
                            'geometry'].buffer(typhoon_gdf.loc[i, 'RADIUS'])

    typhoon_gdf['geometry'] = typhoon_gdf.apply(
        lambda x: x.geometry.buffer(x.RADIUS), axis=1)

    final_df = gpd.sjoin(ph, typhoon_gdf, op='within')

    # Fix date so that it is consistent with all the other functions
    final_df['date'] = pd.to_datetime(final_df['ISO_TIME']).dt.round('H')
    final_df = final_df.sort_values(['date'], ascending=True)
    final_df['date'] = pd.to_datetime(
        final_df['date'].astype(str)) + pd.DateOffset(hours=8)
    final_df = final_df.sort_values('USA_SSHS', ascending=False).drop_duplicates(
        'date').sort_values('date', ascending=True)
    final_df.set_index('date').resample('H').pad()
    final_df = final_df.reset_index()
    final_df['date'] = final_df[['date']].apply(
        lambda x: x[0].timestamp(), axis=1).astype(int)
    final_df['date'] = final_df['date'] - (3600*8)
    final_df = final_df.set_index('date')

    # Prune some columns
    final_df = final_df.loc[:, ['USA_WIND', 'USA_PRES', 'USA_SSHS', 'NAME_1']]

    # Map province data into zone data
    for zone, provinces in zone_to_province_city_mapping.items():
        final_df[zone + ' wind'] = [
            float(value['USA_WIND']) if value['NAME_1'] in provinces
            else 0 for index, value in final_df.iterrows()]
        final_df[zone + ' pres'] = [
            float(value['USA_PRES']) if value['NAME_1'] in provinces
            else 0 for index, value in final_df.iterrows()]
        final_df[zone + ' sshs'] = [
            int(value['USA_SSHS']) if value['NAME_1'] in provinces
            # and value['USA_SSHS'] in ['5', '4', '3', '2', '1', '0', '-1']
            else -2 for index, value in final_df.iterrows()]

    # Save the final dataframe to CSV file
    final_df.to_csv('typhoons_per_zone.csv')


def extract_gdp_per_capita():
    '''Extract GDP per capita per zone'''
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    list_of_regions = [
        'ilocos-region',
        'cordillera',
        'cagayan-valley',
        'central-luzon',
        'metro-manila',
        'calabarzon',
        'bicol-region',
        'western-visayas',
        'central-visayas',
        'eastern-visayas',
    ]

    df = []

    for region in list_of_regions:
        resp = requests.get(
            ('https://www.ceicdata.com/datapage/charts/'
             'o_philippines_gdp-per-capita-' + region + '/'
             '?type=area&from=2009-12-01&to=2017-12-01&lang=en')).text

        soup = BeautifulSoup(resp, 'lxml')
        labels = soup.find_all(
            "tspan", attrs={'class': 'highcharts-text-outline'})
        years = soup.find(
            'g',
            attrs={'class': 'highcharts-xaxis-labels'}
        ).findChildren("text", recursive=False)

        temp_df = pd.DataFrame(columns=['years'])
        years_data = []
        region_data = []
        for i in range(9):
            years_data.append(int(years[i].text))
            region_data.append(float(labels[i].text.replace(' ', '')))

        temp_df['years'] = pd.Series(
            years_data + [str(int(years_data[-1]) + 1)])
        temp_df[region] = pd.Series(region_data + [0])

        if len(df) > 0:
            df = pd.merge(df, temp_df, left_on=['years'], right_on=['years'])
        else:
            df = temp_df

    # Datetime transformations to conform with the original data
    df.rename(columns={'years': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y')
    df = df.set_index('date').resample('H').pad()
    df = df.reset_index()
    df['date'] = df[['date']].apply(
        lambda x: x[0].timestamp(), axis=1).astype(int)
    df['date'] = df['date'] - (3600*8)
    df = df.set_index('date')

    # Relegate the regions into zones
    df['Zone 1'] = df.iloc[:, 0:4].sum(axis=1)
    df['Zone 2'] = df.iloc[:, 4]
    df['Zone 3'] = df.iloc[:, 5:7].sum(axis=1)
    df['Zone 4'] = df.iloc[:, 9]
    df['Zone 5'] = df.iloc[:, 8]
    df['Zone 6'] = df.iloc[:, 7:9].sum(axis=1)
    df['Zone 7'] = df.iloc[:, 8]
    df['Zone 8'] = df.iloc[:, 7]

    # Save data to csv
    df.to_csv('Regional GDP.csv')


def change_month_to_english(month):
    '''Change Filipino Month to English Month'''
    month_mapping = {
        'Ene': 'Jan',
        'Peb': 'Feb',
        'Abr': 'Apr',
        'Hun': 'Jun',
        'Hul': 'Jul',
        'Ago': 'Aug',
        'Set': 'Sep',
        'Okt': 'Oct',
        'Nob': 'Nov',
        'Dis': 'Dec',
    }

    for fil, eng in month_mapping.items():
        month = month.replace(fil, eng)

    return month


def extract_holidays(dates):
    '''Get Philippine Holidays'''
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import datetime
    import pandas as pd

    # Get all years considered for the final dataset
    years = sorted(list(
        set(
            [datetime.datetime.fromtimestamp(
                date).strftime('%Y') for date in dates])))

    holiday_dct = {}

    # Iterate over years to get holidays per month
    for year in years:
        resp = requests.get(
            'https://www.timeanddate.com/holidays/philippines/' + year).text

        soup = BeautifulSoup(resp, 'lxml')
        holiday_list = soup.find_all(
            "tr", attrs={'class': ['c0', 'c1']})

        for holiday in holiday_list:
            date = holiday.findChildren("th")[0].text
            holiday_dct[
                datetime.datetime.strptime(
                    change_month_to_english(date) + ' ' + year,
                    "%d %b %Y"
                ).date()] = holiday.findChildren("td")[2].text

    # Create two dataframes to join together
    date_df = pd.DataFrame(
        index=pd.date_range(years[0], str(int(years[-1]) + 1)))
    holiday_df = pd.DataFrame(
        holiday_dct.values(),
        index=[i.strftime('%Y-%m-%d') for i in holiday_dct.keys()])
    final_df = pd.merge(date_df, holiday_df, how='left',
                        left_index=True, right_index=True)

    # Fix date to match final dataset
    final_df.fillna('No Holiday', inplace=True)
    final_df = final_df.resample('H').pad()
    final_df.reset_index(inplace=True)
    final_df.columns = ['date', 'holiday']
    final_df['date'] = final_df[['date']].apply(
        lambda x: x[0].timestamp(), axis=1).astype(int)
    final_df['date'] = final_df['date'] - (3600*8)
    final_df.set_index('date', inplace=True)

    # Dummify the holiday type
    final_df = pd.get_dummies(final_df, columns=['holiday'])

    final_df.to_csv('Philippine Holidays.csv')


def interpolate_population_data(df, dates):
    '''Interpolate population or population density data'''
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import datetime
    from datetime import timedelta

    df_weight = pd.DataFrame(columns=['weight', 'offset'])
    lr = LinearRegression()

    # Fit a linear regression model on the available dates
    for idx, val in df.iterrows():
        x = np.array(val.index.astype(int))
        f = np.vectorize(lambda year: datetime.datetime(
            year, 1, 1).timestamp())
        x = f(x).reshape(-1, 1)
        y = val.values.reshape(-1, 1)
        lr.fit(x, y)
        df_weight.loc[idx] = [lr.coef_[0][0], lr.intercept_[0]]

    # Use the weights from the linear regression model to interpolate hourly
    data = pd.DataFrame(index=dates)
    for name, val in df_weight.iterrows():
        weight, offset = val['weight'], val['offset']
        data[name] = data.index*weight + offset

    return data


def prepare_population_data(dates):
    '''Clean, impute and organize population data given the dates'''
    import pandas as pd
    import numpy as np

    # Mapping of zones to provinces and cities available in data set
    zone_to_province_city_mapping = {
        'Zone 1': [
            'Abra',
            'Apayao',
            'Benguet',
            'Ifugao',
            'Kalinga',
            'Mt. Province',
            'Ilocos Norte',
            'Ilocos Sur',
            'La Union',
            'Pangasinan',
            'Batanes',
            'Cagayan',
            'Isabela',
            'Nueva Vizcaya',
            'Quirino',
            'Aurora',
            'Bataan',
            'Bulacan',
            'Nueva Ecija',
            'Pampanga',
            'Tarlac',
            'Zambales',
        ],
        'Zone 2': [
            'City of Manila',
            'Calookan City',
            'Las Pinas City',
            'Makati City',
            'Malabon',
            'Mandaluyong City',
            'Marikina City',
            'Muntinlupa City',
            'Navotas',
            'Parañaque City',
            'Pasay City',
            'Pasig City',
            'Pateros',
            'Quezon City',
            'San Juan',
            'Taguig',
            'Valenzuela City',
        ],
        'Zone 3': [
            'Batangas',
            'Cavite',
            'Laguna',
            'Quezon',
            'Rizal',
            'Albay',
            'Camarines Norte',
            'Camarines Sur',
            'Catanduanes',
            'Sorsogon',
        ],
        'Zone 4': [
            'Eastern Samar',
            'Leyte',
            'Northern Samar',
            'Samar',
            'Southern Leyte',
        ],
        'Zone 5': [
            'Cebu',
        ],
        'Zone 6': [
            'Negros Oriental',
            'Negros Occidental',
        ],
        'Zone 7': [
            'Bohol',
        ],
        'Zone 8': [
            'Aklan',
            'Antique',
            'Capiz',
            'Iloilo',
        ]
    }

    # Merge provinces and cities into one big list to match against indices
    location_list = []
    for key, values in zone_to_province_city_mapping.items():
        location_list += values

    # Read the excel file
    pop = pd.read_excel('psy chapter 1_15.xls', sheet_name='T1_1', header=5)
    pop.iloc[:, 0] = pop.iloc[:, 0].str.strip()
    pop.rename(columns={'Region and': 'Location'}, inplace=True)
    pop.set_index(pop.columns[0], inplace=True)
    pop = pop.loc[location_list]

    # Drop the unneeded columns
    pop.drop(
        [column for column in pop.columns if type(
            column) == str and 'Unnamed' in column],
        axis=1, inplace=True)

    # Split the dataframe into two: population and density
    population = pop.iloc[:, :int(len(pop.columns)/2)]
    population_density = pop.iloc[:, int(len(pop.columns)/2):len(pop.columns)]
    population_density.columns = [
        column[:-2] for column in population_density.columns]

    # Interpolate data
    population_data = interpolate_population_data(
        population, dates)
    population_density_data = interpolate_population_data(
        population_density, dates)

    # Merge the provinces and cities into zones
    for zone, city_provinces in zone_to_province_city_mapping.items():
        population_data[zone] = pd.Series([])
        population_density_data[zone] = pd.Series([])
        total_area = 0
        for city_province in city_provinces:
            population_data[zone].fillna(0, inplace=True)
            population_data[zone] += population_data[city_province]
            population_density_data[zone].fillna(0, inplace=True)
            total_area += (population_data[city_province] /
                           population_density_data[city_province])
            population_density_data[zone] += population_data[city_province]
        population_density_data[zone] = (population_density_data[zone] /
                                         total_area)

    # Isolate the 8 Zones
    population_final = population_data.iloc[:, -8:]
    population_final.index.name = 'date'
    population_density_final = population_density_data.iloc[:, -8:]
    population_density_final.index.name = 'date'

    # Save to CSV
    population_final.to_csv('population_per_zone.csv')
    population_density_final.to_csv('population_density_per_zone.csv')


def prepare_temp_precipitation_data():
    '''Get average temperature for each zone (limited for zones 1-3)'''
    import pandas as pd
    import numpy as np

    # Set the relevant sites and filenames
    sites = ['Tuguegarao', 'Tayabas', 'Tanay', 'Sangley Point',
             'NAIA', 'Dagupan', 'Clark']
    files = [site + ' Daily Data.xlsx' for site in sites]

    # Load the Files from Pagasa
    data = {}
    for site, file in zip(sites, files):
        place_holder = pd.read_excel(file, skiprows=17, nrows=4018,
                                     usecols=[0, 1, 2, 3, 4], header=None)
        place_holder.columns = ['year', 'month', 'day', 'rainfall', 'tmean']
        data[site] = place_holder

    # Setup the temperature dataframe
    temp = pd.DataFrame()
    temp['year'] = data[sites[0]]['year']
    temp['month'] = data[sites[0]]['month']
    temp['day'] = data[sites[0]]['day']

    # Set rainfall and temp values
    for data_site, data_df in data.items():
        rainfall_series = data_df['rainfall']
        rainfall_series = rainfall_series.replace(-2, np.nan)
        rainfall_series = rainfall_series.replace('T', 0)
        rainfall_series = rainfall_series.astype(float)
        temp[data_site+'_rainfall'] = rainfall_series
        temp[data_site+'_tmean'] = data_df['tmean']

    temp.loc[len(temp), ['year', 'month', 'day']] = [2018, 1, 1]

    # Fix up the date to conform with the rest of the dataset
    temp['date'] = pd.to_datetime(temp[['year', 'month', 'day']])
    temp.drop(['year', 'month', 'day'], axis=1, inplace=True)
    temp = temp.set_index('date').resample('H').pad()
    temp = temp.reset_index()
    temp['date'] = temp[['date']].apply(
        lambda x: x[0].timestamp(), axis=1).astype(int)
    temp['date'] = temp['date'] - (3600*8)
    temp = temp.set_index('date')

    # Data Imputation
    for column in temp.columns:
        if 'tmean' in column:
            temp[column].fillna(temp[column].mean(), inplace=True)
        else:
            temp[column].fillna(0, inplace=True)

    # Aggregating Zones
    temp['Zone 1 Temperature'] = temp.loc[
        :,
        [
            'Tuguegarao_tmean',
            'Dagupan_tmean',
            'Clark_tmean',
        ]
    ].mean(axis=1)
    temp['Zone 1 Precipitation'] = temp.loc[
        :,
        [
            'Tuguegarao_rainfall',
            'Dagupan_rainfall',
            'Clark_rainfall',
        ]
    ].mean(axis=1)
    temp['Zone 2 Temperature'] = temp.loc[
        :,
        [
            'NAIA_tmean'
        ]
    ]
    temp['Zone 2 Precipitation'] = temp.loc[
        :,
        [
            'NAIA_rainfall',
        ]
    ]
    temp['Zone 3 Temperature'] = temp.loc[
        :,
        [
            'Tayabas_tmean',
            'Tanay_tmean',
            'Sangley Point_tmean',
        ]
    ].mean(axis=1)
    temp['Zone 3 Precipitation'] = temp.loc[
        :,
        [
            'Tayabas_rainfall',
            'Tanay_rainfall',
            'Sangley Point_rainfall',
        ]
    ].mean(axis=1)

    # Trimming outliers for every 30 day window
    window = 30 * 24
    for col in temp.columns[-6:]:
        temp[col] = trim_outliers(
            temp, col, window=window)

    # Fill in 1 hour gaps
    for col in temp.columns[-6:]:
        temp[col] = impute_1h_gaps(temp, col)

    # Fill in 1-168 h gaps
    temp = clean_1_168h(temp, temp.columns[-6:])

    # Save to CSV
    temp.to_csv('Temp_Precipitation.csv')


def add_time_features(df):
    '''Get time features like month of year, day of week and hour of day'''
    import numpy as np
    import pandas as pd

    import datetime
    from datetime import timedelta
    from datetime import datetime as dt

    # adding hour (1-24), day of week (1-7), and month (1-12) as features
    # also denote if weekday or weekend as a feature
    # convert epochs to datetime
    df['month'] = [
        dt.fromtimestamp(df.index[i]).month for i in range(len(df))
    ]
    df['day_of_week'] = [
        dt.fromtimestamp(df.index[i]).isoweekday() for i in range(len(df))
    ]
    df['hour'] = [
        dt.fromtimestamp(df.index[i]).hour for i in range(len(df))
    ]
    df['is_weekday'] = [
        0 if dt.fromtimestamp(
            df.index[i]).isoweekday() >= 6 else 1 for i in range(len(df))
    ]

    # Get dummy variables
    df = pd.get_dummies(df, columns=['month', 'day_of_week', 'hour'])

    return df


def demand_per_zone(zone='Zone 1'):
    '''Get demand/target data for the specified zone'''
    import numpy as np
    import pandas as pd

    demand_df = pd.read_csv(zone + ' RTX', index_col='date')
    demand_df.columns = ['Demand']
    return demand_df


def population_data_per_zone(zone='Zone 1'):
    '''Get population data for the specified zone'''
    import numpy as np
    import pandas as pd

    # Load the raw population counts
    population = pd.read_csv('population_per_zone.csv', index_col='date')
    population = population[zone]
    population.name = 'Population'

    # Load the population density data
    population_density = pd.read_csv(
        'population_density_per_zone.csv', index_col='date')
    population_density = population_density[zone]
    population_density.name = 'Population Density'

    # Merge the two series into a dataframe
    population_final = pd.merge(
        population, population_density, left_index=True, right_index=True)

    return population_final


def add_gdp_per_zone(zone='Zone 1'):
    '''Get the processed gdp for a specified zone'''
    import numpy as np
    import pandas as pd

    gdp = pd.read_csv('Regional GDP.csv', index_col='date')
    ret = gdp.loc[:, zone]
    ret.name = 'GDP'
    return ret


def add_holidays(consolidated_df):
    '''Add holiday data to the consolidated data'''
    import numpy as np
    import pandas as pd

    holidays_df = pd.read_csv('Philippine Holidays.csv', index_col='date')
    return pd.merge(consolidated_df, holidays_df, how='left',
                    left_index=True, right_index=True)


def add_temp_precipitation_per_zone(zone='Zone 1'):
    '''Get the Temperature (C) and Precipitation (mm) for a specified zone.'''
    import numpy as np
    import pandas as pd

    temp = pd.read_csv('Temp_Precipitation.csv', index_col='date')
    temp = temp.loc[:, [zone + ' Temperature', zone + ' Precipitation']]
    temp.columns = ['Temperature (C)', 'Precipitation (mm)']

    return temp


def add_typhoon_data_per_zone(zone='Zone 1'):
    '''Add typhoon data to the consolidated data'''
    import numpy as np
    import pandas as pd

    typhoon_df = pd.read_csv('typhoons_per_zone.csv', index_col='date')
    typhoon_df = typhoon_df.loc[
        :,
        [zone + ' wind', zone + ' pres', zone + ' sshs']]
    typhoon_df.columns = ['Wind', 'Pressure', 'SSHS']

    return typhoon_df


def add_earthquake_data_per_zone(zone='Zone 1'):
    '''Add earthquake data to the consolidated data'''
    import numpy as np
    import pandas as pd

    earthquakes_df = pd.read_csv('earthquake_per_zone.csv', index_col='date')
    earthquakes_df = earthquakes_df.loc[:, [zone + ' mag', zone + ' depth']]
    earthquakes_df.columns = ['Magnitude', 'Depth']

    return earthquakes_df


def consolidate_data_for_zone(zone='Zone 1'):
    '''Consolidate Data Sources for the specified zone'''
    import numpy as np
    import pandas as pd

    consolidated_data = demand_per_zone(zone)
    consolidated_data = add_time_features(consolidated_data)
    consolidated_data = add_holidays(consolidated_data)

    population_data = population_data_per_zone(zone)
    consolidated_data = pd.merge(consolidated_data, population_data,
                                 left_index=True, right_index=True)

    gdp_data = add_gdp_per_zone(zone)
    consolidated_data = pd.merge(consolidated_data, gdp_data, how='left',
                                 left_index=True, right_index=True)
    
    earthquake_data = add_earthquake_data_per_zone(zone)
    consolidated_data = pd.merge(consolidated_data, earthquake_data,
                                 how='left', left_index=True,
                                 right_index=True)
    consolidated_data['Magnitude'].fillna(0, inplace=True)
    consolidated_data['Depth'].fillna(0, inplace=True)
    
    typhoon_data = add_typhoon_data_per_zone(zone)
    consolidated_data = pd.merge(consolidated_data, typhoon_data,
                                 how='left', left_index=True,
                                 right_index=True)
    consolidated_data['Wind'].fillna(0, inplace=True)
    consolidated_data['Pressure'].fillna(0, inplace=True)
    consolidated_data['SSHS'].fillna(-2, inplace=True)

    if zone in ['Zone 1', 'Zone 2', 'Zone 3']:
        temp_data = add_temp_precipitation_per_zone(zone)
        consolidated_data = pd.merge(consolidated_data, temp_data, how='left',
                                     left_index=True, right_index=True)

    return consolidated_data


def generate_consolidated_data_for_all_zones():
    '''Generate the CSV files for all 8 zones'''
    import numpy as np
    import pandas as pd

    dates = prepare_demand_data()
    extract_gdp_per_capita()
    extract_holidays(dates)
    prepare_population_data(dates)
    prepare_temp_precipitation_data()
    prepare_earthquake_data()
    incorporate_typhoon_data()
    for i in range(1, 9):
        df = consolidate_data_for_zone('Zone ' + str(i))
        df.to_csv('Zone ' + str(i) + ' Consolidated Data.csv')
        
