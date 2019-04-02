def mape(actual, pred):
    """
    Computes mean absolute percentage error of actual and predicted values
    
    Parameters
    ----------
    actual : numpy_array
        array of actual values
    pred : numpy_array
        array of predicted values
        
    Returns
    ----------
    mape : int
        computed MAPE
    """
    import numpy as np
    
    return np.abs(np.abs(actual-pred)/actual).mean()

def compute_error_mape_ci(y_true, y_preds):
    """
    Given list of true values and predicted values
    Return errors, MAPE and 95% confidence interval
    """
    import numpy as np
    
    errors = abs(abs(y_preds - y_true)/y_true)
    errors = np.sort(errors)
    MAPE = np.mean(errors)

    # calculate standard error
    SE = np.std(errors)/np.sqrt(len(errors))

    # Calculate 95% confidence interval
    T = 1.96 * SE
    CI_min = MAPE - T
    CI_max = MAPE + T

    return errors, MAPE, (CI_min, CI_max)

def get_percentile(vals, p):
    """
    Given list of error percentage values and percentile p,
    return percentile value using NIST method
    
    Parameters
    ----------
    vals : numpy_array
        array of error values
    p : float
        percentile value
        
    Returns
    ----------
    mape : int
        computed MAPE
    """
    import numpy as np
    
    vals = np.sort(vals)
    N = len(vals)
    kd = p * (N + 1)
    k = int(np.floor(kd))
    d = kd - k

    if p <= 1 / (N+1):
        val = vals[0]
    elif p >= N/(N+1):
        val = vals[-1]
    else:
        val = vals[k - 1] + d * (vals[k] - vals[k-1])
    return val

def get_accuracies_table(y_true, y_preds, forecast_ranges=[24, 24*30, 24*45]):
    """
    Given list of true values, predicted values, and forecast ranges
    Return a dataframe of MAPE, confidence interval of error
    and 95th percentile error for each forecast range

    Arguments
    =========
    y_true - list of true values
    y_preds - list of predicted values
    forecast_ranges - range of forecast ranges in hours

    Returns
    =========
    df_accs = dataframe of MAPE, CI, 95th percentile error per forecast range

    """
    import numpy as np
    import pandas as pd
    
    acc_dict = {"forecast_range (h)": [],
                "MAPE": [], "error_CI": [],
                "95th_percentile_error": []}

    for forecast_range in forecast_ranges:
        errors, mape, ci = compute_error_mape_ci(
            y_true[:forecast_range], y_preds[:forecast_range])
        error_95 = get_percentile(errors, p=p)

        acc_dict["forecast_range (h)"] += [forecast_range]
        acc_dict["MAPE"] += [np.round(mape, 4)]
        acc_dict["error_CI"] += [np.round(ci, 4)]
        acc_dict["95th_percentile_error"] += [np.round(error_95, 4)]

    df_accs = pd.DataFrame.from_dict(acc_dict)
    return df_accs

