import numpy as np

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
    return np.abs(np.abs(actual-pred)/actual).mean()

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


