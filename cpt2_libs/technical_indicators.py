def macd(quotes_df, target_col = 'Demand', longer_ma = 24, shorter_ma = 12, ):
    """
    Compute the Moving Average Convergence Divergence
    
    Parameters
    ----------
    quotes_df : Pandas Dataframe
        time series
    target_col : String
        column name of interest
    longer_ma : int
        longer moving average period
    shorter_ma : int
        shorter moving average period
    
    Returns
    ----------
    df : Pandas DataFrame
        technical indicator calculations
    """
    
    df_macd = quotes_df
    df_macd['macd - longer_ma'] = df_macd[target_col].rolling(longer_ma).mean()
    df_macd['macd - shorter_ma'] = df_macd[target_col].rolling(shorter_ma).mean()
    df_macd['macd - diff'] = df_macd['macd - longer_ma'] - df_macd['macd - shorter_ma']
    df = df_macd
    return df

def rsi(quotes_df, target_col = 'Demand', period = 14):
    """
    Compute the Relative Strength Index
    
    Parameters
    ----------
    quotes_df : Pandas Dataframe
        time series
    target_col : String
        column name of interest
    period : int
        window period for calculating RSI
    
    Returns
    ----------
    df : Pandas DataFrame
        technical indicator calculations
    """
    import pandas as pd
    import numpy as np
    
    df_rsi = quotes_df.reset_index()
    df_rsi['rsi - changes'] = df_rsi[target_col] - df_rsi[target_col].shift(1)

    gains = [0]
    losses = [0]

    for i in range(len(df_rsi)):
        change = df_rsi['rsi - changes'].iloc[i]

        if change >= 0:
            losses.append(0)
            gains.append(np.abs(change))

        elif change < 0:
            losses.append(np.abs(change))
            gains.append(0)

    df_rsi['rsi - losses'] = pd.Series(losses)
    df_rsi['rsi - gains'] = pd.Series(gains)

    df_rsi['rsi - avg gains'] = df_rsi['rsi - gains'].rolling(period).mean()
    df_rsi['rsi - avg losses'] = df_rsi['rsi - losses'].rolling(period).mean()

    df_rsi['rsi - rs'] = df_rsi['rsi - avg gains']/df_rsi['rsi - avg losses']
    df_rsi['rsi - rsi'] = 100 - (100/(1 + df_rsi['rsi - rs']))
    df = df_rsi.drop(['rsi - changes', 'rsi - gains','rsi - losses',
                      'rsi - avg gains', 'rsi - avg losses',
                      'rsi - rs'], axis = 1)
    return df

def ema(quotes_df, target_col = 'Demand', period = 10):
    """
    Compute the Exponential Moving Averages
    
    Parameters
    ----------
    quotes_df : Pandas Dataframe
        time series
    target_col : String
        column name of interest
    period : int
        window period for calculating EMA
    
    Returns
    ----------
    df : Pandas DataFrame
        technical indicator calculations
    """
    multiplier = (2/(period + 1))
    
    df_ema = quotes_df
    df_ema['ema - ma'] = df_ema[target_col].rolling(period).mean()
    df_ema['ema - ema'] = (multiplier*(df_ema[target_col]-df_ema[target_col].shift(1))+df_ema['ema - ma'].shift(1))

    df = df_ema.drop('ema - ma', axis = 1)
    return df

def bollinger(quotes_df, target_col = 'Demand', period = 20):
    """
    Compute the Bollinger Bands values
    
    Parameters
    ----------
    quotes_df : Pandas Dataframe
        time series
    target_col : String
        column name of interest
    period : int
        window period for calculating Bollinger Bands
    
    Returns
    ----------
    df : Pandas DataFrame
        technical indicator calculations
    """
    
    df_bollinger = quotes_df
    df_bollinger['bollinger - middle'] = df_bollinger[target_col].rolling(period).mean()
    df_bollinger['bollinger - std'] = df_bollinger[target_col].rolling(period).std()
    
    df_bollinger['bollinger - upper'] = df_bollinger['bollinger - middle'] + 2*df_bollinger['bollinger - std']
    df_bollinger['bollinger - lower'] = df_bollinger['bollinger - middle'] - 2*df_bollinger['bollinger - std']


    df = df_bollinger.drop('bollinger - std', axis = 1)
    return df
