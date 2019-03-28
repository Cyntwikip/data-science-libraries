def macd(quotes_df, longer_ma = 26, shorter_ma = 12):
    df_macd = quotes_df
    df_macd['macd - longer_ma'] = df_macd['Close'].rolling(longer_ma).mean()
    df_macd['macd - shorter_ma'] = df_macd['Close'].rolling(shorter_ma).mean()
    df_macd['macd - diff'] = df_macd['macd - longer_ma'] - df_macd['macd - shorter_ma']
    df = df_macd
    return df
  
def rsi(quotes_df, period = 14):
    df_rsi = quotes_df.reset_index()
    df_rsi['rsi - changes'] = df_rsi['Close'] - df_rsi['Close'].shift(1)

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
    df = df_rsi.drop(['index', 'rsi - changes', 'rsi - gains','rsi - losses',
                      'rsi - avg gains', 'rsi - avg losses',
                      'rsi - rs'], axis = 1)
    return df
  
def ema(quotes_df, period = 10):
    multiplier = (2/(period + 1))
    
    df_ema = quotes_df
    df_ema['ema - ma'] = df_ema['Close'].rolling(period).mean()
    df_ema['ema - ema'] = (multiplier*(df_ema['Close']-df_ema['Close'].shift(1))+df_ema['ema - ma'].shift(1))

    df = df_ema.drop('ema - ma', axis = 1)
    return df
 
def cci(quotes_df, period = 20):

    
    df_cci = quotes_df
    df_cci['cci - typical price'] = (df_cci['High'] + df_cci['Low'] + df_cci['Close'])/3
    df_cci['cci - sma'] = df_cci['cci - typical price'].rolling(period).mean()
    mean_abs_dev = (df_cci['cci - typical price'] - df_cci['Close']).sum()/period
    
    df_cci['cci - cci'] = (1/0.015)*((df_cci['cci - typical price'] - df_cci['cci - sma'])/mean_abs_dev)

    df = df_cci.drop(['cci - typical price','cci - sma'], axis = 1)
    return df
  
def bollinger(quotes_df, period = 20):
    
    df_bollinger = quotes_df
    df_bollinger['bollinger - middle'] = df_bollinger['Close'].rolling(period).mean()
    df_bollinger['bollinger - std'] = df_bollinger['Close'].rolling(period).std()
    
    df_bollinger['bollinger - upper'] = df_bollinger['bollinger - middle'] + 2*df_bollinger['bollinger - std']
    df_bollinger['bollinger - lower'] = df_bollinger['bollinger - middle'] - 2*df_bollinger['bollinger - std']


    df = df_bollinger.drop('bollinger - std', axis = 1)
    return df
  
 # THE FOLLOWING ARE THE MORE IMPORTANT FUNCTIONS:

def calculate_ti(new_value, df, current_row = -1, current_col = "Close", past_start_col="demand_-1h"):
    
    '''
    To calculate technical indicators
    
    PARAMETERS
    ==========
    new_value (float) : newly predicted demand data
    df (dataframe) : original dataframe
    current_row (arr) : row list of demand values past 24 hours
    current_col (str) : column name of current demand
    past_start_col (str) : starting column name of past demand
    
    RETURNS
    =======
    b (dictionary) : calculated technical indicators 
    
    '''
    col_names = df.columns.tolist()[2:]
    
    new = pd.DataFrame(data={'Close':[new_value]})
    sample = df.append(new, sort=False)
    sample = calculate_past(sample)
    
    demand = pd.DataFrame(sample.iloc[current_row]).T
    index = df.columns.get_loc(past_start_col)

    a = list(demand.iloc[0,index:][::-1]) # In reverse to mimic chronological order 
    a.append(demand['Close'].values[0])
    a = pd.DataFrame(data={'Close':a})
    
    b = macd(a, longer_ma=24, shorter_ma=12)
    b = ema(b) 
    b = rsi(b) 
    b = bollinger(b)
    
    past_demand = pd.DataFrame(sample.iloc[current_row, index:]).T
    past_demand = past_demand.reset_index()
    past_demand = past_demand.iloc[:,1:]
    
    
    b = pd.DataFrame(b.iloc[-1,:]).T
    b = b.reset_index()
    
    b = b.iloc[:,1:]
    
    b = pd.concat([b, past_demand], axis = 1, ignore_index=True)

    b.columns = col_names
    
    return b
  
 def append_new_ti(new_ti, df):
    '''
    To append newly calculated technical indicators to main dataframe
    
    PARAMETERS
    ==========
    new_ti (dataframe) : output of calculate_ti() function
    df (dataframe) : original dataframe
    
    RETURNS
    =======
    df (dataframe) : dataframe with the appended indicators
    '''

    df = df.append(new_ti, sort=False, ignore_index=True)
    
    return df
  
  def calculate_past(df):
    '''
    To calculate past 24 hours
    
    PARAMETERS
    ==========
    df (dataframe) : input
    
    RETURNS
    =======
    df (dataframe) : 
    '''
    
    ## Getting past 24 hours
    for i in range(1,25):
        df['demand_-{}h'.format(i)] = df['Close'].shift(i)
        
    return df
  
  
 def calculate_append_ti(new_value, df):
    '''
    NOTE: NEED TO RUN PREREQUISITE FUNCTIONS
    
    To calculate technical indicators based on new demand and append to original
    
    PARAMETERS
    ==========
    new_value (float) : new demand value from prediction
    df (dataframe) : original dataframe
    
    RETURNS
    =======
    df (dataframe) : dataframe with the new data
    '''
    df_ = calculate_ti(new_value, df)
    df = append_new_ti(df_, df)
    return df 
   