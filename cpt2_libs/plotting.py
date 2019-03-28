def plot_demand(x, y, label, metric_value, xlabel = "Date",
                 ylabel= "Power Demand (MW)", figure_size=(10,6), 
                 title = "This is a title", 
                 subtitle = "This is a subtitle",
                 color_switcher = 0, save_file=False):
    '''
    To plot demand over time
    
    PARAMETERS
    ==========
    x, y (arr) : dates/hours and power demand values respectively
    label (str): region/zone in focus
    metric_value (float) : MAPE value input
    color_switcher (int) : to change color
    
    RETURNS
    =======
    plt (matplotlib plot/png) : demand vs. time plot
    
    EXAMPLES
    ========
    df = pd.read_csv('Luzon Demand per Zone_cleaned.csv')
    df_ = df[['date','ZONE 1 RTX']]

    plot_demand(range(len(df_)),df_['ZONE 1 RTX'],
                'Zone 1 RTX',
                11.672, 
                title='ZONE 1 RTX', 
                subtitle='Power Demand over Time');
    
    '''
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    import seaborn as sns

    import datetime
    from datetime import timedelta  
    
    plt.style.use('seaborn-white')
    color = sns.color_palette('Set1') + sns.color_palette('Set2') 
    
    
    # Getting current time
    dt = list(str(datetime.datetime.now()))[:-7]
    dt[13], dt[16]= 'h','m'
    dt.append('s')
    time = "".join(dt)

    # Plotting proper
    plt.figure(figsize=figure_size);
    plt.tight_layout(pad=10);
    plt.plot(x,y,color=color[color_switcher], alpha = 0.8, label=label);
    plt.ylabel(ylabel,size = 12);
    plt.xlabel(xlabel,size = 12);
    
    # Annotations
    plt.title(title, size=20, pad = 27, fontweight='bold', loc='left');
    plt.gcf().text(0.125, 0.905, subtitle, fontsize=15, color='gray');
    plt.gcf().text(0.15, 0.82, 'MAPE: {0:.3f}%'.format(metric_value), 
                   fontsize=15, color='black');
    plt.legend(loc='lower right');
    
    if save_file:
        plt.savefig('{}_{}_{}'.format(time,title,subtitle));
    else:
        pass
    
    return plt;

def plot_performance(y_true, y_preds, dates, metric_value, leng=24, p=0.95,
                     xlabel = "Date", ylabel= "Power Demand (MW)", 
                     figure_size=(10,6), 
                     title = "Actual demand vs forecasted demand", 
                     subtitle = "Model/Approach: ",
                     color_switcher = 0, save_file=False):
    """
    To plot performance of model: actual vs. predicted 
    including percentile error
    
    PARAMETERS
    ==========
    y_true, y_preds (arr) : actual and predicted values
    dates (arr) : list of dates for label purposes
    p (0 < p < 1): error percentile, default is 0.95
    leng (int) : number of hours
    metric_value (float) : MAPE value input
    color_switcher (int) : to change color
    
    RETURNS
    =======
    plt (matplotlib plot/png) : demand vs. time plot
    """
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    import seaborn as sns

    import datetime
    from datetime import timedelta
	
	import metrics

    y_true = np.array(y_true[:leng])
    y_preds = np.array(y_preds[:leng])
    
    plt.style.use('seaborn-white')
    color = sns.color_palette('Set1') + sns.color_palette('Set2') 
    
    # Getting current time
    dt = list(str(datetime.datetime.now()))[:-7]
    dt[13], dt[16]= 'h','m'
    dt.append('s')
    time = "".join(dt)

    # Plotting proper
    plt.figure(figsize=figure_size);
    plt.tight_layout(pad=10);
    
    plt.plot(dates, y_true, color=color[color_switcher], 
             alpha = 0.8, label="Actual");
    plt.plot(dates, y_preds, color=color[color_switcher+1], 
             alpha = 0.8, label="Predicted");
    
    plt.ylabel(ylabel,size = 12);
    plt.xlabel(xlabel,size = 12);
    plt.xticks(rotation=90)
    
    # Show Percentile Error
    yerr = []
    
    for i in np.arange(0, leng):
        errors, mape, ci = metrics.compute_error_mape_ci(y_true[:i+1], y_preds[:i+1])
        y_pred_error = get_percentile(errors, p=p) * y_preds[i]
        yerr.append(y_pred_error)

    plt.fill_between(dates, y_preds-yerr, y_preds +
                     yerr, color="gray", alpha=0.2, label=f"{p} percentile error")
    
    # Annotations
    plt.title(title, size=20, pad = 27, fontweight='bold', loc='left');
    plt.gcf().text(0.125, 0.905, subtitle, fontsize=15, color='gray');
    plt.gcf().text(0.15, 0.82, 'MAPE: {0:.3f}%'.format(metric_value), 
                   fontsize=15, color='black');
    plt.legend(loc='lower right');
    
    if save_file:
        plt.savefig('{}_{}_{}'.format(time,title,subtitle));
    else:
        pass
    
    return plt;