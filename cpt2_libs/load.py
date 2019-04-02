
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
	import pandas as pd
	
    # read excel as is
    demand_df = pd.read_excel(path)

    # read demand
    demand_df.columns = ["date"] + list(demand_df.columns[1:])

    # change the date to datetime format
    demand_df.date = pd.to_datetime(demand_df.date)

    # return
	return demand_df

	