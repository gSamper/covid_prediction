import pandas as pd

def format_policy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function to format the policy data from OxCGRT. 
    First, it converts the data to a long format, 
    then it converts the date column to a datetime format, 
    and finally, it sorts the data by country and date.

    Args:
        df (pd.DataFrame): OxCGRT policy data.

    Returns:
        pd.DataFrame: Formatted policy data.
    """
    
    #variables que no son dates
    id_vars = ['CountryCode', 'CountryName', 'RegionCode', 'RegionName', 'CityCode',
        'CityName', 'Jurisdiction']
    #convertir a long format, una fila per cada data
    df_long = df.melt(id_vars=id_vars, var_name='Date', value_name='Value')
    #convertir a datetime format
    df_long['Date'] = pd.to_datetime(df_long['Date'], format='%d%b%Y')
    #ordenar per pais i data
    df_long = df_long.sort_values(by=['CountryName', 'Date'])
    
    return df_long