import pandas as pd
import numpy as np

from typing import Tuple, Union, Type
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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

def create_sequences_new(data: pd.DataFrame, target_column: str, endog_columns :list, policy_columns :list, seq_length=3):
    """Creates sequence data for LSTM models.
    This function takes a DataFrame and creates sequences of features and targets for 
    time series forecasting.
    Used country by country inside the scale_data_per_country function.
    The sequences are created by taking the past `seq_length` days of features 
    to predict the target for the next day.

    Args:
        data (pd.DataFrame): Data for training from one country in specific
        target_column (str): Column name for the target column of the data df.
        endog_columns (list): List of names of the columns realted to cases.
        exog_columns (list): List of names of the columns related to policies.
        seq_length (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    X_endog, X_exog, y = [], [], []

    for i in range(len(data) - seq_length):
        # Sequence of endogenous features (e.g., new_cases, deaths)
        x_seq = data[endog_columns].iloc[i: i + seq_length].values  # Shape: (seq_length, num_endog_features)
        X_endog.append(x_seq)

        # Policy features at prediction time (exogenous)
        x_policy = data[policy_columns].iloc[i + seq_length].values  # Shape: (num_policy_features,)
        X_exog.append(x_policy)

        # Target value at prediction time
        y_target = data[target_column].iloc[i + seq_length]
        y.append(y_target)
    return X_endog, X_exog, y

def scale_data_per_country_new(df: pd.DataFrame, countries: list, feature_columns: list, 
                            target_column: str, scaler: Union[Type[MinMaxScaler], 
                            Type[StandardScaler]], endog_columns: list, policy_columns: list, 
                            seq_length: int = 4):
    """
    Scales the feature and target columns separately for each country using the specified scaler
    and prepares the data for sequence modeling.

    Args:
        df (pd.DataFrame): The input DataFrame containing location, features, and target column.
        countries (list): List of country names to process separately.
        feature_columns (list): List of feature column names to be scaled.
        target_column (str): Name of the target column to be scaled.
        scaler (Union[Type[MinMaxScaler], Type[StandardScaler]]): The scaler class to be used for normalization.
        seq_length (int): The sequence length for LSTM input.

    """
    X_endog, X_exog, y_list = [], [], []

    for country in countries:
        # Filter data per country
        country_data = df[df["location"] == country]
        # country_data = country_data.drop(columns=["location"]).reset_index(drop=True)

        # Apply separate scalers per country
        scaler_X = scaler
        scaler_y = scaler

        # Scale features and target separately
        features_scaled = scaler_X.fit_transform(country_data[feature_columns])
        target_scaled = scaler_y.fit_transform(country_data[target_column].values.reshape(-1, 1))

        # Convert back to DataFrame
        scaled_features_df = pd.DataFrame(features_scaled, columns=feature_columns)
        scaled_target_df = pd.DataFrame(target_scaled, columns=[target_column])

        # Merge back
        scaled_data = pd.concat([scaled_features_df, scaled_target_df], axis=1)

        # Create sequences using the scaled data
        X_endog_part, X_exog_part, y = create_sequences_new(scaled_data, target_column, endog_columns, policy_columns, seq_length)
        X_endog.append(X_endog_part)
        X_exog.append(X_exog_part)
        y_list.append(y)

    # Combine sequences from all countries
    X_all_endog = np.vstack(X_endog)
    X_all_exog = np.vstack(X_exog)
    y_all = np.hstack(y_list)

    return y_all, X_all_endog, X_all_exog

def get_pd_from_pkl(path: str) -> pd.DataFrame:
    
    df = pd.read_pickle(path)
    return df