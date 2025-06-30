import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler


from utils.data_utils import scale_data_per_country
from utils.model_utils import StackedLSTM, init_weights

def get_all_but_one_data(target_country: str, df: pd.DataFrame, feature_columns: list, target_column: str):
    
    full_data = df.dropna().copy()
    full_data = full_data[['location', 'C', 'E', 'G', 'S', 'weekly_deaths', 'weekly_cases']]
    full_data.loc[:, 'target'] = full_data['weekly_cases']
    
    train_data = full_data[full_data['location'] != target_country].copy()
    test_data = full_data[full_data['location'] == target_country].copy()
    
    y_train, X_train = scale_data_per_country(
                                        df=train_data, feature_columns=feature_columns,
                                        countries=list(train_data['location'].unique()),
                                        target_column=target_column, scaler=MinMaxScaler()
                                        )
    y_test, X_test = scale_data_per_country(
                                        df=test_data, feature_columns=feature_columns,
                                        countries=list(test_data['location'].unique()),
                                        target_column=target_column, scaler=MinMaxScaler(),
                                        seq_length=4
                                        )
    
    return X_train, y_train, X_test, y_test

def get_data_laoders(X_train, y_train, X_test, y_test, device, batch_size = 32):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1).to(device)

    # Create DataLoader
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader