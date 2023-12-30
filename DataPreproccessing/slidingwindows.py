import os
import torch
import plotly.graph_objects as go
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

def minmax_scale(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return scaled_tensor, tensor_min, tensor_max

def inverse_minmax_scale(scaled_tensor, tensor_min, tensor_max):
    original_tensor = scaled_tensor * (tensor_max - tensor_min) + tensor_min
    return original_tensor
    
def smape(y_true, y_pred):
    return 100 * torch.mean(2 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred))).item()

def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    
def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

import pandas as pd

window_size = 24
horizon = 24
train_prop = 0.6
val_prop = 0.2

df = pd.read_excel('Essex.xlsx')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)
df['Temperature'].fillna(method='backfill', inplace=True)
df['Temperature'].fillna(method='ffill', inplace=True)

agg_rules = {
    'Temperature': 'last',
    'kWh': 'sum'
}

df = df.resample('H', closed='right').apply(agg_rules)
df['Day_of_Year'] = df.index.dayofyear
df['Day_of_Month'] = df.index.day
df['Day_of_Week'] = df.index.dayofweek
df['Hour'] = df.index.hour
df['Temperature'].fillna(method='ffill', inplace=True)
for idx, row in df.iterrows():
    if row['kWh'] == 0:
        start_idx = max(0, df.index.get_loc(idx) - window_size)
        end_idx = df.index.get_loc(idx)
        avg_kWh = df.iloc[start_idx:end_idx]['kWh'].mean()
        
        df.at[idx, 'kWh'] = avg_kWh

cols = [col for col in df.columns if col != 'kWh'] + ['kWh']
df = df[cols]
df
df2 = pd.read_excel('Perth.xlsx')
df2['Datetime'] = pd.to_datetime(df2['Datetime'])
df2.set_index('Datetime', inplace=True)
df2['Temperature'].fillna(method='backfill', inplace=True)
df2['Temperature'].fillna(method='ffill', inplace=True)

agg_rules = {
    'Temperature': 'last',
    'kWh': 'sum'
}

df2 = df2.resample('H', closed='right').apply(agg_rules)
df2['Day_of_Year'] = df2.index.dayofyear
df2['Day_of_Month'] = df2.index.day
df2['Day_of_Week'] = df2.index.dayofweek
df2['Hour'] = df2.index.hour
df2['Temperature'].fillna(method='ffill', inplace=True)
for idx, row in df2.iterrows():
    if row['kWh'] == 0:
        start_idx = max(0, df.index.get_loc(idx) - window_size)
        end_idx = df2.index.get_loc(idx)
        avg_kWh = df2.iloc[start_idx:end_idx]['kWh'].mean()
        
        # Set the value in the DataFrame
        df2.at[idx, 'kWh'] = avg_kWh
cols = [col for col in df2.columns if col != 'kWh'] + ['kWh']
df2 = df2[cols]
class SlidingWindowsDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size=24, horizon=24):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        
        self.feature_mins = self.data.min(axis=0).values
        self.feature_maxs = self.data.max(axis=0).values
    
    def __len__(self):
        return len(self.data) - (self.window_size + self.horizon) + 1

    def __getitem__(self, idx):
        window = self.data[idx: idx + self.window_size + self.horizon]
        past_all_features = window[:self.window_size]
        future_energy = window[self.window_size:, -1:]
        
        scaled_past_all_features = (past_all_features - self.feature_mins) / (self.feature_maxs - self.feature_mins)
        
        scaled_future_energy, energy_min, energy_max = minmax_scale(future_energy)

        return scaled_past_all_features, scaled_future_energy, (energy_min, energy_max)


import torch

df_dict = {
    'Essex': df,
    'Perth': df2
}

train_datasets = {}
val_datasets = {}
test_datasets = {}

for dataset_name, dataset_df in df_dict.items():
    # Convert to tensor and perform the split
    data_tensor = torch.from_numpy(dataset_df.values).float()

    data_shape = data_tensor.shape[0]
    train_end = int(data_shape * train_prop)
    val_end = int(data_shape * (train_prop + val_prop))

    train_data = data_tensor[:train_end]
    val_data = data_tensor[train_end:val_end]
    test_data = data_tensor[val_end:]
    
    train_datasets[dataset_name] = SlidingWindowsDataset(train_data, window_size, horizon)
    val_datasets[dataset_name] = SlidingWindowsDataset(val_data, window_size, horizon)
    test_datasets[dataset_name] = SlidingWindowsDataset(test_data, window_size, horizon)

