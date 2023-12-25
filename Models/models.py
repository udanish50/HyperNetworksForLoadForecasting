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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from openpyxl import Workbook

# LSTM Model 
class LSTM(nn.Module):
    def __init__(self, num_features):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(num_features, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        output = self.fc(x)
       # output = self.fc(x[:, -1, :]) 
        return output
# GRU
class GRU(nn.Module):
    def __init__(self, num_features):
        super(GRU, self).__init__()
        
        self.gru = nn.GRU(input_size=num_features, hidden_size=64, batch_first=True,dropout=0.1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        #output = self.fc(x[:, -1, :])
        output = self.fc(x)
        return output
# Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer.encoder(x)
        output = self.fc(x)
        return output

# RNN Model
class RNN(nn.Module):
    def __init__(self, num_features):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size=num_features, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        output = self.fc(x)  # Here, I'm taking the output from the last time step
        return output
# MLP 
class MLP(nn.Module):
    def __init__(self, num_features, sequence_length):
        super(MLP, self).__init__()
        self.input_dim = num_features * sequence_length
        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32, 24)

    def forward(self, x):
        x = x.view(x.size(0), -1)        
        x = F.relu(self.fc1(x))
        output = self.fc2(x)        
        return output.view(x.size(0), 24, 1)
# NBEats
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, forecast_length, theta_dim, hidden_units, n_hidden_layers):
        super(NBeatsBlock, self).__init__()
        
        self.fc_layers = nn.ModuleList([nn.Linear(input_dim, hidden_units)])
        self.fc_layers.extend(
            [nn.Linear(hidden_units, hidden_units) for _ in range(n_hidden_layers - 1)]
        )
        self.theta_layer = nn.Linear(hidden_units, theta_dim)
        self.backcast_dim = input_dim
        self.forecast_dim = forecast_length
        self.backcast_projection = nn.Parameter(torch.randn((self.backcast_dim, theta_dim)))
        self.forecast_projection = nn.Parameter(torch.randn((self.forecast_dim, theta_dim)))

    def forward(self, x):
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        theta = self.theta_layer(x)
        # Compute backcast and forecast using matrix multiplication
        backcast = theta @ self.backcast_projection.T
        forecast = theta @ self.forecast_projection.T
        
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, input_dim, forecast_length, theta_dim, hidden_units, n_hidden_layers, n_blocks):
        super(NBeats, self).__init__()
        
        self.forecast_length = forecast_length
        
        self.blocks = nn.ModuleList([
            ModifiedNBeatsBlock(input_dim, forecast_length, theta_dim, hidden_units, n_hidden_layers)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the last two dimensions
        forecasts = []
        
        for block in self.blocks:
            backcast, forecast = block(x)
            x = x - backcast
            forecasts.append(forecast)
        
        # Sum the forecasts from all blocks
        output = sum(forecasts)
        return output.view(output.size(0), self.forecast_length, 1)  # Reshape to desired output shape

#Conv1*1
class Conv1x1(nn.Module):
    def __init__(self, num_features):
        super(Conv1x1, self).__init__()
        
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = F.relu(self.conv1(x))
        output = self.conv2(x)
        return output.transpose(1, 2)

  # ARFFN
  class AR_FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(AR_FFNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        output = self.fc(x)
        return output
#HyperNetwork LSTM    
class HyperNetwork(nn.Module):
    def __init__(self, num_features,num_references, kernel_type='none', lstm_hidden_size=128, fc_output=1):
        super(HyperNetwork, self).__init__()
        self.kernel_type = kernel_type
        self.references = nn.Parameter(torch.randn(num_references, num_features))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.degree = nn.Parameter(torch.tensor(2.0)) 
        self.coef0 = nn.Parameter(torch.tensor(3.0))  
        self.alpha = nn.Parameter(torch.tensor(1.0))
        params_lstm = 4 * ((num_features * lstm_hidden_size) + (lstm_hidden_size**2) + lstm_hidden_size)
        
        lstm_weights = params_lstm - (lstm_hidden_size * 4)  # Minus biases as they are separately handled
        lstm_biases = lstm_hidden_size * 4

        fc_weights = lstm_hidden_size * fc_output
        fc_biases = fc_output

        self.total_params = lstm_weights + lstm_biases + fc_weights + fc_biases

        self.fc1 = nn.Linear(num_references, 128)
        self.fc2 = nn.Linear(128, self.total_params)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        
    def linear_kernel(self, x):
        return torch.mm(x, self.references.t())

    def polynomial_kernel(self, x):
        return (torch.mm(x, self.references.t()) + self.coef0) ** self.degree

    def sigmoid_kernel(self, x):
        return torch.tanh(self.alpha * torch.mm(x, self.references.t()) + self.coef0)

    def rbf_kernel(self, x):
        x = x.unsqueeze(1) - self.references.unsqueeze(0)
        return torch.exp(-self.gamma * (x ** 2).sum(dim=-1))
        
    def forward(self, x):
        if self.kernel_type == 'linear':
            x = self.linear_kernel(x)
        elif self.kernel_type == 'polynomial':
            x = self.polynomial_kernel(x)
        elif self.kernel_type == 'sigmoid':
            x = self.sigmoid_kernel(x)
        elif self.kernel_type == 'rbf':
            x = self.rbf_kernel(x)
        else:
            raise ValueError(f"Invalid kernel type: {self.kernel_type}")
        x = swish(self.fc1(x))
        x = self.fc2(x)
        return x


class LSTM(nn.Module):
    def __init__(self, num_features,num_references,kernel_type='rbf', lstm_hidden_size=128):
        super(LSTMBranchHyper, self).__init__()
        self.num_features = num_features
        self.lstm_hidden_size = lstm_hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(self.num_features, self.lstm_hidden_size, batch_first=True,dropout=0.1)
        self.lstm.flatten_parameters() 
        # Fully connected layer to predict 24-hour future energy
        self.fc = nn.Linear(self.lstm_hidden_size, 1)

        self.hypernetwork = HyperNetwork(num_features, num_references, kernel_type, lstm_hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Get weights from the hypernetwork
        # hyper_out = self.hypernetwork(x[:, 0, :])
        
        # Reshaping logic for LSTM weights and biases
        # lstm_start = 0
        # lstm_end = lstm_start + (self.lstm_hidden_size * self.num_features * 4)
        # lstm_weights = hyper_out[:, lstm_start:lstm_end].view(-1, 4 * self.lstm_hidden_size, self.num_features)
        
        # lstm_biases_start = lstm_end
        # lstm_biases_end = lstm_biases_start + (self.lstm_hidden_size * 4 * 2)  # Corrected bias size
        # lstm_biases = hyper_out[:, lstm_biases_start:lstm_biases_end].view(-1, 4 * self.lstm_hidden_size * 2)

        # Update LSTM weights and biases with corrected slicing
        with torch.no_grad():
            self.lstm.weight_ih_l0.copy_(lstm_weights.squeeze())
            self.lstm.bias_ih_l0.copy_(lstm_biases.squeeze()[:4*self.lstm_hidden_size])
            self.lstm.bias_hh_l0.copy_(lstm_biases.squeeze()[4*self.lstm_hidden_size:8*self.lstm_hidden_size])

        # LSTM forward pass
        x, _ = self.lstm(x)
        # Fully connected layer to predict the next 24 hours
        output = self.fc(x)
        return output
