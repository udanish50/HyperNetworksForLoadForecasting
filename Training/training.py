import pandas as pd
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch import nn
import math
import torch
import torch.nn as nn
import math
# Set the device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print("Training will be performed on GPU")
else:
    print("Training will be performed on CPU")

# Learning Rate
learning_rate = 0.001
def train(model, data_loader, optimizer, loss_fn):
    model.train()
    losses = []
    metrics = {'MAE': [], 'RMSE': [], 'SMAPE': []}

    for inputs, scaled_targets, future_energy_min_max in data_loader:
        inputs, scaled_targets = inputs.to(device), scaled_targets.to(device)
        #print("Scaled Targets:", scaled_targets[0].cpu().numpy())  # Print the first scaled target
        targets_min, targets_max = future_energy_min_max
        targets_min, targets_max = targets_min.to(device), targets_max.to(device)
        #print("Min for targets:", targets_min[0].item())   # Print min value used for scaling for the first target
        #print("Max for targets:", targets_max[0].item())   # Print max value used for scaling for the first target

        outputs = model(inputs)
        #print("Outputs:", outputs[0].detach().cpu().numpy())
        loss = loss_fn(outputs, scaled_targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        targets = inverse_minmax_scale(scaled_targets, targets_min, targets_max)
        #print("Inverse Scaled Targets:", targets[0].cpu().numpy())  # Print the first inverse scaled target
        outputs = inverse_minmax_scale(outputs, targets_min, targets_max)
        #print("Inverse Scaled Outputs:", outputs[0].detach().cpu().numpy())    # Print the first inverse scaled output

        metrics['MAE'].append(mae(outputs, targets).item())
        metrics['RMSE'].append(rmse(outputs, targets).item())
        metrics['SMAPE'].append(smape(outputs, targets))

    return np.mean(losses), {k: np.mean(v) for k, v in metrics.items()}
# Evaluation (for Validation and Testing)
def evaluate(model, data_loader, loss_fn):
    model.eval()
    losses = []
    metrics = {'MAE': [], 'RMSE': [], 'SMAPE': []}

    with torch.no_grad():
        for inputs, scaled_targets, future_energy_min_max in data_loader:
            inputs, scaled_targets = inputs.to(device), scaled_targets.to(device)
            targets_min, targets_max = future_energy_min_max
            targets_min, targets_max = targets_min.to(device), targets_max.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, scaled_targets)
            losses.append(loss.item())

            targets = inverse_minmax_scale(scaled_targets, targets_min, targets_max)
            outputs = inverse_minmax_scale(outputs, targets_min, targets_max)

            metrics['MAE'].append(mae(outputs, targets).item())
            metrics['RMSE'].append(rmse(outputs, targets).item())
            metrics['SMAPE'].append(smape(outputs, targets))

    return np.mean(losses), {k: np.mean(v) for k, v in metrics.items()}

Model = # please define model that you want run, please refer to model.py 
model = Model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-7)
#scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

loss_fn = nn.MSELoss()
# or 
loss_fn = nn.L1Loss()

def train_and_validate(datasets, num_epochs, patience,loss_fn):
    plots_dir = "./directory/"
    
    for dataset_name in datasets['train'].keys():
        print(f"Training for dataset: {dataset_name}")
        
        train_loader = DataLoader(datasets['train'][dataset_name], batch_size=1, shuffle=False)
        val_loader = DataLoader(datasets['val'][dataset_name], batch_size=1, shuffle=False)
        
        best_val_loss = np.inf
        epochs_no_improve = 0

        train_loss_history = []
        val_loss_history = []
        train_mae_history = []
        val_mae_history = []
        train_rmse_history = []
        val_rmse_history = []
        train_smape_history = []
        val_smape_history = []

        for epoch in range(num_epochs):
            train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn)
            val_loss, val_metrics = evaluate(model, val_loader, loss_fn)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch + 1} epochs for dataset {dataset_name}.")
                break

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_mae_history.append(train_metrics['MAE'])
            val_mae_history.append(val_metrics['MAE'])
            train_rmse_history.append(train_metrics['RMSE'])
            val_rmse_history.append(val_metrics['RMSE'])
            train_smape_history.append(train_metrics['SMAPE'])
            val_smape_history.append(val_metrics['SMAPE'])

            print(f"Epoch {epoch + 1}/{num_epochs} for dataset {dataset_name}")
            print(f"Train Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f}")
            print(f"Train MAE: {train_metrics['MAE']:.2f}, Validation MAE: {val_metrics['MAE']:.2f}")
            print(f"Train RMSE: {train_metrics['RMSE']:.2f}, Validation RMSE: {val_metrics['RMSE']:.2f}")
            print(f"Train SMAPE: {train_metrics['SMAPE']:.2f}, Validation SMAPE: {val_metrics['SMAPE']:.2f}")

        # Create plots and save them for each dataset
        def plot_metrics(history1, history2, title, ylabel, filename):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history1, mode='lines', name='Train'))
            fig.add_trace(go.Scatter(y=history2, mode='lines', name='Validation'))
            fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title=ylabel)
            fig.write_image(f"{plots_dir}/{dataset_name}_{filename}")

        plot_metrics(train_loss_history, val_loss_history, 'Loss vs Epoch', 'Loss', 'drop_loss_plot.png')
        plot_metrics(train_mae_history, val_mae_history, 'MAE vs Epoch', 'MAE', 'drop_mae_plot.png')
        plot_metrics(train_rmse_history, val_rmse_history, 'RMSE vs Epoch', 'RMSE', 'drop_rmse_plot.png')
        plot_metrics(train_smape_history, val_smape_history, 'SMAPE vs Epoch', 'SMAPE', 'drop_smape_plot.png')

        # Save the model for each dataset
        torch.save(model.state_dict(), f'./directory/AR_{loss_fn}_{dataset_name}.pth')

datasets = {
    'train': train_datasets,
    'val': val_datasets,
    'test': test_datasets
}

patience = patience
num_epochs = num_epochs
# Uncomment to train
#train_and_validate(datasets, num_epochs, patience,loss_fn)
