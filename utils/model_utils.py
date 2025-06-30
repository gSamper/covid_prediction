import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

class StackedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, output_size=1, dropout=0.0):
        super(StackedLSTM, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.relu = nn.ReLU()  # To prevent vanishing gradients

    def forward(self, x):
        out, _ = self.lstm(x)  # Get LSTM output
        out = out[:, -1, :]  # Take last timestep's output
        out = self.fc(out)  # Fully connected layer
        # return self.relu(out)  # Apply ReLU activation
        return out
import torch.nn as nn

class StackedRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, output_size=1, dropout=0.0):
        super(StackedRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.rnn(x)  # Sortida del RNN
        out = out[:, -1, :]   # Agafem només la sortida de l'últim timestep
        out = self.fc(out)
        # return self.relu(out)
        return out

    
def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

def init_model(input_size:int, hidden_size:int, num_layers:int, output_size:int, device: torch.device) -> nn.Module:
    """Instantiates the model and initializes weights.
    The model is a stacked LSTM with the specified input size, 
    hidden size, number of layers, and output size.

    Args:
        input_size (int): Input size to the LSTM layer.
        hidden_size (int): Hidden size of the LSTM layer.
        num_layers (int): Number of LSTM layers.
        output_size (int): Output size of the model.
        device (torch.device): Device to which the model should be moved.

    Returns:
        nn.Module: The initialized model.
    """
    model = StackedLSTM(input_size, hidden_size, num_layers, output_size)
    model.apply(init_weights)
    return model.to(device)

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    print_every: int = 5
) -> tuple[list[float], list[float], list[np.ndarray], list[np.ndarray]]:
    """
    Trains a PyTorch model using the provided training data loader and evaluates it on the test data loader.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        criterion (torch.nn.Module): Loss function to be used for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device on which computation will be performed (e.g., 'cuda' or 'cpu').
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        print_every (int, optional): Frequency (in epochs) to print training and validation loss. Defaults to 5.

    Returns:
        tuple: 
            - test_predictions (list[np.ndarray]): Predictions made on the test set in the final epoch.
            - test_actuals (list[np.ndarray]): Ground truth values from the test set.
            - train_losses (list[float]): List of average training losses per epoch.
            - val_losses (list[float]): List of RMSE values computed on the test set per epoch.
    """
    # Store losses
    train_losses = []
    val_losses = []  # This will now store RMSE on test set per epoch

    # Start training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Only keep predictions and inputs from the last epoch
        if epoch == num_epochs - 1:
            predictions_last = []
            actuals_last = []
            inputs_last = []

        # Training loop
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if epoch == num_epochs - 1:
                predictions_last.extend(y_pred.cpu().detach().numpy())
                actuals_last.extend(y_batch.cpu().numpy())
                inputs_last.extend(X_batch.cpu().numpy())

        # Compute average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- RMSE Validation loop on test set ---
        model.eval()
        test_predictions = []
        test_actuals = []


        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)

                test_predictions.extend(y_pred.cpu().numpy().reshape(-1))
                test_actuals.extend(y_batch.cpu().numpy().reshape(-1))

        # Compute RMSE as validation loss
        rmse_val = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        val_losses.append(rmse_val)

        # Print losses
        if (epoch + 1) % print_every == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, RMSE (Test): {rmse_val:.4f}")
        
    return test_predictions, test_actuals, train_losses, val_losses