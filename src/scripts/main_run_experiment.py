import argparse
import torch
import pandas as pd
import numpy as np
from torch import nn

from utils.data_utils import get_pd_from_pkl
from utils.model_utils import init_model, train_model
from src.run_experiment import get_all_but_one_data, get_data_laoders
from utils.plot_utils import plot_actual_vs_predicted


def evaluate_model(
    test_predictions: list[np.ndarray],
    test_actuals: list[np.ndarray]
) -> dict:
    """
    Evaluates the model performance on the validation/test set using common regression metrics.

    Args:
        test_predictions (list[np.ndarray]): Predicted values from the model on the validation set.
        test_actuals (list[np.ndarray]): Ground truth values from the validation set.

    Returns:
        dict: A dictionary containing RMSE, MAE, and R² score.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # Convert to numpy arrays in case they're lists of arrays
    y_pred = np.array(test_predictions).flatten()
    y_true = np.array(test_actuals).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def check_country_in_data(country: str, df: pd.DataFrame):
    """
    Checks if a given country is present in the 'location' column of the dataset.

    Args:
        country (str): Name of the country to check.
        df (pd.DataFrame): DataFrame containing a 'location' column with country names.

    Raises:
        ValueError: If the specified country is not found in the dataset.

    Returns:
        None
    """
    if country not in df['location'].unique():
        raise ValueError(f"The country '{country}' is not present in the dataset.")


def main(target_country: str, df: pd.DataFrame):

    print("Running expeeriment for")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
  
    
    feature_columns = ['C', 'E', 'G', 'S', 'weekly_deaths', 'weekly_cases']
    X_train, y_train, X_test, y_test = get_all_but_one_data(target_country, df, 
                                                            feature_columns, target_column="target")
    train_loader, test_loader = get_data_laoders(X_train, y_train, X_test, y_test, device, batch_size = 32)
    
    input_size= X_train.shape[2]
    hidden_size=64
    num_layers= 2
    output_size=1
    learning_rate=0.001
    weight_decay = 1e-4
    model = init_model(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, output_size=output_size,
                        device=device)
    
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    test_predictions, test_actuals, train_losses, val_losses = train_model(model, train_loader, 
                                                                           test_loader, criterion, 
                                                                           optimizer, device, 
                                                                           num_epochs=10, print_every=5)
    metrics = evaluate_model(test_predictions, test_actuals)
    print(metrics)
    plot_actual_vs_predicted(
        test_actuals,
        test_predictions,
        title=f"{target_country} – Actual vs Predicted",
        xlabel="Time Step",
        ylabel="New Cases",
        save_path=f"results/run_one_country_exp/{target_country}_prediction_plot.png"
        )
    return 0
    
if __name__ == "__main__":
    
    # ---Get the target country from command line arguments---
    parser = argparse.ArgumentParser(description="Run experiment for a target country.")
    parser.add_argument("--target_country", "-tc", type=str, required=True, help="Name of the target country")
    args = parser.parse_args()
    country = args.target_country
    
    
    #---Get the data---
    PATH = r"data\training_data\mod_data\south_africa_0.pkl"
    PATH = r"data\training_data\training_added_feats_v1.pkl"
    df = get_pd_from_pkl(PATH)


    check_country_in_data(country, df)
    main(target_country=country, df=df)