import matplotlib.pyplot as plt

def plot_actual_vs_predicted(actuals, preds, title, xlabel, ylabel, save_path=None, figsize=(12, 6)):
    """
    Plots actual vs predicted values.

    Args:
        actuals (array-like): True values.
        preds (array-like): Predicted values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str, optional): Path to save the figure. If None, the figure is not saved. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (12, 6).

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    plt.plot(actuals, label='Actual', linewidth=2)
    plt.plot(preds, label='Predicted', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    plt.show()