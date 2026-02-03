"""Plotting functions."""

import matplotlib.pyplot as plt
from matplotlib.figure import Axes, Figure
from numpy import arange, ndarray
from polars import DataFrame
from scipy.stats import probplot

from constants import PLOTTING_FEATURES, TARGETS


def plot_features_time_series(X_test: DataFrame, features: dict = PLOTTING_FEATURES) -> None:
    """Plots feature time series in three figures, each with three subplots."""
    time: ndarray = X_test["index"].to_numpy()

    for group_name, group_features in features.items():
        fig: Figure; ax: ndarray
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes: ndarray = ax.flatten()
        
        colors: list[str] = ["blue", "orange", "green"]

        for i, (feature, color) in enumerate(zip(group_features, colors)):
            values: ndarray = X_test[feature].to_numpy()
            
            axes[i].plot(time, values, color=color, label=feature, alpha=0.7)
            axes[i].set_ylabel(f"{feature} value")
            axes[i].set_title(f"{feature}")
            axes[i].legend()
            axes[i].grid(True)

        axes[2].set_xlabel("Time")
        fig.suptitle(f"{group_name} time series", y=1.02)
        plt.tight_layout()
        plt.show()


def plot_evals_result(
    evals: dict, metric: str, models: dict, targets: list[str]
) -> None:
    """Plots train and validation evaluation metrics as scatterplots in 
    separate subplots per target."""

    num_targets: int = len(targets)
    _: Figure; axes: ndarray
    _, axes = plt.subplots(num_targets, 1, figsize=(12, 6 * num_targets))
    
    if num_targets == 1:
        axes = [axes]
    
    train_color: str = "blue"
    val_color: str = "red"
    marker_size: int = 4
    
    for target, ax in zip(targets, axes):
        # Target evaluation results.
        e: dict = evals[target]
        # Metric values.
        train_metric: float = e['train'][metric]
        test_metric: float = e['test'][metric]
        
        iterations = range(len(train_metric))
        best_iter: int = models[target].best_iteration
        
        # Add the title as the first legend entry.
        ax.plot([], [], ' ', label=f"{metric} for {target}")
        
        # Training points (blue).
        ax.scatter(
            iterations, 
            train_metric,
            label=f"Training {metric}",
            facecolors="none",
            edgecolors=train_color,  
            marker='o',
            s=marker_size,
        )
        # Validation points (red).
        ax.scatter(
            iterations, test_metric,
            label=f"Validation {metric}",
            color=val_color, 
            marker='s',
            s=marker_size,
        )
        ax.axvline(
            x=best_iter, 
            color='black',
            linestyle="--", 
            alpha=0.5,
            label=f"Best iteration ({best_iter})"
        )
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def prediction_scatter_plot(y_pred: dict, y_test: dict, targets: list[str]) -> None:
    """Plots sector predictions vs. true values as a scatter plot."""
    n_targets: int = len(targets)
    
    # Grid size.
    n_cols: int = 2
    n_rows: int = (n_targets + n_cols - 1) // n_cols
    
    _: Figure; axs: ndarray
    _, axs = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows))
    axes: ndarray = axs.flatten()
    
    # Loop over the targets.
    for target, ax in zip(targets, axes):
        # Extract true and predicted values for the current target
        true: ndarray = y_test[target][target].to_numpy()
        pred: ndarray = y_pred[target][target].to_numpy()
        # Scatter plot.
        ax.scatter(true, pred, marker='.', s=5, alpha=0.5)
        ax.plot(
            [true.min(), true.max()], 
            [true.min(), true.max()], 
            "r--",
        )
        # Axes' limits.
        ax.set_xlim(true.min() - 5, true.max() + 5)
        ax.set_ylim(pred.min() - 5, pred.max() + 5)
        # Labels.
        ax.set_xlabel(f"True {target} (nT)")
        ax.set_ylabel(f"Predicted {target} (nT)")
        # Title
        #ax.set_title(f"True vs. predicted {target} (nT)")
    
    # Hide any unused axes.
    for ax in axes[len(targets):]:
        ax.set_visible(False)
    # Adjust layout and display.
    plt.tight_layout()
    plt.show()


def global_prediction_scatter_plot(y_pred: dict, y_test: dict, target: str) -> None:
    """Plots global prediction vs. true values as a scatter plot."""
    _: Figure; axes: ndarray
    _, axes = plt.subplots(figsize=(8, 6))

    true: ndarray = y_test[target][target].to_numpy()
    pred: ndarray = y_pred[target][target].to_numpy()

    axes.scatter(true, pred, marker='.', s=5, alpha=0.5)
    axes.plot(
        [true.min(), true.max()], 
        [true.min(), true.max()], 
        "r--",
    )
    # Set axes' limits.
    axes.set_xlim(true.min() - 5, true.max() + 5)
    axes.set_ylim(pred.min() - 5, pred.max() + 5)
    # Set labels and title.
    axes.set_xlabel(f"True {target} (nT)")
    axes.set_ylabel(f"Predicted {target} (nT)")
    axes.set_title(f"True vs. predicted {target} (nT)")
    # Adjust layout and display,
    plt.tight_layout()
    plt.show()


def prediction_time_series(y_pred: dict, y_test: dict, targets: list[str]) -> None:
    """Plots sector predictions vs. true values as time series."""
    n_targets: int = len(targets)
    
    # Grid size.
    n_cols: int = 2
    n_rows: int = (n_targets + n_cols - 1) // n_cols

    _: Figure; axs: ndarray
    _, axs = plt.subplots(n_cols, n_rows, figsize=(14, 10), sharex=True) 
    axes: ndarray = axs.flatten()
    
    true_color: str = "blue"
    pred_color: str = "orange"
    
    # Get time from the first target.
    time: ndarray = y_test[targets[0]]["index"].to_numpy()

    for target, ax in zip(targets, axes):
        true: ndarray = y_test[target][target].to_numpy()
        pred: ndarray = y_pred[target][target].to_numpy()
        
        ax.plot(
            time,
            true,
            color=true_color,
            label=f"True {target}",
            alpha=0.7,
        )
        ax.plot(
            time,
            pred,
            color=pred_color,
            linestyle="--",
            label=f"Pred {target}",
            alpha=0.7,
        )
        ax.set_ylabel(f"{target} (nT)")
        ax.set_title(f"True vs. predicted {target} (nT)")
        ax.legend()
        ax.grid(True)
    
    # Set xlabels for the bottom row.
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Time")
    
    # Hide any unused axes.
    for ax in axes[len(targets):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()


def global_prediction_time_series(y_pred: dict, y_test: dict, target: str) -> None:
    """Plots global prediction vs. true values as time series."""
    plt.figure(figsize=(10, 6))
    
    # Extract time axis and values
    time: ndarray = y_test[target]["index"].to_numpy()
    true_vals: ndarray = y_test[target][target].to_numpy()
    pred_vals: ndarray = y_pred[target][target].to_numpy()

    # Plot
    plt.plot(time, true_vals, color="blue", label=f"True {target}", alpha=0.7)
    plt.plot(time, pred_vals, color="orange", linestyle="--", label=f"Pred {target}", alpha=0.7)

    plt.xlabel("Time")
    plt.ylabel(f"{target} (nT)")
    plt.title(f"True vs. predicted {target} (nT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_target_histograms(
    data: DataFrame, label: str, targets: list[str] = TARGETS
) -> None:
    """Plots all 'label' (training/validation) targets' density histograms 
    from the data."""
    
    _: Figure; axes: ndarray
    _, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes: ndarray = axes.flatten()
    
    # A color for each target.
    colors = ["gray", "blue", "orange", "green", "red"]
    
    # Plot histograms for each target.
    for i, (target, color) in enumerate(zip(targets, colors)):
        axes[i].hist(data[target], bins=50, color=color, alpha=0.7, density=True)
        axes[i].set_title(f"{target} {label} density")
        axes[i].grid(True)
    
    # Hide the unused subplot (bottom right).
    axes[5].axis("off")
    # Display the images.
    plt.tight_layout()
    plt.show()


def residual_density_histogram(residuals: list[DataFrame]) -> None:
    """Plots normalized histograms of prediction residuals."""
    rows: int = 3
    cols: int = 2
    _: Figure; axes: ndarray
    _, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), squeeze=False)
    axes: ndarray = axes.flatten()

    for i, res in enumerate(residuals):
        target: str = res.columns[0]
        values: ndarray = res[target].to_numpy()
        axes[i].hist(
            values,
            bins=100,
            edgecolor="black",
            alpha=0.5,
            density=True,
            label=target,
        )
        axes[i].set_title(f"{target} residual density")
        axes[i].legend()

    # Hide the unused subplot (6th slot).
    axes[5].set_visible(False)

    plt.tight_layout()
    plt.show()


def residuals_vs_predicted_scatter(
    residuals: list[DataFrame], predictions: dict
) -> None:
    """Scatter plot of residuals vs. predicted values."""
    targets: list[str] = [k for k in predictions.keys()]
    n_cols: int = len(targets)
    _: Figure; axes: ndarray
    _, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    
    for i, target in enumerate(targets):
        # Target residuals and predictions.
        res: ndarray = residuals[i].to_numpy().flatten()
        pred: ndarray = predictions[target][target].to_numpy().flatten()
        
        axes[i].scatter(pred, res, alpha=0.5, s=10)
        axes[i].axhline(y=0, color='r', linestyle="--")
        axes[i].set_title(f"{target} residuals vs. predicted")
        axes[i].set_xlabel("Predicted values")
        
        if i == 0:
            axes[i].set_ylabel("Residuals")
    
    plt.tight_layout()
    plt.show()


def global_true_values_scatter_plot(y_test: DataFrame, target: str = "SMR") -> None:
    """Plots SMR true values as a scatter plot."""
    _: Figure; axes: ndarray
    _, axes = plt.subplots(figsize=(8, 6))
    axes.scatter(y_test["index"], y_test[target], marker='.', s=5, alpha=0.5)
    # Set labels and title.
    axes.set_xlabel("Time")
    axes.set_ylabel(f"True {target}")
    axes.set_title(f"{target} true values")
    # Adjust layout and display,
    plt.tight_layout()
    plt.show()


def true_values_scatter_plot(
    y_test: DataFrame, targets: list[str] = TARGETS
) -> None:
    """Plots SMR MLT indices' true values as a scatter plot."""
    _: Figure; axs: ndarray
    _, axs = plt.subplots(2, 2, figsize=(10, 10))
    axes: ndarray = axs.flatten()
    # Loop over the targets.
    for target, ax in zip(targets[1:], axes):
        ax.scatter(y_test["index"], y_test[target], marker='.', s=5, alpha=0.5)
        # Set axes' labels.
        ax.set_xlabel("Time")
        ax.set_ylabel(f"True {target}")
        # set title.
        ax.set_title(f"{target} true values")
    # Adjust layout and display,
    plt.tight_layout()
    plt.show()


def q_q_plot(residuals: DataFrame) -> None:
    """Q-Q plot for checking normality of the residuals in a single figure.
    Note that XGBoost does not require normality."""

    n_cols: int = len(residuals.columns)
    _: Figure; axes: ndarray
    _, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
    
    for idx, c in enumerate(residuals.columns):
        # Create Q-Q plot on specific subplot
        probplot(residuals[c], dist="norm", plot=axes[idx])
        axes[idx].set_title(f"{c} residual Q-Q plot")
    
    plt.tight_layout()
    plt.show()


def plot_top_cdaweb_features() -> None:
    """Plots a bar chart of the top four CDAWeb features from SMR models."""
    models: list[str] = ['SMR-00', 'SMR-06', 'SMR-12', 'SMR-18', 'SMR']
    
    features: dict[str, list[float]] = {
        'Magnitude': [0.0182, 0.0107, 0.0129, 0.0130, 0.0169],
        'Pos_x': [0.0159, 0.0098, 0.0053, 0.0038, 0.0050],
        'BGSM_z': [0.0067, 0.0048, 0.0060, 0.0054, 0.0068],
        'Pos_y': [0.0073, 0.0065, 0.0079, 0.0045, 0.0055]
    }

    x: ndarray = arange(len(models))  # the label locations
    width: float = 0.2  # the width of the bars
    multiplier: float = 0.0

    _: Figure; ax: Axes
    _, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    feature_labels: dict[str, str] = {
        'Magnitude': 'Magnetic field magnitude',
        'Pos_x': 'Spacecraft position x coordinate',
        'BGSM_z': 'Magnetic field z component',
        'Pos_y': 'Spacecraft position y coordinate'
    }

    for attribute, measurement in features.items():
        offset: float = width * multiplier
        ax.bar(x + offset, measurement, width, label=feature_labels[attribute])
        multiplier += 1

    ax.set_ylabel("Importance Score")
    ax.set_xticks(x + 1.5 * width) 
    ax.set_xticklabels(models)
    
    ax.legend(loc="upper right")
    ax.grid(axis='y', linestyle="--", alpha=0.7)

    figstring: str = "smr_importance.png"
    plt.savefig(figstring, dpi=300)
    print(f"Plot saved as {figstring}")
