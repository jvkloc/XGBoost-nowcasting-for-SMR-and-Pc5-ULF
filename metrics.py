"""Functions for evaluating the model."""

from numpy import (
    abs as npabs,
    diff as npdiff, 
    mean as npmean,
    median as npmedian,
    ndarray,
    sqrt as npsqrt
)
from polars import col, DataFrame, Expr
from xgboost import Booster

from constants import TARGETS


def compute_average_metrics(
    all_metrics: list[dict], targets: list[str] = TARGETS
) -> dict[str, dict]:
    """Returns the average metrics in a dictionary across all training folds."""

    # Dictionary for computed metrics.
    computed: dict[str, dict] = {target: {} for target in targets}

    for target in targets:
        # Filter metrics for the specific target.
        metrics = [m for m in all_metrics if m['target'] == target]
        
        # Extract metrics for the target across all relevant folds.
        values: DataFrame = DataFrame({
            "MAE": [m.get(f"{target}_MAE") for m in metrics],
            "MdAE": [m.get(f"{target}_MdAE") for m in metrics],
            "RMSE": [m.get(f"{target}_RMSE") for m in metrics],
            "MASE": [m.get(f"{target}_MASE") for m in metrics],
            "MdASE": [m.get(f"{target}_MdASE") for m in metrics],
            "RMSSE": [m.get(f"{target}_RMSSE") for m in metrics],
            "training_MAE": [m.get(f"{target}_training_MAE") for m in metrics],
            "training_MdAE": [m.get(f"{target}_training_MdAE") for m in metrics],
            "training_MASE": [m.get(f"{target}_training_MASE") for m in metrics],
            "training_MdASE": [m.get(f"{target}_training_MdASE") for m in metrics],
            "training_RMSE": [m.get(f"{target}_training_RMSE") for m in metrics],
            "training_RMSSE": [m.get(f"{target}_training_RMSSE") for m in metrics],
        })
        
        # Compute mean and std for each metric.
        exprs: list[Expr] = (
            [col(c).mean().alias(f"{c}_mean") for c in values.columns] + 
            [col(c).std().alias(f"{c}_std") for c in values.columns]
        )

        # A dictionary for the metrics.
        agg_stats: dict = values.select(exprs).to_dicts()[0]

        # Add the metrics to the result dictionary.
        keys: list[str] = ["MAE", "MdAE", "RMSE", "MASE", "MdASE", "RMSSE"]
        for key in keys:
            computed[target][key] = agg_stats[f"{key}_mean"]
            computed[target][f"{key}_std"] = agg_stats[f"{key}_std"]

        # Add training metrics.
        training_keys: list[str] = [
            "training_MAE", "training_MdAE", "training_MASE", 
            "training_MdASE", "training_RMSE", "training_RMSSE"
        ]
        for key in training_keys:
            computed[target][key] = agg_stats[f"{key}_mean"]
            computed[target][f"{key}_std"] = agg_stats[f"{key}_std"]

    return computed


def print_metrics(metrics: dict[str, dict], targets: list[str] = TARGETS) -> None:
    """Prints the average model metrics from the dictionary argument."""
    for target in targets:
        print(f"\nAverage metrics for {target}:")
        print(f"MAE: {metrics[target]['MAE']:.2f} (±{metrics[target]['MAE_std']:.2f})")
        print(f"MdAE: {metrics[target]['MdAE']:.2f} (±{metrics[target]['MdAE_std']:.2f})")
        print(f"RMSE: {metrics[target]['RMSE']:.2f} (±{metrics[target]['RMSE_std']:.2f})")
        print(f"MASE: {metrics[target]['MASE']:.2f} (±{metrics[target]['MASE_std']:.2f})")
        print(f"MdASE: {metrics[target]['MdASE']:.2f} (±{metrics[target]['MdASE_std']:.2f})")
        print(f"RMSSE: {metrics[target]['RMSSE']:.2f} (±{metrics[target]['RMSSE_std']:.2f})")
        print(f"Training MAE: {metrics[target]['training_MAE']:.2f} (±{metrics[target]['training_MAE_std']:.2f})")
        print(f"Training MdAE: {metrics[target]['training_MdAE']:.2f} (±{metrics[target]['training_MdAE_std']:.2f})")
        print(f"Training RMSE: {metrics[target]['training_RMSE']:.2f} (±{metrics[target]['training_RMSE_std']:.2f})")
        print(f"Training MASE: {metrics[target]['training_MASE']:.2f} (±{metrics[target]['training_MASE_std']:.2f})")
        print(f"Training MdASE: {metrics[target]['training_MdASE']:.2f} (±{metrics[target]['training_MdASE_std']:.2f})")
        print(f"Training RMSSE: {metrics[target]['training_RMSSE']:.2f} (±{metrics[target]['training_RMSSE_std']:.2f})")


def print_prediction_metrics(
    y_test: DataFrame,
    y_pred: dict[str, DataFrame],
    y_train: DataFrame,
    targets: list[str] = TARGETS
) -> None:
    """Prints prediction metrics."""
    for target in targets:
        if target in y_test.columns and target in y_pred:
            print(f"\nMetrics for {target}:")
            
            # Get target columns as NumPy arrays for metrics.
            true_val: ndarray = y_test.select(target).to_numpy().flatten()
            prediction: ndarray = y_pred[target].select(target).to_numpy().flatten()
            
            # Differences of true and predicted validation values.
            diff: ndarray = true_val - prediction
            
            # Validation MAE.
            mae: float = npmean(npabs(diff))
            mdae: float = npmedian(npabs(diff))
            rmse: float = npsqrt(npmean(diff**2))
            
            # Naive forecaster error (y_i - y_{i-1}) for scaled errors.
            naive_diff: ndarray = npdiff(y_train.select(target).to_numpy().flatten())
            scale_mae: float = npmean(npabs(naive_diff))
            scale_rmse: float = npsqrt(npmean(naive_diff ** 2))

            # Compute and print metrics.
            print(f"MAE: {mae:.2f}")
            print(f"MdAE: {mdae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MASE: {mae / scale_mae:.2f}")
            print(f"MdASE: {mdae / scale_mae:.2f}")
            print(f"RMSSE: {rmse / scale_rmse:.2f}")
        else:
            print(f"Target {target} not found in y_test or y_pred")


def print_feature_importance(models: dict[str, Booster]) -> None:
    """Prints feature importances of the models in descending order."""
    for target in models:
        model: Booster = models[target]
        try:
            importances: dict = model.get_score(importance_type="gain")
        except Exception as e:
            print(f"{target} model.get_score(): {e}")
        if not importances:
            print(f"\nNo {target} feature importances available.")
            return
        total_splits: int = sum(importances.values())
        
        # Set the importances to a list.
        sorted: list[tuple[str, float]] = [
            (feature, importances.get(feature, 0) / total_splits)
            for feature in model.feature_names
        ]

        # Sort the list by importance, the most important first.
        sorted.sort(key=lambda x: x[1], reverse=True)
        
        # Print the sorted importances.
        print(f"\n{target} Feature importances:")
        for feature, importance in sorted:
            print(f"{feature}: {importance:.4f}")
