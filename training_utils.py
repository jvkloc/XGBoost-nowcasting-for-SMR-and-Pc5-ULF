"""Utility functions for training loop funciton."""

from datetime import datetime
from math import sqrt

from numpy import ndarray
from polars import col, DataFrame, Float32, lit, Series, when
from pyarrow import Array
from xgboost import Booster, DMatrix


def get_prediction_dataframe(
    pred: ndarray, y_test: DataFrame, target: str | None = None
) -> DataFrame:
    """Returns Polars dataframe with the given predictions and index column."""
    
    targets: list[str] = [col for col in y_test.columns if col != "index"]
    index_col: ndarray = y_test["index"].to_numpy()
    
    if pred.ndim == 1:  # Single-target prediction.
        return DataFrame(
            {"index": index_col, target: pred},
            schema={"index": y_test["index"].dtype, target: Float32}
        )
    else:  # Multi-target prediction.
        return DataFrame(
            {"index": index_col, **{target: pred[:, i] for i, target in enumerate(targets)}},
            schema={"index": y_test["index"].dtype, **{target: Float32 for target in targets}}
        )


def get_prediction_metrics(
    y_test: DataFrame,
    y_pred: DataFrame,
    model: Booster,
    d_train: DMatrix,
    y_train: DataFrame,
    target: str,
    batch_threshold: int = 10_000_000,
    batch_size: int = 100_000,
) -> dict[str, float]:
    """Returns prediction metrics (MSE, R^2, MAE, RMSE, training RMSE) for a 
    single target."""
    
    # Testing and prediction values.
    t: Series = y_test[target]
    p: Series = y_pred[target]
    
    # Differences of true and predicted validation values.
    diff: Series = t - p
    
    # Validation MAE.
    mae: float = diff.abs().mean()
    # Validation MdAE.
    mdae: float = diff.abs().median()
    # Validation RMSE.
    rmse: float = sqrt(diff.pow(2).mean())
    
    # Naive forecaster error (y_i - y_{i-1}) for scaled errors.
    naive_diff: Series = y_train[target].diff().drop_nulls()
    scale_mae: float = naive_diff.abs().mean()
    scale_rmse: float = sqrt(naive_diff.pow(2).mean())

    # Get the number of training samples.
    n_samples: int = d_train.num_row()
    # Predict with training data.
    if n_samples < batch_threshold: # Data has less than 10 million rows.
        train_pred = Series(model.predict(d_train).flatten())
    else: # Data has 10 million or more rows: use batching.
        train_predictions: list[ndarray] = []
        for i in range(0, n_samples, batch_size):
            end_idx: int = min(i + batch_size, n_samples)
            batch_indices = list(range(i, end_idx))
            batch_dmatrix: DMatrix = d_train.slice(batch_indices)
            batch_pred: ndarray = model.predict(batch_dmatrix)
            train_predictions.extend(batch_pred.flatten())
        # Combine predictions.
        train_pred = Series(train_predictions)
    
    # Get predictions from training data.
    train_t: Series = y_train[target]
    # Differences between true and predicted training data values.
    train_diff: Series = train_t - train_pred
    
    # Get training MAE.
    train_mae: float = train_diff.abs().mean()
    # Get training MdAE.
    train_mdae: float = train_diff.abs().median()
    # Get training RMSE.
    train_rmse: float = sqrt(train_diff.pow(2).mean())

    # Set the metrics to a dictionary.
    metrics: dict[str, float] = {
        # Validation data.
        f"{target}_MAE": mae,
        f"{target}_MdAE": mdae,
        f"{target}_RMSE": rmse,
        f"{target}_MASE": mae /scale_mae,
        f"{target}_MdASE": mdae / scale_mae,
        f"{target}_RMSSE": rmse /scale_rmse,
        # Training data.
        f"{target}_training_MAE": train_mae,
        f"{target}_training_MASE": train_mae / scale_mae,
        f"{target}_training_MdAE": train_mdae,
        f"{target}_training_MdASE": train_mdae / scale_mae,
        f"{target}_training_RMSE": train_rmse,
        f"{target}_training_RMSSE": train_rmse / scale_rmse,
    }

    # Return the metrics.
    return metrics


def get_target_label(y: DataFrame, target: str) -> Array:
    """Returns the 'label' param for XGBoost training."""
    return y.select(target).drop_nulls().to_arrow()


def get_weights(
    full_data: DataFrame, extreme_values_weeks: list[tuple[datetime, datetime]]
) -> Array:
    """Returns 'sample_weights' param for XGBoost model or None if no extreme
    values are found."""

    # Create a column for weights and initialise it with default 1.0.
    weighted: DataFrame = full_data.with_columns(weight=lit(1.0))

    for week_start, week_end in extreme_values_weeks:
        weighted: DataFrame = (
            weighted
            .with_columns(
                weight=when(
                    (col("index") >= week_start) & (col("index") < week_end)
                )
                .then(5.0) # 5x attention for extreme weeks.
                .otherwise(col("weight")) # Default for other times.
                .alias("weight")
            )   
        )

    sample_weights: Array = weighted.get_column("weight").to_arrow()
    del weighted

    return sample_weights
