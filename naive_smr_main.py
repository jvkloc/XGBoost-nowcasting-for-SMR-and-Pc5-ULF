"""The main function for naive nowcasting of SuperMAG SMR index and partial 
SMR indices."""

from time import perf_counter

from polars import col, DataFrame

from constants import TARGETS, FOLDER
from data_utils import load_preprocessed_data
from metrics import print_naive_metrics
from plotting import *
from utils import get_elapsed_time


def main( targets: list[str] = TARGETS, folder: str = FOLDER) -> None:
    """Naive forecaster for SMR indices nowcast."""
    
    # Get script start time.
    script_start: float = perf_counter()

    y: DataFrame = load_preprocessed_data(
        file_path=f"{folder}/test_data/smr_y.parquet"
    ).collect()
    
    # A dictionary for naive predictions y_t = y_{t-1}.
    y_pred: dict[str, DataFrame] = {}
    
    # Iterate all targets.
    for target in targets:

        # Naive forecast values for the target.
        y_pred[target] = (
            y.select(["index", target])
            .with_columns(
                # Move every value one step back -> the first row is null.
                col(target).shift(1).alias(f"{target}_pred")
            )
            .drop_nulls()
            .drop(target)
            .rename({f"{target}_pred": target})
        )

    # Align the target indices with the naive nowcast indices.
    y: DataFrame = y.join(y_pred[target].select("index"), on="index", how="inner")

    # Print prediction metrics.
    print_naive_metrics(y_pred, y)

    # Set the y data to a dictionary for further prints.
    y_test: dict[str, DataFrame] = {t: None for t in targets}
    for target in targets:
        y_test[target] = y.select(["index", target])
    
    # Get residuals.
    residuals: list[DataFrame] = [
        y_test[t].drop("index") - y_pred[t].drop("index") for t in targets
    ]
 
    # Print the elapsed time.
    print(f"The script finished in {get_elapsed_time(script_start)} minutes.")
    
    # Plot residuals' density.
    residual_density_histogram(residuals)

    # Plot residuals vs. predicted values.
    residuals_vs_predicted_scatter(residuals, y_pred)
    
    # Plot SMR MLT predictions vs. true values scatter plot from final fold.
    prediction_scatter_plot(y_pred, y_test, targets[1:])

    # Plot SMR global predictions vs. true values scatter plot from final fold.
    global_prediction_scatter_plot(y_pred, y_test, targets[0])

    # Plot SMR MLT prediction vs. true values time series from final fold.
    prediction_time_series(y_pred, y_test, targets[1:])

    # Plot SMR global prediction vs. true values time series from final fold.
    global_prediction_time_series(y_pred, y_test, targets[0])


if __name__ == "__main__":
    main()
