"""The main function for training an XGBoost model for SuperMAG SMR index 
prediction with rolling basis cross-validation. The script can be run also 
without training, using a saved model for predicting."""

from argparse import Namespace
from time import perf_counter

from numpy import ndarray
from polars import DataFrame, LazyFrame
from xgboost import Booster

from constants import (
    XGB_METRIC,
    XGB_PARAMS,
    N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    TARGETS,
    FOLDER
)
from printing import print_data_gaps
from prune_features import select_features
from data_utils import load_preprocessed_data, process_cdaweb_data, save_processed_data
from load_model import load_xgb_models, predict_with_loaded_models
from metrics import *
from plotting import *
from training_utils import get_prediction_dataframe
from utils import (
    get_arguments,
    set_environment_variable,
    get_data,
    get_data_gaps,
    training,
    get_elapsed_time,
)


def main(
    metric: str = XGB_METRIC,
    xgb_params: dict = XGB_PARAMS,
    n_estimators: dict = N_ESTIMATORS,
    early_stop: dict = EARLY_STOPPING_ROUNDS,
    targets: list[str] = TARGETS,
    folder: str = FOLDER
) -> None:
    """
    Command line arguments: 
        --data      : Loads, processes and saves data.
        --train     : Loads data from disk, trains models and saves them.
        --mission   : 'ace' or 'wind' or 'both' missions, defaults to 'both'. 
        no argument : Predicts with saved models and test data loaded from disk.
    """
    
    # Get script start time.
    script_start: float = perf_counter()
    
    # Parse command-line arguments.
    args: Namespace = get_arguments()

    # Set PySPEDAS environment variable.
    set_environment_variable()
    
    # Dictionary for model statistics. 
    evals: dict | None = None

    # Path for saving and loading models.
    model_path: str = f"{FOLDER}/saved_models/"

    if args.data: # Prepare and save data.
        
        # CDAWeb data datetime range.
        start: str = "1999-01-01 00:00:00"
        end: str = "2023-12-31 23:59:59"

        # Get CDAWeb data.
        #CDAWeb: LazyFrame = process_cdaweb_data(args.mission, start, end)
        CDAWeb: LazyFrame = load_preprocessed_data(f"{folder}/cdaweb_data/2024_cdaweb.parquet")
        
        # Print the elapsed time.
        t: float = get_elapsed_time(script_start)
        print(f"CDAweb files loaded and processed, {t} minutes from script start.")
        
        # SuperMAG data file.
        file: str = "2024_smr.csv"

        # Align CDAWeb and SuperMAG time indices, split data into X and y.
        X: DataFrame; y: DataFrame
        X, y = get_data(CDAWeb, file, script_start)
        
        # Save the data as X and y.
        save_processed_data(X, f"{folder}/test_data/smr_X_full.parquet")
        save_processed_data(y, f"{folder}/test_data/smr_y_full.parquet")
        
        # Get possible remaining data gaps.
        gaps: dict[str, dict] = {"X": get_data_gaps(X), "y": get_data_gaps(y)}

        # Print data gaps.
        print_data_gaps(gaps)

        # Print the elapsed time.
        print(f"The script finished in {get_elapsed_time(script_start)} minutes.")

        return
    
    elif args.train: # Train models with preprocessed data.
        
        # Load the data.
        X: DataFrame = load_preprocessed_data(
            file_path=f"{folder}/train_data/smr_X_pruned.parquet"
        ).collect()
        y: DataFrame = load_preprocessed_data(
            file_path=f"{folder}/train_data/smr_y_pruned.parquet"
        ).collect()
        
        # Train and evaluate models. Save final models.
        models: dict; y_pred: dict; y_test: dict; X_test: dict
        evals: dict; train: DataFrame; validation: DataFrame
        models, y_pred, y_test, X_test, evals, train, validation = training(
            X,
            y,
            model_path,
            xgb_params,
            n_estimators,
            early_stop,
            targets,
        )

    else: # Load saved test data and models and make predictions.
        
        # Load the data.
        X: DataFrame = load_preprocessed_data(
            file_path=f"{folder}/test_data/smr_X_pruned.parquet"
        ).collect()
        y: DataFrame = load_preprocessed_data(
            file_path=f"{folder}/test_data/smr_y.parquet"
        ).collect()
        
        # Load the models.
        models: dict[str,Booster] = load_xgb_models(model_path, targets)
        
        # Predict.
        predictions: dict[str, ndarray] = predict_with_loaded_models(models, X)
        
        # Set the predictions to a dictionary of DataFrames.
        y_pred: dict[str, DataFrame] = {p: None for p in predictions}
        for p in predictions:
            framed: DataFrame = get_prediction_dataframe(predictions[p], y ,p)
            y_pred[p] = framed
        
        y_train: DataFrame = load_preprocessed_data(
            file_path=f"{folder}/train_data/smr_y_pruned.parquet"
        ).collect()
        
        # Print prediction metrics.
        print_prediction_metrics(y, y_pred, y_train)

        # Set the y data to a dictionary for further prints.
        y_test: dict[str, DataFrame] = {t: None for t in targets}
        for target in targets:
            y_test[target] = y.select(["index", target])


    # Print feature importance from the final model.
    print_feature_importance(models)
    
    # Get residuals.
    residuals: list[DataFrame] = [
        y_test[t].drop("index") - y_pred[t].drop("index") for t in targets
    ]
 
    # Print the elapsed time.
    print(f"The script finished in {get_elapsed_time(script_start)} minutes.")
    
    if evals is not None: # Script ran in training mode.
        
        # Plot train and test root mean square errors.
        plot_evals_result(evals, metric, models, targets)
        
        # Plot train and test data SMR density histograms.
        plot_target_histograms(train, "Train")
        plot_target_histograms(validation, "Validation")

    # Plot residuals' density.
    residual_density_histogram(residuals)

    # Plot residuals vs. predicted values.
    residuals_vs_predicted_scatter(residuals, y_pred)

    # Plot test data feature time series.
    #plot_features_time_series(X_test)
    
    # Plot SMR MLT predictions vs. true values scatter plot from final fold.
    prediction_scatter_plot(y_pred, y_test, targets[1:])

    # Plot SMR predictions vs. true values scatter plot from final fold.
    global_prediction_scatter_plot(y_pred, y_test, targets[0])

    # Plot SMR MLT prediction vs. true values time series from final fold.
    prediction_time_series(y_pred, y_test, targets[1:])

    # Plot SMR prediction vs. true values time series from final fold.
    global_prediction_time_series(y_pred, y_test, targets[0])


if __name__ == "__main__":
    main()
