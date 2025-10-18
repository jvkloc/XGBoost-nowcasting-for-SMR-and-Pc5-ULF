"""Training loop and training loop utility functions."""

from datetime import datetime

from numpy import ndarray
from polars import DataFrame
from pyarrow import Array
from xgboost import Booster, DMatrix, train

from training_utils import (
    get_target_label,
    get_prediction_dataframe,
    get_prediction_metrics,
    get_weights,
)


def train_xgboost(
    X_train: DataFrame,
    X_test: DataFrame,
    train_label: Array,
    test_label: Array,
    params: dict,
    estimators: int,
    early_stop: int,
    sample_weights: Array
) -> tuple[Booster, DMatrix, DMatrix, dict]:
    """Initializes and trains an XGBoost model. Returns the model, train and 
    test data and optionally train and test root mean square error values in 
    evals_result."""

    # Drop index columns.
    Xtrain: DataFrame = X_train.drop("index")
    Xtest: DataFrame = X_test.drop("index")

    # Get feature names.
    feature_names: list[str] = Xtrain.columns

    # Set XGBoost parameters.
    dtrain = DMatrix(
        data=Xtrain.to_arrow(), 
        label=train_label, 
        feature_names=feature_names,
        weight=sample_weights,
    )
    dtest = DMatrix(
        data=Xtest.to_arrow(),
        label=test_label,
        feature_names=feature_names,
    )
    evals: list[tuple[DMatrix, str]] = [(dtrain, "train"), (dtest, "test")]
    evals_result: dict = {}
    
    # Train a model.
    model: Booster = train(
        params=params,
        dtrain=dtrain,
        num_boost_round=estimators,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )
    
    # Return the trained model and test data. 
    return model, dtrain, dtest, evals_result


def training_loop(
    cv_folds: list[tuple[DataFrame, DataFrame, DataFrame, DataFrame]],
    extreme_values_weeks: list[tuple[datetime, datetime]],
    metrics_per_fold: list[dict],
    targets: list[str],
    parameters: dict[str, dict],
    estimators: dict[str, int],
    early_stop: dict[str, int]
) -> tuple[dict, dict, dict, dict, dict]:
    """Trains an XGBoost model per target for each rolling basis cross-
    validation folds. Returns the models, predictions, true values, X 
    validation data, and evaluations from the last fold. In addition, the loop 
    saves model metrics from each cross-validation split into the 
    'metrics_per_fold' list for average metric computations."""
    
    # Number of cross-validation splits.
    total_folds: int = len(cv_folds)
    
    # Dictionaries for saving the last rounds' results.
    y_preds: dict = {target: None for target in targets}
    y_tests: dict = {target: None for target in targets}
    X_tests: dict = {target: None for target in targets}
    evaluations: dict = {target: None for target in targets}
    models: dict = {target: None for target in targets}
    
    for fold, (X_train, y_train, X_test, y_test) in enumerate(cv_folds, start=1):
        print(f"training & validating cross-validation fold {fold}/{total_folds}")
        sample_weights: Array = get_weights(X_train, extreme_values_weeks)
        
        for target in targets: # Train a model for each target.
            model: Booster; dtrain: DMatrix; dtest: DMatrix; evals: dict
            model, dtrain, dtest, evals = train_xgboost(
                X_train,
                X_test,
                get_target_label(y_train, target),
                get_target_label(y_test, target),
                parameters[target],
                estimators[target],
                early_stop[target],
                sample_weights
            )
            
            # Predict and evaluate.
            iter_range: tuple = (0, model.best_iteration + 1)
            prediction: ndarray = model.predict(
                dtest, iteration_range=iter_range
            )

            # Set the predictions to a DataFrame.
            y_pred: DataFrame = get_prediction_dataframe(
                prediction, y_test.select(["index", target]), target=target
            )
            
            # Store metrics.
            fold_metrics: dict = get_prediction_metrics(
                y_test.select(target),
                y_pred,
                model,
                dtrain,
                y_train,
                target=target
            )
            metrics_per_fold.append(
                {**fold_metrics, "target": target, "fold": fold}
            )
            
            # Free memory.
            del dtrain, dtest, prediction, fold_metrics
        
            if fold != total_folds:
                del model, y_pred, evals, #y_pred_inverse
            else: # Last cross-validation split.
                y_preds[target] = y_pred
                y_tests[target] = y_test.select(["index", target]) 
                X_tests[target] = X_test
                evaluations[target] = evals
                models[target] = model
    
    # Return rolling basis cross-validation results from the last iteration.
    return models, y_preds, y_tests, X_tests, evaluations
