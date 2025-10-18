"""Script for Optuna hyperparameter search."""

from json import dump

from numpy import ndarray, sqrt
import optuna
from polars import DataFrame
from sklearn.metrics import mean_squared_error
from xgboost import Booster, DMatrix, train

from constants import FOLDER
from data_utils import load_preprocessed_data
from ulf_constants import TARGETS as P_TARGETS
from utils import get_cv_folds


def objective(trial, cv_folds, target: str):
    """Optuna objective function."""
    params: dict = {
        "objective": "reg:squarederror",
        "tree_method": "auto",
        "device": "cuda",
        "sampling_method": "gradient_based",
        "max_depth": trial.suggest_int("max_depth", 8, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 4, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, log=True),
        "lambda": trial.suggest_float("lambda", 2.0, 5.0, log=True),
        "alpha": trial.suggest_float("alpha", 1.5, 3.0, log=True),
        "gamma": trial.suggest_float("gamma", 15.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.65, 0.75),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.9, 1.0),
    }
    n_estimators: int = trial.suggest_int("n_estimators", 1500, 1700, step=100)
    early_stopping_rounds: int = trial.suggest_int("early_stopping_rounds", 140, 160)

    # List for each fold's RMSE.
    fold_scores: list[float] = []
    
    # Train model on each fold.
    for X_train_fold, y_train_fold, X_valid_fold, y_valid_fold in cv_folds:
        # Get target training and validation data.
        y_train_target: DataFrame = y_train_fold[target]
        y_valid_target: DataFrame = y_valid_fold[target]
        
        # Set the target data to DMatrices.
        dtrain = DMatrix(X_train_fold, label=y_train_target)
        dvalid = DMatrix(X_valid_fold, label=y_valid_target)
        
        # A list of datasets to evaluate at each boosting round.
        evals: list[tuple[DMatrix, str]] = [(dtrain, "train"), (dvalid, "valid")]
        
        # Train a model.
        model: Booster = train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        
        # Get validation predictions and compute RMSE.
        preds: ndarray = model.predict(dvalid)
        rmse: float = sqrt(mean_squared_error(y_valid_target, preds))
        
        # Append the RMSE to the result list.
        fold_scores.append(rmse)

    # Return the trial's average RMSE across all folds.
    return sum(fold_scores) / len(fold_scores)


def main() -> None:
    # Load the data.
    X: DataFrame = load_preprocessed_data(
        file_path=f"{FOLDER}/train_data/p_X_pruned.parquet"
    ).collect()
    y: DataFrame = load_preprocessed_data(
        file_path=f"{FOLDER}/train_data/p_y_pruned.parquet"
    ).collect()

    # Get cross-validation folds.
    _, _, cv_folds, _ = get_cv_folds(X, y, P_TARGETS)

    # Run hyperparameter search.
    best_params: dict = {}
    for target in ["Pc5_Day"]:# "", "Pc5_Dusk"]:
        print(f"Starting Optuna search for target: {target}")
        
        # Create a new study for each target.
        study = optuna.create_study(direction="minimize")
        
        # Pass the target to the objective function.
        study.optimize(lambda trial: objective(trial, cv_folds, target), n_trials=25)
        
        # Store the best results for the current target.
        best_params[target] = {
            "best_rmse": float(study.best_value),
            "best_params": {
                k: (float(v) if isinstance(v, (int, float)) else v) 
                for k, v in study.best_params.items()
            }
        }

    # Save the best parameters to a JSON file.
    filepath: str = f"{FOLDER}/optuna_search/best_params.json"
    with open(f"{filepath}", "w") as f:
        dump(best_params, f, indent=4)

    print(f"Best parameters for all targets saved to {filepath}")


if __name__=="__main__":
    main()
