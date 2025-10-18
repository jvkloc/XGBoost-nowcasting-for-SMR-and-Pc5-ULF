"""Functions for using a saved model."""

from numpy import ndarray
from polars import col, DataFrame, lit
from xgboost import Booster, DMatrix


def load_xgb_models(
    path: str, targets: list[str]
) -> dict[str, Booster]:
    """Returns a XGBoost models and their feature names from the path."""
    models: dict[str, Booster] = {t: None for t in targets}
    for t in targets:
        model = Booster()
        model.load_model(f"{path}{t}_xgb.json")
        models[t] = model
    return models


def get_dmatrix(X_test: DataFrame, feature_names: list[str]) -> DMatrix:
    """Returns a DMatrix for predicting with XGBoost models."""
    # Drop index from the data.
    Xtest: DataFrame = X_test.drop("index")
    print(Xtest.columns)
    # Order feature names to match the model's order.
    ordered_names: list[str] = []
    for feature in feature_names:
        if feature in Xtest.columns: # Append the name to the list.
            ordered_names.append(col(feature))
        else:
            # Stop the script: predicting wouldn't work.
            raise ValueError(f"'{feature}' not found from test data.")
    
    # Create DataFrame with the same order as the model.
    X: DataFrame = Xtest.select(ordered_names)
    
    # Return a DMatrix.
    return DMatrix(data=X.to_numpy(), feature_names=X.columns)


def predict_with_loaded_models(
    models: dict[str,Booster], X_test: DataFrame
) -> dict[str, ndarray]:
    """Returns predictions by given models based on the X_test data."""
    # Get the model features in correct order.
    expected_features = list(models.values())[0].feature_names
    
    # Drop index from the data.
    Xtest: DataFrame = X_test.drop("index").select(expected_features)
    
    # DMatrix for XGBoost.
    dtest = DMatrix(data=Xtest.to_numpy(), feature_names=Xtest.columns)
    
    # Dictionary for predictions.
    y_preds: dict[str, ndarray] = {m: None for m in models}
    
    for m in models: # Predict with each model.
        y_preds[m] = models[m].predict(dtest)
    
    # Return predictions.
    return y_preds


def predict_with_loaded_models2(
    models: dict[str, tuple[Booster, list[str]]], 
    dtest: DMatrix
) -> dict[str, ndarray]:
    """Returns predictions by given models based on the aligned DMatrix."""
    
    # Dictionary for predictions.
    y_preds: dict[str, ndarray] = {}
    
    for target, (model, _) in models.items():  # Predict with each model.
        y_preds[target] = model.predict(dtest)
    
    return y_preds