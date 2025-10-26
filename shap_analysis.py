"""SHAP analysis for all models. https://shap.readthedocs.io/en/latest/#"""

from json import load, dump

from cudf import DataFrame as CudaFrame
from numpy import abs as npabs, argmax, load as npload, ndarray, save as npsave
from polars import DataFrame
from matplotlib.pyplot import show as mpshow
from shap import dependence_plot, summary_plot, TreeExplainer 
from xgboost import Booster

from constants import FOLDER, TARGETS as SMR_TARGETS
from data_utils import load_preprocessed_data
from load_model import load_xgb_models
from ulf_constants import TARGETS as P_TARGETS


def save_shap_values(shap: dict, folder: str, models: list[str]) -> None:
    """Saves SHAP values and expected values of 'models' from 'shap' to 
    'folder' as .npy files."""
    
    for model in models:
        # Save SHAP values as .npy file
        npsave(f"{folder}/{model}_shap_values.npy", shap[model]['values'])
        # Save expected value as JSON
        with open(f"{folder}/{model}_expected_value.json", "w") as f:
            dump({"expected_value": float(shap[model]['expected_value'])}, f)
        print(f"{model} SHAP values saved to {folder}")


def load_shap_values(folder: str, models: list[str]) -> None | dict:
    """Loads "models'" SHAP values and expected values from "folder"."""
    shap: dict[str, ndarray | float] = {}
    for model in models:
        # File paths.
        values_path: str = f"{folder}/{model}_shap_values.npy"
        expected_path: str = f"{folder}/{model}_expected_value.json"
        
        try:
            # Load SHAP values if found from the path.
            values: ndarray = npload(values_path)
        except FileNotFoundError:
            # If not found, return None.
            return None

        # Load expected value.
        with open(expected_path, "r") as f:
            expected_value: float = load(f)['expected_value']
        
        # Set values to the dictionary.
        shap[model] = {"values": values, "expected_value": expected_value}
        
        print(f"{model} SHAP values loaded from {folder}")
    
    # Return all models' values.
    return shap


def SHAP_analysis(models: dict, model_names: list[str], data: DataFrame) -> dict:
    """Computes SHAP values for the 'models' given the 'data'."""
    shap: dict[str, ndarray | float] = {n: None for n in model_names}
    
    for model in model_names:
        print(f"Computing SHAP values for {model}...", end='')
        
        explainer = TreeExplainer(models[model])
        
        # type(data) = numpy array, pandas DataFrame, or cuDF DataFrame.
        values: ndarray = explainer.shap_values(CudaFrame(data.to_numpy()))
        
        # Save the results.
        shap[model] = {
            "values": values, # 2D numpy array
            "expected_value": explainer.expected_value # float
        }
        print(" done.")
    
    # Return the values.
    return shap


def plot_summary(model_names: list[str], shap: dict, data: DataFrame) -> None:
    """Plots summaries of the results of each model."""
    for model in model_names:
        print(f"{model} summary plot:")
        summary_plot(
            shap[model]['values'], 
            data.to_numpy(), 
            feature_names=data.columns,
            show=False
        )
        mpshow(block=True)


def get_top_feature(values: ndarray, features: list[str]) -> str:
    """Returns the top feature (according to SHAP) of the model based on the 
    models 'values'."""
    
    # Mean absolute SHAP value for each feature (column).
    mean_abs: ndarray = npabs(values).mean(axis=0)
    
    # Index of the largest value.
    return features[argmax(mean_abs)]


def plot_dependence(models: list[Booster], shap: dict, data: DataFrame) -> None:
    """Plots dependence of the top feature (according to 'shap') for each 
    model in 'models'."""
    
    # Get features.
    features: list[str] = data.columns
    
    for model in models:
        # Get model's SHAP values.
        values: ndarray = shap[model]['values']
        
        # Plot dependencies.
        dependence_plot(
            get_top_feature(values, features), 
            values, 
            data.to_numpy(),
            feature_names=features,
            interaction_index="auto", # The feature with the strongest interaction.
            show=False
        )
        mpshow(block=True)


def main(targets: list[str] = SMR_TARGETS, folder: str = FOLDER) -> None:
    # Load data.
    data: DataFrame = load_preprocessed_data(
        file_path=f"{folder}/test_data/smr_X_pruned.parquet"
    ).collect(engine="gpu")
    
    # Load models.
    models: dict = load_xgb_models(f"{folder}/saved_models/", targets)
    
    # Models' names.
    model_names: list[str] = [m for m in models]
    
    # Models' features.
    expected_features: list[str] = list(models.values())[0].feature_names
    
    # Drop index and select features in the order dictated by the models.
    test_data: DataFrame = data.drop("index").select(expected_features)
    
    if not (shap := load_shap_values(f"{folder}/shap_values", model_names)):
        shap: dict[str, ndarray | float] = SHAP_analysis(
            models, model_names, test_data
        )
        save_shap_values(shap, f"{folder}/shap_values", model_names)
    
    # Plot the magnitude and direction of the SHAP values.
    plot_summary(model_names, shap, test_data)

    # Plot dependencies of top features.
    plot_dependence(model_names, shap, test_data)


if __name__=="__main__":
    main()
