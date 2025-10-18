"""Utility functions for the main script."""

from argparse import ArgumentParser
from datetime import datetime, timedelta
from gc import collect
from os import environ
from os.path import abspath, dirname, join
from time import perf_counter

from polars import col, count, DataFrame, Float32, Int64, LazyFrame, Series
from xgboost import Booster

from constants import (
    TARGETS, 
    DESCRIPTION, 
    CDAWEB_PATH, 
    PATH, 
    FEATURES,
)
from data_utils import (
    add_lagged_values, 
    add_time_features, 
    get_supermag_data,
    shift_targets,
)

from metrics import compute_average_metrics, print_metrics
from training import training_loop


def get_data(
    CDAWeb: LazyFrame,
    file: str,
    script_start: float,
    targets: list[str] = TARGETS,
    smag_path: str = PATH,
) -> tuple[LazyFrame, DataFrame, DataFrame]:
    """Loads SuperMAG data, adds ULF and lagged features, and aligns it with 
    CDAWeb data. Returns X and y for training models."""

    # Load SuperMAG data
    SuperMAG_data: LazyFrame = get_supermag_data(path=smag_path, file=file)
    t: float = get_elapsed_time(script_start)
    print(f"SuperMAG data loaded, {t} minutes from script start.")
    
    # Align CDAWeb and SuperMAG indices and merge the data.
    aligned: LazyFrame = align_cdaweb_and_supermag(CDAWeb, SuperMAG_data)
    t: float = get_elapsed_time(script_start)
    print(f"CDAWeb and SuperMAG data combined, {t} minutes from script start.")
    
    # Shift targets for nowcasting three hours into the future.
    shift: int = 180
    shifted: LazyFrame = shift_targets(aligned, targets, shift)
    t: float = get_elapsed_time(script_start)
    print(f"Targets shifted {shift} minutes, {t} minutes from script start.")
    
    # Add lagged values to the data.
    lagged: LazyFrame = add_lagged_values(shifted)
    t: float = get_elapsed_time(script_start)
    print(f"Lagged targets added, {t} minutes from script start.")
    
    # Add time features to the data.
    training_data: LazyFrame = add_time_features(lagged)
    t: float = get_elapsed_time(script_start)
    print(f"Time features added, {t} minutes from script start.")
    
    # Free memory.
    del SuperMAG_data, aligned, shifted, lagged
    collect()

    # Split data into X and y.
    X: DataFrame; y: DataFrame
    X, y = get_X_and_y_split(training_data, targets)
    t: float = get_elapsed_time(script_start)
    print(f"Data splitted into X and y, {t} minutes from script start.")
    
    # Return X and y.
    return X, y


def get_X_and_y_split(data: LazyFrame, targets: list[str]) -> tuple[DataFrame, DataFrame]:
    """Returns the data as features (X) and targets (y)."""
    # Get column names.
    names: list[str] = data.collect_schema().names()
    # Get features.
    features: list[str] = [n for n in names if n not in targets]
    # Split into targets and features.
    target_frame: LazyFrame = data.select(["index"] + targets)
    feature_frame: LazyFrame = data.select(features)    
    
    # Cast into float32, execute the computations and sort by index.
    X: DataFrame = (
        cast_lazyframe_to_float32(feature_frame)
        .collect()
        .sort("index")
        .set_sorted("index")
    )
    y: DataFrame = (
        cast_lazyframe_to_float32(target_frame)
        .collect()
        .sort("index")
        .set_sorted("index")
    )
    
    return X, y


def cast_lazyframe_to_float32(lf: LazyFrame) -> LazyFrame:
    """Returns a Polars LazyFrame with all columns except 'time' and 'index'
    cast to Float32."""
    
    names: list[str] = lf.collect_schema().names()
    cols: list[str] = [n for n in names if n not in ["time", "index"]]
    # Cast all columns except 'time' and 'index' to Float32.
    return lf.with_columns([col(c).cast(Float32) for c in cols])


def training(
    X: DataFrame,
    y: DataFrame,
    model_path: str,
    xgb_params: dict[str, dict],
    n_estimators: dict[str, int],
    early_stop: dict[str, int],
    targets: list[str],
) -> tuple[Booster, dict, dict, dict, dict, DataFrame, DataFrame]:
    """Training and evaluation logic. Saves a final model."""

    # Get rolling basis cross validation splits.
    train: DataFrame; validation: DataFrame; 
    cv_folds: list[tuple[DataFrame, DataFrame, DataFrame, DataFrame]]
    extreme_values_weeks: list[tuple[datetime, datetime]]
    train, validation, cv_folds, extreme_values_weeks = get_cv_folds(X, y, targets)
    
    # Train models.
    metrics_per_fold: list[dict] = []
    models: dict[str, Booster]; y_pred: dict; y_test: dict; X_test: dict; evals: dict
    models, y_pred, y_test, X_test, evals = training_loop(
        cv_folds,
        extreme_values_weeks,
        metrics_per_fold,
        targets,
        xgb_params,
        n_estimators,
        early_stop
    )

    # Get model metrics.
    metrics: dict = compute_average_metrics(metrics_per_fold, targets=targets)

    # Print the metrics.
    print_metrics(metrics, targets=targets)

    # Save the final models.
    for model in models:
        save_path: str = f"{model_path}{model}_xgb.json"
        models[model].save_model(save_path)
        print(f"\nA model predicting {model} was trained and saved to {save_path}")
    
    return models, y_pred, y_test, X_test, evals, train, validation


def align_cdaweb_and_supermag(cdaweb: LazyFrame, supermag: LazyFrame) -> LazyFrame:
    """Returns CDAWeb and SuperMAG data merged into a single LazyFrame with a
    one minute accuracy on indices."""

    cdaweb: LazyFrame = cdaweb.with_columns(col("index").alias("on")).sort("on")
    supermag: LazyFrame = supermag.with_columns(col("index").alias("on")).sort("on")
    
    # Join CDAWeb and SuperMAG data.
    merged: LazyFrame = (
        supermag
        .join_asof( # Join on 'index' with a 60-second tolerance.
            cdaweb,
            on="on",
            strategy="nearest",  # Other options are 'backward' and 'forward'.
            tolerance="60s",
        )
        .drop(["on", "index_right"]) # Drop temporary join columns.
    )

    return merged.drop_nulls()


def get_cv_folds(
    X: DataFrame,
    y: DataFrame,
    targets: list[str],
    nowcast_shift: int = 180,
) -> tuple[DataFrame, DataFrame, list[tuple], list[str]]:
    """Returns cross-validation folds for all given periods, and the whole train
    (X) and validation (y) data."""

    folds: list[tuple[DataFrame, DataFrame, DataFrame, DataFrame]] = []

    # Get validation period start and end dates.
    period_ranges: list[tuple[datetime, datetime]]
    extreme_values_weeks: list[tuple[datetime, datetime]]
    period_ranges, extreme_values_weeks = get_validation_weeks(y, targets)
    
    # Create timedelta for nowcast gap.
    gap = timedelta(minutes=nowcast_shift)

    print("Cross-validation folds:")
    # Create folds for each period.
    for i, (val_start, val_end) in enumerate(period_ranges, 1):
        # Training must stop before seeing into the future.
        train_cutoff: datetime = val_start - gap
        
        # Training data: everything before (val_start - gap).
        X_train = X.filter(col("index") < train_cutoff)
        y_train = y.filter(col("index") < train_cutoff)
        
        # Validation data.extreme_values_weeks
        X_val = X.filter((col("index") >= val_start) & (col("index") < val_end))
        y_val = y.filter((col("index") >= val_start) & (col("index") < val_end))
        
        folds.append((X_train, y_train, X_val, y_val))
        
        print(
            f"{i}. train until {train_cutoff.strftime("%Y-%m-%d")}, "
            f"{nowcast_shift} min gap, "
            f"validate from {val_start.strftime('%Y-%m-%d')} to {val_end.strftime("%Y-%m-%d")}, "
            f"{y_val.height} validation rows"
        )

    # Last validation period start date.
    last_period_start: datetime = period_ranges[-1][0]
    
    # Full training data (everything before last validation period - gap).
    cutoff: datetime = last_period_start - gap
    training: DataFrame = (
        y.filter(col("index") < cutoff)
        .join(X.filter(col("index") < cutoff), on="index")
        .drop("index")
    )

    # Full validation data (last validation period).
    validation:DataFrame = (
        y.filter((col("index") >= last_period_start) & (col("index") < period_ranges[-1][1]))
        .join(
            X.filter((col("index") >= last_period_start) & (col("index") < period_ranges[-1][1])),
            on="index"
        )
        .drop("index")
    )

    return training, validation, folds, extreme_values_weeks


def get_extreme_weeks(
    y: DataFrame, targets: list[str]
) -> list[tuple[datetime, datetime]]:
    """Returns a list of random unique two-week period start and end datetime 
    tuples from unique years. The periods contain the highest number of extreme 
    values of the targets."""

    dates: dict[str, list[datetime]] = {t: [] for t in targets}
    years: set[int] = set()

    for target in targets:
        # Get the 0.1% and 99.9% quantiles of the target.
        quantiles: DataFrame = y.select([
            col(target).quantile(0.001).alias("low"),
            col(target).quantile(0.999).alias("high")
        ])
        q_low: float; q_high: float
        q_low, q_high = quantiles[0, "low"], quantiles[0, "high"]

        # Filter rows with extreme values.
        extremes: DataFrame = y.filter(
            (col(target) < q_low) | (col(target) > q_high) &
            (col("index").dt.year() != 1999)
        )

        # Get a two-week period ID and count extreme values per period.
        counts: DataFrame = (
            extremes
            .group_by_dynamic(
                "index",
                every="2w",  
                period="2w",
                closed="left"
            ).agg(
                count().alias("extreme_count")
            )
        )
        
        # Add year to get unique years.
        counts: DataFrame = counts.with_columns(col("index").dt.year().alias("year"))

        unique_years: DataFrame = (
            counts
            .group_by("year")
            .agg(
                col("extreme_count").max().alias("max_extreme_count"),
                col("index").filter(
                    col("extreme_count") == col("extreme_count").max()
            )
            .first()
            .alias("index")
            ).sort( # Use year to break possible ties.
                 by=["max_extreme_count", "year"], descending=[True, True]
            )
        )
        
        for target in targets:
            # Check if we already have 3 years for this target.
            if len(dates[target]) == 3:
                continue

            # Iterate through the top years for the current target.
            for row in unique_years.iter_rows(named=True):
                year: int = row['year']
                start_date: datetime = row['index']
                if year not in years:
                    dates[target].append(start_date)
                    years.add(year)
                    break 
    
    # Return a list of start and end date tuples for get_validation_weeks().
    return [
        (d, d + timedelta(days=14))
        for d in [dt for dt_list in dates.values() for dt in dt_list]
    ]


def get_validation_weeks(
    y: DataFrame, targets: list[str]
) -> tuple[list[tuple[datetime, datetime]], list[tuple[datetime, datetime]]]:
    """Returns lists of datetime-tuples containing the start and end dates 
    of two-week validation periods."""

    # Get a dictionary of extreme week start dates.
    extreme_weeks: list[tuple[datetime, datetime]] = get_extreme_weeks(
        y, targets
    )

    # Get the remaining years.
    all_years: set[int] = set(y["index"].dt.year().unique().to_numpy())
    extreme_years: set[int] = {period[0].year for period in extreme_weeks} # int(period[0].split('-')[0])
    remaining_years: list[int] = list(all_years - extreme_years)
    remaining_years.sort()
    
    # Get the remaining weeks.
    remaining_weeks: list[tuple[datetime, datetime]] = []
    for year in remaining_years:
        end_date = datetime(year, 12, 31)
        start_date = end_date - timedelta(days=14)
        remaining_weeks.append((start_date, end_date))
    
    # Return all validation weeks.
    return extreme_weeks + remaining_weeks, extreme_weeks


def get_data_gaps(data: DataFrame, limit: int = 300) -> dict:
    """Returns a dictionary of gaps longer than limit seconds. Gaps are saved
    in minutes."""
    
    data: DataFrame = data.sort("index")
    # Time differences in nanoseconds.
    diffs: Series = data["index"].cast(Int64).diff()
    
    # Boolean mask for gaps longer than the limit (in ns).
    gap_mask: Series = diffs > int(limit * 1e9)
    # Gaps in minutes.
    gaps: Series = diffs.filter(gap_mask).cast(Float32) * (1 / (60 * 1e9)) 
    # Gap start indices.
    indices: Series = data["index"].filter(gap_mask)
    
    return {"limit": limit, "gaps": gaps.to_numpy(), "indices": indices.to_numpy()}


def set_environment_variable(data_dir: str = CDAWEB_PATH) -> None:
    """Sets SPEDAS_DATA_DIR environment variable."""
    environ["SPEDAS_DATA_DIR"] = data_dir
    print(f"environ['SPEDAS_DATA_DIR'] = {data_dir}\n")


def set_environment_var(data_dir: str = "spedas_data") -> None:
    """Sets SPEDAS_DATA_DIR environment variable."""
    download_dir: str = join(dirname(abspath(__file__)), data_dir)
    environ["SPEDAS_DATA_DIR"] = download_dir
    print(f"environ['SPEDAS_DATA_DIR'] = {download_dir}\n")


def get_elapsed_time(start: float) -> float:
    """Returns the elapsed time since 'start' in minutes with two decimals."""
    elapsed_time: float = (perf_counter() - start) / 60
    return f"{elapsed_time:.2f}"


def get_arguments(description: str = DESCRIPTION) -> None:
    """Command line arguments."""
    parser = ArgumentParser(description=description)
    
    parser.add_argument(
        "--data",
        action="store_true",
        help="Preprocess and save training and validation data."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model using preprocessed data from --datapath."
    )
    parser.add_argument(
        "--mission",
        type=str,
        default="both",
        help="Choose 'ace', 'wind' or 'both' mission data. Defaults to 'both'."
    )
    
    return parser.parse_args()
