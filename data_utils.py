"""Functions for loading and processing data."""

from gc import collect
from glob import glob
from os import path

from typing import NamedTuple

from numpy import any as npany, isinf, isnan, nan_to_num, ndarray, pi
from polars import (
    all_horizontal,
    coalesce,
    col,
    DataFrame,
    Datetime,
    Duration,
    Expr,
    Float32,
    Int64,
    LazyFrame,
    lit,
    scan_csv,
    scan_parquet,
    Series,
    when,
) 
from pyspedas import ace, cdf_to_tplot, wind
from pytplot import get_data, del_data, tplot_names
from xgboost import XGBRegressor

from constants import (
    TARGETS,
    CDAWEB_PARAMS,
    CDAWEB_PATH,
    CDAWEB_LIMITS,
    FILL,
    RE,
    K,
    MP,
)


def process_cdaweb_data(mission: str, start: str, end: str) -> LazyFrame:
    """Returns CDAWeb data processed for merging with SuperMAG data in 
    a LazyFrame."""
    
    # PySPEDAS replaces fill values with NumPy NaN.
    # load_cdaweb_data(start, end) # Download from CDAWeb.
    load_cdaweb_from_disk(start, end) # Load from disk.
    
    # Set CDAWeb data to a LazyFrame.
    CDAWeb_data: LazyFrame = get_cdaweb_data(start=start, end=end)
    
    if mission != "both":
        CDAWeb_mission: LazyFrame = get_cdaweb_mission_data(CDAWeb_data, mission)
    else:
        CDAWeb_mission: LazyFrame = combine_ace_and_wind(CDAWeb_data)
    del CDAWeb_data

    # Replace unphysical values.
    CDAWeb_physical: LazyFrame = replace_cdaweb_unphysical_values(
        CDAWeb_mission, mission
    )
    del CDAWeb_mission
    
    # Interpolate data gaps up to five minutes.
    CDAWeb_interpolated: LazyFrame = interpolate_cdaweb_data(
        CDAWeb_physical, max_gap=5
    )
    del CDAWeb_physical
    
    # Add L1 propagation delay and shift the data.
    CDAWeb_processed: LazyFrame = resample_cdaweb_data(CDAWeb_interpolated, mission)
    del CDAWeb_interpolated

    # Return the processed CDAWeb data.
    return CDAWeb_processed


def add_lagged_values(data: LazyFrame, targets: list[str] = TARGETS) -> LazyFrame:
    """Adds lagged values to the data and removes rows with NaN values. 
    Returns the same type of frame than the incoming argument is."""
    
    drivers: list[str] = [
        "Magnitude", "T", "BGSM_z", "V_GSM_x",
        "SC_pos_GSM_Re_x", "SC_pos_GSM_Re_y", "SC_pos_GSM_Re_z",
    ]
    columns: list[str] = targets + drivers
    sorted: LazyFrame = data.sort("index").set_sorted("index")
    exprs: list[Expr] = []
    for column in columns:
        exprs.extend([
            col(column).shift(1).alias(f"{column}_lag1"),
            col(column).shift(2).alias(f"{column}_lag2"),
            col(column).shift(3).alias(f"{column}_lag3"),
            col(column).shift(4).alias(f"{column}_lag4"),
        ])
    data: LazyFrame = sorted.with_columns(exprs)
    return data.drop_nulls()


def add_propagation_delay(
    cdaweb_data: LazyFrame, mission: str, re: int = RE
) -> LazyFrame:
    """Returns a new Polars LazyFrame with all the time indices shifted by 
    the solar wind propagation delay and a new column for time delay."""
    
    if mission == "ace":
        position: str = f"{mission}_swe_"
        velocity: str = f"{mission}_swe_"
    elif mission == "wind":
        position: str = f"{mission}_mfi_"
        velocity: str = f"{mission}_swe_"
    else:
        position: str = ''
        velocity: str = ''
    
    # Add a delay column.
    cdaweb_resampled: LazyFrame = (
        cdaweb_data
        .with_columns([
            (( # Negative positions are squared so no need for minus signs.
                (col(f"{position}SC_pos_GSM_Re_x")**2 +
                col(f"{position}SC_pos_GSM_Re_y")**2 +
                col(f"{position}SC_pos_GSM_Re_z")**2) * re
            ) / (
                abs(# Absolute value so no minus signs here, either.
                    col(f"{position}SC_pos_GSM_Re_x") * col(f"{velocity}V_GSM_x") +
                    col(f"{position}SC_pos_GSM_Re_y") * col(f"{velocity}V_GSM_y") +
                    col(f"{position}SC_pos_GSM_Re_z") * col(f"{velocity}V_GSM_z")
                )
            )).alias("L1_delay")
        ]).fill_null(3600.0).fill_nan(3600.0)
    )

    del cdaweb_data
    collect()

    # Shift time by the delay (in ms) and return the data.
    return (
        cdaweb_resampled
        .with_columns([
            (col("time") + (col("L1_delay") * 1000).cast(Duration("ms")))
            .alias("time")
        ])
    )


def load_cdaweb_data(start: str, end: str, params: dict = CDAWEB_PARAMS) -> None:
    """Checks if existing files are up to date and downloads new files from 
    CDAWeb if necessary. Replaces fill values with NumPy NaN under the hood."""
    
    # Set time range.
    trange: list[str] = [start, end]
    
    # Check/download the files.
    _: list = ace.mfi(trange=trange, varnames=params["ace_mfi"], datatype="h0")
    _: list = ace.swe(trange=trange, varnames=params["ace_swe"], datatype="h0")
    _: list = wind.mfi(trange=trange, varnames=params["wind_mfi"], datatype="h0")
    _: list = wind.swe(trange=trange, varnames=params["wind_swe"], datatype="k0")


def load_cdaweb_from_disk(
    start: str,
    end: str,
    params: dict = CDAWEB_PARAMS,
    base_path: str = CDAWEB_PATH
) -> None:
    """Loads local .cdf files from disk into tplot variables."""
    # Clear all tplot variables from memory.
    del_data('*')
    
    # Dataset paths and their corresponding PySPEDAS variable names.
    dataset_paths: dict[str, str] = {
        "ace_mfi": path.join(base_path, "ace_data/mag/level_2_cdaweb/mfi_h0"),
        "ace_swe": path.join(base_path, "ace_data/swepam/level_2_cdaweb/swe_h0"),
        "wind_mfi": path.join(base_path, "wind_data/mfi/mfi_h0"),
        "wind_swe": path.join(base_path, "wind_data/swe/swe_k0")
    }
    
    # Convert start and end times to years for filtering folders.
    start_year: int = int(start.split('-')[0])
    end_year: int = int(end.split('-')[0])

    # Load .cdf files for each dataset.
    for dataset, filepath in dataset_paths.items():
        # Get variable names from params.
        varnames: list[str] = params.get(dataset, []) if params is not None else []
        if not varnames:
            print(f"No parameters found for dataset {dataset}")
            continue
            
        # Iterate through years in the time range.
        for year in range(start_year, end_year + 1):
            year_path: str = path.join(filepath, str(year))
            if not path.exists(year_path):
                print(f"Directory not found: {year_path}")
                continue
                
            # Find all .cdf files in the year directory.
            cdf_files: list[str] = glob(path.join(year_path, "*.cdf"))
            if not cdf_files:
                print(f"No .cdf files found from {year_path}")
                continue
                
            # Load each .cdf file into tplot variables.
            try:
                # cdf_to_plot() replaces fill values with NumPy NaN.
                cdf_to_tplot(cdf_files, varnames=varnames, merge=True)
                print(f"Loaded {len(cdf_files)} .cdf files from {year_path}")
            except Exception as e:
                print(f"Error loading .cdf files from {year_path}: {e}")
                continue
    
    # Verify loaded tplot variables.
    loaded_vars: list[str] = tplot_names()
    if not loaded_vars:
        raise ValueError("No tplot variables were loaded from .cdf files")


def combine_ace_and_wind(CDAWeb_data: LazyFrame) -> LazyFrame:
    """Returns a LazyFrame where the same ACE and WIND data have been combined 
    to single columns and the whole data sorted by time."""

    def get_measurement_name(col: str) -> str:
        """Returns column name without mission and instrument."""
        return col.split('_', 2)[-1]  # e.g., 'SC_pos_GSM_Re_x'

    cols: list[str] = CDAWeb_data.collect_schema().names()
    measurements: set = {
        get_measurement_name(col) for col in cols if col != "time"
    }
    exprs: list[Expr] = []

    # Combine columns with the same measurement.
    for m in measurements:
        # Get all columns with the measurement m.
        columns: list[str] = [c for c in cols if get_measurement_name(c) == m]
        if not columns:
            continue
        # Create expression from the column(s).
        if len(columns) == 1:
            combined: Expr = col(columns[0])
        else:
            combined: Expr = coalesce(columns)
        # Add the expression to the list.
        exprs.append(combined.alias(m))
    
    # Return sorted data with coalesced columns.
    return (CDAWeb_data
            .with_columns(exprs)
            .select([col("time"), *measurements])
            .filter(col("time").is_not_null())
            .sort("time")
            .set_sorted("time"))


def get_cdaweb_data(
    start: str,
    end: str,
    parameters: dict = CDAWEB_PARAMS,
    re: int = RE,
    k: float = K,
    mp: float = MP
) -> LazyFrame:
    """Processes CDAWeb data into a Polars LazyFrame sorted by time.
    Arguments
    ---------
        start: the first desired time index 
        end: the last desired time index
        parameters: CDAWeb datasets (keys) and parameters (values)
        fill: CDAWeb data fill value -1e31
        re: Earth radius (km)
        k: Boltzmann constant (J/k)
        mp: proton mass (kg)
    """
    frames: list[LazyFrame] = []

    for dataset, params in parameters.items():
        for param in params:
            # Get the data from tplot variables (in memory).
            result: NamedTuple = get_data(param, dt=True)
            if result is None:
                print(f"Variable {param} not found (or invalid) in {dataset}")
                continue
            # Extract time and values.
            times: ndarray = result.times
            y: ndarray = result.y
            
            # Convert time to Polars Datetime.
            time_series = Series(times).cast(Datetime).dt.round("1m")

            # Create a Polars LazyFrame from the times and values.
            if len(y.shape) > 1:
                lf = LazyFrame({
                    "time": time_series,
                    **{f"dim_{i}": y[:, i] for i in range(y.shape[1])},
                })  
            else:
                lf = LazyFrame({
                    "time": time_series,
                    "value": y
                })

            # Deduplicate timestamps
            if len(y.shape) > 1:
                lf: LazyFrame = lf.group_by("time").agg([
                    col(f"dim_{i}").first().alias(f"dim_{i}") 
                    for i in range(y.shape[1])
                ])
            else:
                lf: LazyFrame = lf.group_by("time").agg(col("value").first())

            # Process the data.
            if "SC_pos_GSM" in param:
                for i, axis in enumerate(['x', 'y', 'z']):
                    if len(y.shape) <= 1 or y[:, i].size == 0:
                        continue
                    lf_out: LazyFrame = lf.select(
                        col("time"),
                        (col(f"dim_{i}") / re).cast(Float32).alias(f"{dataset}_{param}_Re_{axis}")
                    )
                    frames.append(lf_out)
            elif param == "Tpr" and dataset == "wind_swe":
                if y.size == 0:
                    continue
                lf_out: LazyFrame = lf.select(
                    col("time"),
                    ((mp * col("value")**2) / (2 * k)).cast(Float32).alias(f"{dataset}_T")
                )
                frames.append(lf_out)
            elif len(y.shape) > 1:
                for i, axis in enumerate(['x', 'y', 'z']):
                    if y[:, i].size == 0:
                        continue
                    lf_out: LazyFrame = lf.select(
                        col("time"),
                        col(f"dim_{i}").cast(Float32).alias(f"{dataset}_{param}_{axis}")
                    )
                    frames.append(lf_out)
            else:
                if y.size == 0:
                    continue
                lf_out: LazyFrame = lf.select(
                    col("time"),
                    col("value").cast(Float32).alias(f"{dataset}_{param}")
                )
                frames.append(lf_out)
    
    # Clear all tplot variables from memory.
    del_data('*')

    # Check for an empty list.
    if not frames:
        raise ValueError("No valid CDAWeb data was loaded.")

    # Join all LazyFrames on time.
    result: LazyFrame = frames[0]
    for lf in frames[1:]:
        result: LazyFrame = result.join(lf, on="time", how="full", coalesce=True)

    # Filter the data with 'start' and 'end' times.
    filtered: LazyFrame = result.filter(
        (col("time") >= lit(start).str.to_datetime("%Y-%m-%d %H:%M:%S")) &
        (col("time") <= lit(end).str.to_datetime("%Y-%m-%d %H:%M:%S"))
    ).sort("time") # Sort by time.

    del result
    collect()

    return filtered


def get_cdaweb_mission_data(cdaweb_data: LazyFrame, mission: str) -> LazyFrame:
    """Returns the data from the given mission in a Polars LazyFrame."""

    names: list[str] = cdaweb_data.collect_schema().names()
    columns: list[str] = [
        name for name in names if name == "time" or name.startswith(mission)
    ]
    return cdaweb_data.select(columns)


def get_supermag_data(
    path: str,
    file: str,
    fill: int = FILL,
    targets: list[str] = TARGETS
) -> LazyFrame:
    """Returns SuperMAG data in a Polars LazyFrame."""
    
    # Load the .csv with the target variables to a LazyFrame.
    supermag: LazyFrame = (
        scan_csv(f"{path}{file}")
        .with_columns(
            col("Date_UTC")
            # Check the .csv datetime format and use the correct format string.
            .str.strptime(Datetime, "%Y-%m-%d %H:%M:%S", strict=False) #"%Y-%m-%dT%H:%M:%S%.f"  
            .dt.round("1m")          
            .cast(Datetime("ms")) # To match CDAWeb datetime.
            .alias("index")
        )
        .drop("Date_UTC")
        .sort("index")
        .set_sorted("index")
    ).select(targets + ["index"])
    
    # Drop all rows where any target has the fill value.
    fills: Expr = all_horizontal(col(targets) != fill)
    supermag_filtered: LazyFrame = (
        supermag
        .filter(fills)
        .fill_nan(None)
        .unique(subset=["index"])
    )

    return supermag_filtered


def replace_cdaweb_unphysical_values(
    cdaweb_data: LazyFrame, mission: str, limits: dict = CDAWEB_LIMITS
) -> LazyFrame:
    """Returns the data with unphysical values replaced with None."""
    
    for m in cdaweb_data.collect_schema().names():
        if m == "time":
            continue
        # Extract the measurement name (e.g. 'BGSM_x' or 'SC_pos_GSM_Re_x').
        measurement = "_".join(m.split("_")[2:])
        
        if measurement in limits:
            # Get minimum and maximum physical values.
            min_val: float; max_val: float
            min_val, max_val = limits[measurement]
            # Set values outside the limits to None.
            cdaweb_data: LazyFrame = (
                cdaweb_data
                .with_columns(
                    when(col(m).cast(Float32).is_between(min_val, max_val))
                    .then(col(m))
                    .otherwise(None)
                    .alias(m)
                )
            )
    
    # Handle possible zero velocity magnitude.
    prefix: str = '' if mission == "both" else f"{mission}_swe_"
    vx: str = f"{prefix}V_GSM_x"
    vy: str = f"{prefix}V_GSM_y"
    vz: str = f"{prefix}V_GSM_z"
    v_mag_sq: float = col(vx)**2 + col(vy)**2 + col(vz)**2
    cdaweb_physical: LazyFrame = cdaweb_data.with_columns([
        when(v_mag_sq == 0).then(None).otherwise(col(vx)).alias(vx),
        when(v_mag_sq == 0).then(None).otherwise(col(vy)).alias(vy),
        when(v_mag_sq == 0).then(None).otherwise(col(vz)).alias(vz),
    ])
    
    return cdaweb_physical


def interpolate_cdaweb_data(cdaweb_data: LazyFrame, max_gap: int) -> LazyFrame:
    """Returns the data after interpolating gaps up to 'max_gap' minutes."""

    names: list[str] = cdaweb_data.collect_schema().names()
    cols: list[str] = [c for c in names if c != "time"]
    
    for name in cols:
        # Replace NaNs with Nones so Polars identifies them as nulls.
        cdaweb_data: LazyFrame = (
            cdaweb_data
            .sort("time")
            .set_sorted("time")
            .with_columns([col(name).fill_nan(None).alias(name)])
        )
        # A boolean mask for missing values.
        is_null: Expr = col(name).is_null()
        # Identify gap groups by cumulative sum of non-null values.
        gap_id: Expr = (~is_null).cast(int).cum_sum()
        # Calculate length of consecutive nulls per group.
        gap_len: Expr = is_null.cast(int).sum().over(gap_id)

        cdaweb_data: LazyFrame = (
            cdaweb_data
            .with_columns([(
                # Interpolate values for gaps of length up to 'max_gap'.
                when(is_null & (gap_len <= max_gap))
                .then(col(name).interpolate(method="linear"))
                .otherwise(col(name))
                .alias(name)
            )]))
        
        # Backward fill for edge NaNs.
        cdaweb_data: LazyFrame = cdaweb_data.with_columns(
            [col(c).fill_null(strategy="backward").alias(c) for c in cols]
        )

    return cdaweb_data


def resample_cdaweb_data(cdaweb_data: LazyFrame, mission: str) -> LazyFrame:
    """Returns a LazyFrame with CDAWeb data resampled to one minute intervals,
    with propagation delay column, and time shifted by the delay."""

    # Add propagation delay and shift the sample times.
    cdaweb_delayed: LazyFrame = add_propagation_delay(cdaweb_data, mission)

    names: list[str] = cdaweb_delayed.collect_schema().names()
    cols: list[str] = [c for c in names if c != "time"]

    cdaweb_resampled: LazyFrame = (
        cdaweb_delayed
        .sort("time")
        .set_sorted("time")
        # Create 1 minute bins and set the average of each bin as the value.
        .group_by_dynamic("time", every="1m")
        .agg([col(c).mean() for c in cols])
        .rename({"time": "index"}) # Rename 'time' column to 'index'.
    )

    del cdaweb_delayed
    collect()

    return cdaweb_resampled


def quantile_filter(
    data: LazyFrame, 
    q1: float = 0.01, 
    q2: float = 0.99,
    columns: list[str] | None = None,
    excluded: list[str] = ["index", "unix_time"]
) -> DataFrame:
    """Filters the 'input' parquet 'columns', excluding 'excluded', by the given 
    quantiles 'q1' and 'q2' and returns the filtered data in a LazyFrame."""

    if not columns: # Get column names, excluding 'excluded'.
        columns: list[str] = [c for c in data.columns if c not in excluded]

    # Get the quantiles.
    quantiles: DataFrame = data.select([
        col(c).quantile(q1).alias(f"{c}_q{q1}") for c in columns
    ] + [
        col(c).quantile(q2).alias(f"{c}_q{q2}") for c in columns
    ])#.collect()

    # Get a filter for all individual columns.
    conditions: list[Expr] = []
    for c in columns:
        lower: float = quantiles[f"{c}_q{q1}"][0]
        upper: float = quantiles[f"{c}_q{q2}"][0]
        conditions.append(col(c).is_between(lower, upper))
    
    # Get a filter for rows where all columns are within bounds.
    combined_condition: Expr = all_horizontal(conditions)
    
    # Filter the data and return it as a LazyFrame.
    return data.filter(combined_condition)


def shift_targets(data: LazyFrame, targets: list[str], shift: int) -> LazyFrame:
    """Returns a new LazyFrame with 'target' column shifted 'minutes' forward."""
    return data.with_columns([
        col(t).shift(-shift).alias(f"{t}") for t in targets
    ]).drop_nulls()


def add_time_features(data: LazyFrame, index: str = "index") -> LazyFrame:
    """Returns a new LazyFrame with added time-based columns."""
    return data.with_columns((col(index).cast(Int64) / 1e9).alias("unix_time"))

    return data.with_columns([
        (col(index).cast(Int64) / 1e9).alias("unix_time"),
        (col(index).dt.month() * (2 * pi / 12)).sin().alias("month_sin"),
        (col(index).dt.month() * (2 * pi / 12)).cos().alias("month_cos"),
        (col(index).dt.hour() * (2 * pi / 24)).sin().alias("hour_sin"),
        (col(index).dt.hour() * (2 * pi / 24)).cos().alias("hour_cos"),
        (col(index).dt.ordinal_day() * (2 * pi / 365)).sin().alias("day_sin"),
        (col(index).dt.ordinal_day() * (2 * pi / 365)).cos().alias("day_cos"),
    ])


def load_preprocessed_data(file_path: str, print_cols: bool = False) -> LazyFrame:
    """Loads a LazyFrame from data_path."""
    if path.exists(file_path):
        data: LazyFrame = scan_parquet(file_path)
        if print_cols:
            print(f"Loaded data column names: {data.collect_schema().names()}")
        return data
    else:
        raise FileNotFoundError(f"No data found at {file_path}")
    

def save_processed_data(data: DataFrame | LazyFrame, path: str) -> None:
    """Saves the data to datapath as a parquet file."""
    try:
        if isinstance(data, DataFrame):
            data.write_parquet(path)
        elif isinstance(data, LazyFrame):
            data.sink_parquet(path)
        print(f"\nProcessed data saved to {path}")
    except Exception as e:
        print(f"\nData saving error: {e}")
        raise
