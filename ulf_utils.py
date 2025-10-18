"""Utility functions for ULF prediction main script."""

from gc import collect
from glob import glob
from pathlib import Path

from numpy import ndarray, repeat, tile
from pandas import DataFrame as PandaFrame, Series, to_datetime
from polars import (
    all_horizontal,
    col,
    concat,
    DataFrame,
    Datetime,
    LazyFrame,
    len as plen,
    when,
)
from xarray import Dataset, open_dataset

from data_utils import add_lagged_values, shift_targets, quantile_filter
from ulf_constants import COLUMNS, SUPERMAG_DATA_PATH, TARGETS
from ulf_ulf_functions import add_ulf_power
from utils import align_cdaweb_and_supermag, get_X_and_y_split, get_elapsed_time


def load_netcdf(
    columns: list[str],
    start: str,
    end: str,
    file: str | None = None,
    folder: str = SUPERMAG_DATA_PATH
) -> LazyFrame:
    """Returns the data loaded from .netcdf files found from 'folder' in a 
    LazyFrame. Data is filtered to magnetic latitudes between and including 
    'mag_lat_min' and 'mag_lat_max'. 'start_time' and 'end_time' filter the 
    data to a time range similarly."""

    # Get the filenames from the folder.
    if not file:
        paths: list[str] = sorted(glob(f"{folder}downloads/*.netcdf"))
    else:
        paths: list[str] = sorted(glob(f"{folder}downloads/{file}"))
    
    # Magnetic latitude range.
    mag_lat_min: int = 60
    mag_lat_max: int = 70
    
    # A list for LazyFrames.
    frames: list[str] = []

    # Looping all the files and saving yearly .parquet files.
    for filepath in paths:
        ds: Dataset = open_dataset(filepath)

        # Create time index.
        time: PandaFrame = ds[
            ["time_yr", "time_mo", "time_dy", "time_hr", "time_mt", "time_sc"]
        ].to_dataframe()
        time.columns = ["year", "month", "day", "hour", "minute", "second"]
        time_vals: Series = to_datetime(time)
        
        # Apply time filter.
        time_mask: Series = (start <= time_vals) & (time_vals <= end)
        indices: Series = time_vals[time_mask]
        
        # Get station IAGAs.
        station_ids: ndarray = ds["id"].values[0]
        # Get magnetic latitudes.
        mag_lats: ndarray = ds["mlat"].values[0] 
        # Get stations in the latitude range.
        filtered_stations: ndarray = station_ids[
            (mag_lat_min <= mag_lats) & (mag_lats <= mag_lat_max)
        ]

        # Array lengths.
        n_times: int = len(indices)
        n_stations: int = len(filtered_stations)
        total_rows: int = n_times * n_stations
        
        # Dictionary for creating a Polars LazyFrame.
        data: dict[str, ndarray] = {
            "index": repeat(indices.values, n_stations), 
            "station": tile(filtered_stations, n_times)
        }
        
        excluded: set[str] = {"index", "station"}
        # Add measurement columns with consistent length.
        for c in columns:
            if c not in excluded:
                values: ndarray = ds[c].values # Get column values. 
                filtered: ndarray = values[time_mask.values] # Filter values.
                data[c] = filtered.ravel()[:total_rows] # Ensure correct length.
    
        # Create Polars LazyFrame from the data dictionary.
        frame: LazyFrame = (
            LazyFrame(data) 
            .with_columns([ 
                # Cast indices to ms precision for CDAWeb-data compatibility.
                col("index").cast(Datetime("ms")),
                # Change NaN-values to None for Polars functions.
                *[col(c).fill_nan(None).alias(c) 
                for c in columns if c not in excluded]
            ])
        )
        year: str = Path(filepath).stem
        frame.sink_parquet(f"{folder}{year}.parquet")

        # Add the LazyFrame to the list.
        frames.append(frame)
    
    # Concatenate all frames and sort by time.
    if len(frames) == 1:
        return frames[0]
    else:
        return concat(frames).sort("index")


def add_bh(netcdf: LazyFrame, columns: list[str], max_gap: int = 5) -> LazyFrame:
    """Returns a LazyFrame with 'columns' and horizontal magnetic component 
    added as 'bh' column. Maximum allowed gap where no station has data is 
    defined by 'max_gap'."""
    
    # Identify rows where all measurement columns are NaN.
    cols: list[str] = [c for c in columns if c not in {"index","station"}]
    flagged: LazyFrame = netcdf.with_columns(
        # 'all_nan' col has 'True', if all measurement columns are NaN.
        all_nan = all_horizontal([col(c).is_null() for c in cols])
    )
    
    # Group consecutive all-NaN rows and count their size.
    grouped: LazyFrame = (
        flagged
        .with_columns(
        # 'nan_group' has an int id for all consecutive rows with the same 'all_nan'.
            nan_group = (col("all_nan") != col("all_nan").shift(1)).cum_sum()
        )
        .with_columns(
        # 'group_size' has the count of rows in each sequence of consecutive all-NaN rows.
            group_size = (
                when(col("all_nan"))
                .then(plen().over("nan_group"))
                .otherwise(None) # None for rows not in a all-NaN sequence.
            )   
        )
    )
    
    # Keep rows where 'group size' < 'max_gap'.
    supermag: LazyFrame = (
        grouped
        .filter((col("group_size").is_null()) | (col("group_size") < max_gap))
        .drop(["all_nan", "nan_group", "group_size"])  # Drop temporary columns.
        .with_columns( # Interpolate and fill dbe_nez and dbn_nez.
            col("dbe_nez").forward_fill().backward_fill().interpolate().alias("dbe_nez"),
            col("dbn_nez").forward_fill().backward_fill().interpolate().alias("dbn_nez"),   
        )
    )

    del netcdf, grouped
    
    # Add horizontal component of magnetic field.
    return supermag.with_columns((col("dbn_nez")**2 + col("dbe_nez")**2).sqrt().alias("bh"))   


def get_data(
    CDAWeb: LazyFrame,
    file: str,
    start: str,
    end: str,
    script_start: float,
    targets: list[str] = TARGETS,
    columns: list[str] = COLUMNS,
) -> tuple[DataFrame, DataFrame, DataFrame] | tuple[DataFrame, DataFrame]:
    """Loads SuperMAG data, adds Pc5 ULF wave power and lagged values, and 
    combines the data with CDAWeb data. Returns X and y for training models."""
    
    # Load .NetCDF data.
    NetCDF_data: LazyFrame = load_netcdf(columns, start, end, file)
    t: float = get_elapsed_time(script_start)
    print(f"NetCDF file loaded, {t} minutes from script start.")
    
    # Add horizontal component of magnetic field.
    SuperMAG_data: LazyFrame = add_bh(NetCDF_data, columns)
    t: float = get_elapsed_time(script_start) 
    print(f"Horizontal magnetic component added, {t} minutes from script start.")

    # Add ULF to the data.
    SuperMAG_with_ulf: LazyFrame = add_ulf_power(SuperMAG_data)
    t: float = get_elapsed_time(script_start)
    print(f"Pc5 ULF power added, {t} minutes from script start.")
    
    # Align CDAWeb and SuperMAG indices and merge the data.
    data: LazyFrame = align_cdaweb_and_supermag(CDAWeb, SuperMAG_with_ulf)
    t: float = get_elapsed_time(script_start)
    print(f"CDAWeb and SuperMAG data combined, {t} minutes from script start.")
    
    # Filter the data.
    q1: float = 0.01 
    q2: float = 0.99
    filtered: LazyFrame = quantile_filter(data.collect(), q1, q2)
    t: float = get_elapsed_time(script_start)
    print(f"Data filtered between {q1*100}% and {q2*100}% quantiles, {t} minutes from script start.")
    
    # Shift targets for nowcasting three hours into the future.
    shift: int = 180
    shifted: LazyFrame = shift_targets(filtered.lazy(), targets, shift)
    t: float = get_elapsed_time(script_start)
    print(f"Targets shifted {shift} minutes, {t} minutes from script start.")
    
    # Add lagged values to the data.
    training_data: LazyFrame = add_lagged_values(shifted, targets=targets)
    t: float = get_elapsed_time(script_start)
    print(f"Lagged targets added, {t} minutes from script start.")
    
    del SuperMAG_data, SuperMAG_with_ulf, data, filtered, shifted
    collect()

    # Split data into X and y.
    X: DataFrame; y: DataFrame
    X, y = get_X_and_y_split(training_data, targets)
    t: float = get_elapsed_time(script_start)
    print(f"Data splitted into X and y, {t} minutes from script start.")

    # Return training data.
    return X, y
