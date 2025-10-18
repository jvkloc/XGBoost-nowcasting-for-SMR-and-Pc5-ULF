"""Functions for printing various statistics from data."""

from datetime import datetime

from polars import (
    DataFrame,
    Datetime,
    Expr,
    col,
    LazyFrame,
    len as plen,
    Series,
    when,
)

from ulf_constants import SECTORS


def print_delay_fill_value_count(cdaweb: LazyFrame) -> None:
    """Prints the count of 'L1_delay' column fill values."""
    print(
        "CDAWeb rows where L1_delay = 3600.0 (fill value):",
        (
            cdaweb
            .filter(col("L1_delay") == 3600.0)
            .select(plen())
            .collect()
            .item()
        )
    )


def print_column_nulls(
    data: LazyFrame, column: Expr = col("BH"), name: str = "BH") -> None:
    """Prints the amount of leading and trailing nans, if present, as well
    as total nans of the given 'column' Expression."""

    null_counts: DataFrame = data.select([
        column.is_null().sum().alias("total_nulls"),
        column.is_null().arg_min().alias("first_non_null_idx"),
        column.is_null().reverse().arg_min().alias("trailing_nulls")
    ]).collect()
    
    total_nulls: int = null_counts["total_nulls"].item()
    first: int = null_counts["first_non_null_idx"].item()
    trailing: int = null_counts["trailing_nulls"].item()
    total_rows: int = data.select(plen()).collect().item()
    
    if first is None:  # All values are null.
        print(f"All {total_rows} rows in '{name}' column are null.")
        return
    if first > 0:
        print(f"{first} leading nulls in '{name}' column (out of {total_rows} rows)")
    if trailing is not None and trailing > 0:
        print(f"{trailing} trailing nulls in '{name}' column (out of {total_rows} rows)")

    print(f"Total nulls in '{name}' column: {total_nulls}/{total_rows} ({total_nulls/total_rows:.1%})")


def print_column_nans(
    data: LazyFrame, column: Expr = col("BH"), name: str = "BH") -> None:
    """Prints the amount of leading and trailing nans, if present, as well
    as total nans of the given 'column' Expression. Note that is_nan() is 
    not supported for strings."""

    nan_counts: DataFrame = data.select([
        column.is_nan().sum().alias("total_nans"),
        column.is_nan().arg_min().alias("first_non_nan_idx"),
        column.is_nan().reverse().arg_min().alias("trailing_nans")
    ]).collect()
    
    total_nans: int = nan_counts["total_nans"].item()
    first: int = nan_counts["first_non_nan_idx"].item()
    trailing: int = nan_counts["trailing_nans"].item()
    total_rows: int = data.select(plen()).collect().item()
    
    if first is None:  # All values are nan
        print(f"All {total_rows} rows in '{name}' column are nan.")
        return
    if first > 0:
        print(f"{first} leading nans in '{name}' column (out of {total_rows} rows)")
    if trailing is not None and trailing > 0:
        print(f"{trailing} trailing nans in '{name}' column (out of {total_rows} rows)")

    print(f"Total nans in '{name}' column: {total_nans}/{total_rows} ({total_nans/total_rows:.1%})")


def print_Pc5_statistics(y: DataFrame) -> None:
    """Prints Pc5 statistics."""
    print("Global:", y['Pc5_global'].describe())
    print("Dawn:", y['Pc5_Dawn'].describe())
    print("Day:", y['Pc5_Day'].describe())
    print("Dusk:", y['Pc5_Dusk'].describe())
    print("Night:", y['Pc5_Night'].describe())


def print_supermag_large_gaps(supermag: LazyFrame, max_gap: int = 5) -> None:
    """Prints nuber of data gaps larger than 'max_gap' and the number of 
    SuperMAG rows in those gaps. Note that this works only within 
    load_supermag_data()."""
    
    # Count gaps larger than 'max_gap' and total rows in those gaps.
    gap_stats: LazyFrame = (
        supermag
            .filter(col("all_nan"))
            .group_by("nan_group")
            .agg(plen().alias("gap_size"))
            .filter(col("gap_size") >= max_gap)
    )
    
    # Execute the computation.
    result: DataFrame = gap_stats.select([
        plen().alias("num_large_gaps"),
        col("gap_size").sum().alias("total_gap_rows")
    ]).collect()

    # Extract the results.
    num_large_gaps: int = result["num_large_gaps"][0]
    total_gap_rows: int = result["total_gap_rows"][0]
    
    # Print the number of gaps larger than max_gap and rows in those gaps.
    print(f"Number of SuperMAG data gaps > {max_gap} minutes: {num_large_gaps}")
    print(f"Total SuperMAG rows in large gaps: {total_gap_rows}")


def print_aligned_data_range_and_rows(X: DataFrame, y: DataFrame, rows: int) -> None:
    """Prints row count of the time-aligned data and time ranges of X and y. 
    Note that this works only within align_cdaweb_and_supermag()."""
    
    print(f"Row count of the combined and time-aligned data: {rows}\n")

    # X min and max 'index'.
    x_min: datetime = X.select(col("index").min()).item()
    x_max: datetime = X.select(col("index").max()).item()
    
    # y min and max 'index'.
    y_min: datetime = y.select(col("index").min()).item()
    y_max: datetime = y.select(col("index").max()).item()

    # Print X and y time ranges.
    print(f"X time range: {x_min} to {x_max}")
    print(f"y time range: {y_min} to {y_max}\n")


def print_missing_value_analysis(supermag_with_pc5: LazyFrame) -> None:
    """Printing function to analyse the added Pc5 columns."""
    pc5_vs_mlt: LazyFrame = (
        supermag_with_pc5
        .filter(col("Pc5_power").is_nan() & col("BH").is_not_nan())
        .group_by("IAGA")
        .count()
        .sort("count", descending=True)
    )
    print("\nStations with most wavelet-generated NaNs (BH not NaN but Pc5_power is NaN):")
    print(pc5_vs_mlt.collect())
    
    pc5_nans: LazyFrame = (
        supermag_with_pc5
        .filter(col("Pc5_power").is_nan())
        .select(["BH", "IAGA", "index"])
    )
    print("\nSample rows with Pc5_power NaNs (all types):")
    print(pc5_nans.limit(20).collect())

    mlt_distribution: LazyFrame = supermag_with_pc5.group_by("MLT").len().sort("MLT")
    print("\nMLT distribution (sector coverage check):")
    print(mlt_distribution.collect())
    
    mlt_vs_pc5_night: LazyFrame = (
        supermag_with_pc5
        .filter(col("Pc5_Night").is_null())
        .group_by("MLT")
        .count()
        .sort("MLT")
        )
    print("\n[e] MLT distribution when Pc5_Night is null:")
    print(mlt_vs_pc5_night.collect())


def print_bh_constant_cols(supermag_day: LazyFrame) -> None:
    """Prints all-constant and all-NaN BH cols."""
    debug_info: LazyFrame = (
        supermag_day
        .sort(["IAGA", "index"])
        .group_by("IAGA", maintain_order=True)
        .agg(
            bh_values=col("BH"),
            is_constant=col("BH").eq(col("BH").first()).all(),  # Check if all BH values are the same
            is_all_nan=col("BH").is_nan().all(),  # Check if all BH values are NaN
        )
    )

    # Collect and print debug info for constant or all-NaN cases
    debug_df: DataFrame = debug_info.collect()
    for row in debug_df.rows(named=True):
        if row["is_constant"] or row["is_all_nan"]:
            print(f"Station {row['IAGA']}: BH values = {row['bh_values']}, "
                  f"Constant: {row['is_constant']}, All NaN: {row['is_all_nan']}")
            

def more_pc5_mlt_debugging(supermag_day: LazyFrame) -> None:
    """Comprehensive diagnostics for Pc5 power computation issues"""
    
    # 1. Station-Level BH Statistics
    bh_stats = (
        supermag_day
        .filter(col("IAGA").is_in(["BEY", "CPS", "T55"]))
        .group_by("IAGA")
        .agg([
            # Basic statistics
            col("BH").count().alias("total_count"),
            col("BH").is_nan().mean().alias("nan_ratio"),
            col("BH").filter(col("BH").is_not_nan()).mean().alias("mean_BH"),
            col("BH").filter(col("BH").is_not_nan()).std().alias("std_BH"),
            
            # Data quality flags
            (col("BH").diff().abs().max() == 0).alias("has_flat_segments"),
            (abs(col("BH")) > 1e4).mean().alias("extreme_value_ratio"),
            
            # Temporal distribution
            col("index").filter(col("BH").is_nan()).min().alias("first_nan_time"),
            col("index").filter(col("BH").is_nan()).max().alias("last_nan_time"),
            
            # New: Value bounds check
            col("BH").filter(col("BH").is_not_nan()).min().alias("min_BH"),
            col("BH").filter(col("BH").is_not_nan()).max().alias("max_BH")
        ])
        .collect()
    )
    print("\n[1] Station-Level BH Statistics:")
    print(bh_stats)
    
    # 2. Sample Problematic BH Values (Enhanced)
    problematic_bh = (
        supermag_day
        .filter(
            (col("IAGA").is_in(["BEY", "T55"])) &
            (
                col("BH").is_nan() |
                (abs(col("BH")) > 1e4) |
                (col("BH").diff().abs() == 0)  # Flat segments
            )
        )
        .select(["IAGA", "index", "BH"])
        .limit(20)
        .collect()
    )
    print("\n[2] Sample Problematic BH Values:")
    print(problematic_bh)
    
    # 3. MLT Distribution Analysis
    mlt_dist = (
        supermag_day
        .group_by((col("MLT") // 1).alias("mlt_bin"))  # 1-hour bins
        .agg(plen())
        .sort("mlt_bin")
        .collect()
    )
    print("\n[3] MLT Distribution (1-hour bins):")
    print(mlt_dist)
    
    # 4. Conditional Pc5 Power Analysis
    if "Pc5_power" in supermag_day.collect_schema().names():
        # Pc5 Power Statistics
        power_stats = (
            supermag_day
            .filter(col("IAGA").is_in(["BEY", "CPS", "T55"]))
            .group_by("IAGA")
            .agg([
                col("Pc5_power").is_nan().mean().alias("power_nan_ratio"),
                col("Pc5_power").mean().alias("mean_power"),
            ])
            .collect()
        )
        print("\n[4] Pc5 Power Statistics:")
        print(power_stats)
        
        # Sector Null Analysis (Enhanced)
        for sector, _ in SECTORS:
            sector_nulls = (
                supermag_day
                .filter(col(f"Pc5_{sector}").is_null())
                .group_by("MLT")
                .agg([
                    plen().alias("null_count"),
                    col("IAGA").n_unique().alias("unique_stations")
                ])
                .sort("MLT")
                .collect()
            )
            print(f"\n[5] {sector} Sector Nulls by MLT:")
            print(sector_nulls)


def analyze_mlt_dip(supermag_day: LazyFrame) -> None:
    """Comprehensive analysis of the midnight MLT dip"""
    
    # 1. Basic station distribution in the dip
    midnight_stations = (
        supermag_day
        .filter((col("MLT") >= 23) | (col("MLT") < 1))
        .select(["IAGA", "GEOLAT", "GEOLON"])
        .unique()
        .collect()
    )
    print("\n[1] Stations Active in MLT 23-1 Range:")
    print(f"Total stations: {len(midnight_stations)}")
    print(midnight_stations)
    
    # 2. Comparison with other MLT bins
    mlt_comparison = (
        supermag_day
        .with_columns(
            when((col("MLT") >= 23) | (col("MLT") < 1))
              .then("midnight")
              .otherwise("other").alias("mlt_category")
        )
        .group_by("mlt_category")
        .agg([
            plen().alias("count"),
            col("GEOLAT").mean().alias("mean_lat"),
            col("GEOLON").mean().alias("mean_lon")
        ])
        .collect()
    )
    print("\n[2] MLT Category Comparison:")
    print(mlt_comparison)
    
    # 3. Data quality metrics in the dip
    midnight_quality = (
        supermag_day
        .filter((col("MLT") >= 23) | (col("MLT") < 1))
        .agg([
            col("BH").is_nan().mean().alias("nan_ratio"),
            (abs(col("BH")) > 500).mean().alias("spike_ratio"),
            col("Pc5_power").is_nan().mean().alias("power_nan_ratio")  # If available
        ])
        .collect()
    )
    print("\n[3] Midnight Data Quality:")
    print(midnight_quality)
    
    # 4. Temporal patterns
    midnight_hourly = (
        supermag_day
        .filter((col("MLT") >= 23) | (col("MLT") < 1))
        .with_columns(col("index").dt.hour().alias("utc_hour"))
        .group_by("utc_hour")
        .agg(plen().alias("count"))
        .sort("utc_hour")
        .collect()
    )
    print("\n[4] Midnight Observations by UTC Hour:")
    print(midnight_hourly)


def print_duplicate_indices_count(lf: LazyFrame) -> None:
    """Prints the count of duplicate indices per station."""
    # Get all unique stations.
    stations: LazyFrame = lf.select(
        col("IAGA")
    ).unique().collect().get_column("IAGA")
    print(f"\nFound {len(stations)} unique stations:")
    print(stations)

    print("\nDuplicate timestamp counts per station:")

    # Group by station, then check for duplicates in 'index'
    for station in stations:
        duplicates: LazyFrame = (
            lf
            .filter(col("IAGA") == station)
            .group_by("index")
            .agg(plen().alias("count"))
            .filter(col("count") > 1)
            .select(plen().alias("duplicate_timestamps"))
            .collect()
            .item()
        )
        print(f"{station}: {duplicates} duplicate timestamps")


def print_lazyframe_time_ranges(cdaweb: LazyFrame, supermag: LazyFrame) -> None:
    """Prints the time ranges of the given LazyFrames."""
    cdaweb_range: LazyFrame = cdaweb.select([
        col("index").min().alias("min_time"),
        col("index").max().alias("max_time")
    ]).collect()
    print(f"CDAWeb time range start: {cdaweb_range['min_time'][0]}")
    print(f"CDAWeb time range end:   {cdaweb_range['max_time'][0]}")
    supermag_range: LazyFrame = supermag.select([
        col("index").min().alias("min_time"),
        col("index").max().alias("max_time")
    ]).collect()
    print(f"SuperMAG time range start: {supermag_range['min_time'][0]}")
    print(f"SuperMAG time range end:   {supermag_range['max_time'][0]}")


def print_memory_usage(lf: LazyFrame) -> None:
    """Prints estimated memory usage of the given LazyFrame."""
    bytes: int = lf.collect().estimated_size()
    kb: float = bytes / 1024
    mb: float = kb / 1024
    print(f"Estimated memory usage: {bytes} bytes")
    print(f"Estimated memory usage: {kb:.2f} KB")
    print(f"Estimated memory usage: {mb:.2f} MB")


def print_data_gaps(gap_info: dict) -> None:
    """Prints data gaps from the given dictionary."""
    print("")
    for data in gap_info:
        longer_gaps: Series = gap_info[data]['gaps']
        index_values: Series = gap_info[data]['indices']
        limit: int = gap_info[data]['limit']
        if len(longer_gaps) > 0:
            print(f"Gaps over {limit} seconds in the {data} data:")
            for gap_size, idx in zip(longer_gaps, index_values):
                try:
                    pos: int = index_values.eq(idx).arg_true()[0]
                    if pos > 0:
                        start = index_values[pos - 1]
                        print(f"Gap of {gap_size} seconds between {start} and {idx}")
                    else:
                        print(f"Gap of {gap_size} seconds starts at index {idx}")
                except IndexError:
                    continue
        else:
            print(f"No gaps over {limit} seconds found from {data} data.")


def print_statistics(
    frame: LazyFrame | DataFrame, columns: list[str] | None = None
) -> None:
    """Print statistics from a Polars LazyFrame or DataFrame:
    - Column names
    - Count of missing values
    - For numeric columns (except "index"): min, max, 0.1/0.9 quantiles, mean, median
    - For "index" column: only min, max, and missing values
    """

    if isinstance(frame, LazyFrame): # Collect the data.
        frame: DataFrame = frame.collect()
    # Get column names if not given as an argument.
    if not columns:
        columns: list[str] = frame.columns
    # Get basic stats
    stats: DataFrame = frame.describe()
    
    print('')
    for column in columns:
        print(f"Column: {column}")
        
        # Missing values count.
        null_count: int = frame[column].null_count()
        print(f"Missing values: {null_count}")
            
        if column == "index":
            min_val: Datetime = stats.filter(col("statistic") == "min").select(column).item()
            max_val: Datetime = stats.filter(col("statistic") == "max").select(column).item()
            print(f"Range: from {min_val} to {max_val}\n")
        else:
            min_val: float = stats.filter(col("statistic") == "min").select(column).item()
            max_val: float = stats.filter(col("statistic") == "max").select(column).item()
            mean: float = stats.filter(col("statistic") == "mean").select(column).item()
            median: float = stats.filter(col("statistic") == "50%").select(column).item()
            q10: float = frame[column].quantile(0.1)
            q90: float = frame[column].quantile(0.9)
            
            print(f"Range: from {min_val} to {max_val}")
            print(f"10% Quantile: {q10}")
            print(f"90% Quantile: {q90}")
            print(f"Mean: {mean}")
            print(f"Median: {median}\n")


def print_quantiles(
    frame: LazyFrame | DataFrame,
    columns: list[str] | None = None,
    q1: float = 0.01,
    q2: float = 0.99
) -> None:
    """Prints 'q1' and 'q2' quantiles of columns of a Polars LazyFrame or DataFrame."""

    if isinstance(frame, LazyFrame): # Collect the data.
        frame: DataFrame = frame.collect()
    
    if not columns: # Get column names if not given as an argument.
        columns: list[str] = [c for c in frame.columns if c != "index"]
    
    for column in columns:
        quantile1: float = frame[column].quantile(q1)
        quantile2: float = frame[column].quantile(q2)
        print(f"{column} {q1}q: {quantile1} | {q2}q: {quantile2}")
