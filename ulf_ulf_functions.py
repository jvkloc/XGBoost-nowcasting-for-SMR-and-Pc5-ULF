"""Functions for adding Pc5 power to the data."""

from datetime import datetime, timedelta
from gc import collect
from typing import Callable

from polars import col, concat, DataFrame, Datetime, Expr, LazyFrame, lit

from ulf_constants import SECTORS
from ulf_pc5_functions import Pc5_expr


def add_ulf_power(supermag: LazyFrame, overlap = timedelta(hours=2)) -> LazyFrame:
    """Returns SuperMAG data with global and sector Pc5 power added in 
    new columns. Processes each day separately, using 'overlap' for overlapping 
    the days to minimize wavelet edge effects."""
    
    # Min and max date.
    meta: DataFrame = supermag.select([
        col("index").min().alias("min_date"),
        col("index").max().alias("max_date")
    ]).collect()

    min_date: datetime = meta.get_column("min_date").item()
    max_date: datetime = meta.get_column("max_date").item()
    
    # A list of days from min date to max date.
    current_date: datetime = min_date
    days: list[datetime] = []
    while current_date <= max_date:
        days.append(current_date)
        current_date += timedelta(days=1)

    # List for processed data.
    data: list[LazyFrame] = []
    # Loop all the days.
    for d in days:
        # Add overlaps.
        end_overlap = overlap + timedelta(days=1)
        
        s: datetime = d - overlap if d != days[0] else d
        e: datetime = d + end_overlap if d != days[-1] else d + timedelta(days=1)
        
        # Convert back to Polars Datetime.
        start: Expr = lit(s).cast(Datetime("ms"))
        end: Expr = lit(e).cast(Datetime("ms"))

        # Add Pc5 columns.
        pc5: LazyFrame | None = add_Pc5_columns(supermag, (start, end), d)
        
        if pc5 is not None:
            data.append(pc5)
    
    # Combine the daily data to one LazyFrame.
    ulf_added: LazyFrame = concat(data, how="vertical")

    del data
    collect

    return ulf_added


def get_sector_Pc5_power(
    supermag_day: LazyFrame, 
    sectors: list[tuple[str, Callable]] = SECTORS
) -> LazyFrame:
    """Returns the data with 'Pc5_{sector}' column added for each sector."""
    sector_avgs: list[LazyFrame] = []
    for sector, get_rows in sectors:
        # Get rows of the sector.
        data: LazyFrame = supermag_day.filter(
            get_rows(col("mlt")) & col("Pc5").is_finite()
        )
        
        # Get sector average.
        sector_avg: LazyFrame = (
            data
            .group_by("index")
            .agg(col("Pc5").mean().alias(f"Pc5_{sector}"))
        )
        
        if lazyframe_is_empty(sector_avg): # No data, add nulls.
            nulls: LazyFrame = supermag_day.with_columns(
                col("index"),
                lit(None).alias(f"Pc5_{sector}")
            )
            sector_avgs.append(nulls)
        else: # Append sector data.
            sector_avgs.append(sector_avg)
           
    # Join sector averages to the data.
    for sector_avg in sector_avgs:
        supermag_day: LazyFrame = supermag_day.join(sector_avg, on="index", how="left")

    return supermag_day


def get_global_Pc5_power(supermag_day: LazyFrame) -> LazyFrame:
    """"Returns the data with 'Pc5_global' column."""
    
    global_pc5: LazyFrame = (
        supermag_day
        .filter(col("Pc5").is_finite())
        .group_by("index")
        .agg(col("Pc5").mean().alias("Pc5_global"))
    )
    
    # Join global average to the data.
    return supermag_day.join(global_pc5, on="index", how="left")


def get_Pc5_power(supermag_day: LazyFrame) -> LazyFrame:
    """Returns the data with 'Pc5_power' column added. Computes the power 
    station-wise, because of the wavelet function, which would propagate NaN 
    values of one station to all stations otherwise."""
    
    pc5_values: LazyFrame = (
        supermag_day
        .sort(["station", "index"])
        .group_by("station", maintain_order=True)
        .agg([
            col("index"),
            Pc5_expr(col("bh")).alias("Pc5")
        ])
        .explode(["index", "Pc5"])
    )

    # Join Pc5_power to the data.
    return supermag_day.join(pc5_values, on=["station", "index"], how="left")


def add_Pc5_columns(
    supermag: LazyFrame,
    time_range: tuple[Expr, Expr],
    day: datetime,
) -> LazyFrame | None:
    """Returns the data in a LazyFrame with global and sector Pc5 power of the 
    given day added."""
    
    # Filter data for the day.
    s: Expr; e: Expr
    s, e = time_range
    data: LazyFrame = (supermag.filter((col("index") >= s) & (col("index") < e)))
    
    if lazyframe_is_empty(data):
        print(f"No SuperMAG data from {day}")
        return None

    # Add Pc5 power for all stations.
    data: LazyFrame = get_Pc5_power(data)
   
    # Add global Pc5 power as an average of all stations.
    data: LazyFrame = get_global_Pc5_power(data)
    
    # Get Pc5 power for each sector.
    data: LazyFrame = get_sector_Pc5_power(data)
    
    # Deduplicate the indices by averaging all values of a given time index.
    deduplicated: LazyFrame = deduplicate_supermag_indices(data.drop("station"))
    
    # Return updated data without 2-hour overlap.
    return deduplicated.filter(
        (col("index") >= lit(day).cast(Datetime("ms"))) &
        (col("index") < lit(day + timedelta(days=1)).cast(Datetime("ms")))
    )


def deduplicate_supermag_indices(supermag: LazyFrame) -> LazyFrame:
    """Returns SuperMAG data with deduplicated time indices i.e. indices 
    changed from one per station per timestamp to one per timestamp. The 
    one remaining timestamp contains the average of all stations' values."""

    # Get column names.
    names: list[str] = supermag.collect_schema().names()
    columns: list[str] = [n for n in names if n != "index"]
    
    # Set averages to each timestamp.
    return (
        supermag
        .group_by("index") # Group by timestamp.
        .agg([col(c).mean().alias(c) for c in columns]) # Average the values.
    )


def lazyframe_is_empty(lf: LazyFrame) -> bool:
    """Returns a Boolean indicating if a Polars LazyFrame is empty."""
    return lf.limit(1).collect().is_empty()
