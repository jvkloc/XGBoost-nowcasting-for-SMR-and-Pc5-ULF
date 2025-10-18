"""Functions for printing SuperMAG .csv file info and combining SuperMAG 
.csv files."""

from gc import collect
from glob import glob
from os import path

from polars import (
    all_horizontal,
    col,
    concat,
    DataFrame,
    Datetime,
    Expr,
    LazyFrame,
    read_csv,
    scan_csv,
)

from constants import SMAG_PATH, PATH, FILL, TARGETS


def print_supermag_info(smag_path: str = PATH) -> None:
    """Prints info on the given file from the SuperMAG folder."""
    superMAG: DataFrame = read_csv(smag_path)
    print(superMAG.head())
    print(superMAG.describe())


def load_supermag_data(
    path: str = SMAG_PATH, fill: int = FILL, targets: list[str] = TARGETS
) -> LazyFrame:
    """Returns SuperMAG data in a Polars LazyFrame."""
    # Load the .csv to LazyFrame.
    supermag: LazyFrame = (
        scan_csv(f"{path}")
        .with_columns(
            col("Date_UTC")
            .str.strptime(Datetime, "%Y-%m-%d %H:%M:%S")
        )
        .select(targets + ["Date_UTC"])
        .set_sorted("Date_UTC")
    )
    
    # Filter out rows where any target has the fill value.
    filter: Expr = all_horizontal(col(targets) != fill)
    supermag: LazyFrame = supermag.filter(filter)
    
    # Return the LazyFrame.
    return supermag


def get_dataframe_list(csv_path: str = SMAG_PATH) -> list[LazyFrame]:
    """Creates LazyFrames from all .csv files in the given folder and returns 
    them in a list."""
    # Get .csv file path list from the folder.
    filepaths: list[str] = glob(f"{csv_path}*.csv")
    # Ensure chronological order according to which filenames are named.
    filepaths.sort()
    # Empty list for processed SuperMAG files.
    superMAGs: list[LazyFrame] = []
    # Preprocess all the files.
    for filepath in filepaths:
        # Split the file path into folder and filename.
        folder, file = path.split(filepath)
        # Add / to the end of the path.
        folder += '/'
        # Create a DataFrame and append it to the list.
        superMAGs.append(load_supermag_data(file=file))
    # Return the DataFrame list.
    return superMAGs


def combine_dataframes(frames: list[LazyFrame]) -> LazyFrame:
    """Returns all LazyFrames from the given list combined to a new LazyFrame. 
    Deletes the LazyFrame list and frees the memory allocated for it."""
    combined: LazyFrame = concat(frames, how="vertical")
    del frames # Delete the list given as argument.
    collect()  # Free the memory allocated to the deleted DataFrames list.
    return combined


def save_large_dataframe(
    frame: LazyFrame, csv_path: str = SMAG_PATH, filename: str = "supermag.csv"
) -> None:
    """Saves the DataFrame to a .csv file into the given folder with the given 
    file name."""

    # Add file name to the path.
    file_out: str = path.join(csv_path, filename)
    # Save the Dask DataFrame to a .csv file.
    frame.sink_csv(file_out)


def main() -> None:
    
    # Print SuperMAG .csv file info.
    #print_supermag_info()
    
    # Get all the .csv files converted to Dask DataFrames in a list.
    superMAGs: list[LazyFrame] = get_dataframe_list()
    
    # Combine the Dask DataFrames. 
    combined: DataFrame = combine_dataframes(superMAGs)

    # Save the new DataFrame to the same folder as a .csv file.
    save_large_dataframe(combined)


if __name__=="__main__":
    main()
