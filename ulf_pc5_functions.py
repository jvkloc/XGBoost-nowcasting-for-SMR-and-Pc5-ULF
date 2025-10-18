"""Functions for computing Pc5 power."""

from numpy import (
    all as npall,
    abs as npabs,
    full_like,
    isnan,
    nan, 
    ndarray,
    sum as npsum,
    where
)
from polars import DataFrame, Expr, Float64, LazyFrame, Series
from pycwt import cwt, Morlet
from pycwt.helpers import find


def Pc5_power(bh: ndarray) -> ndarray:
    """Returns the power spectrum (squared magnitude) computed from the 1D 
    time series array BH values input and normalized by scales. Torrence and 
    Compo 1998."""
    
    w0: int = 6         # ω0 parameter for Morlet wavelet.
    dt: float = 60.0    # Timestep in seconds.
    s0: float = 2 * dt  # Starting scale: 2 * 60 s = 2 mins.
    dj: float = 1 / 12  # Twelve sub-octaves per octave.
    J = int(9 / dj)     # 9 octaves with 12 sub-octaves each.
    mother = Morlet(w0) # Mother wavelet function.
    
    # Get the continuous wavelet transform.
    wave: ndarray; scale: ndarray; frequency: ndarray
    wave, scale, frequency, _, _, _ = cwt(bh, dt, dj, s0, J, mother)
    
    # Squared transform scaled.
    scaled_power: ndarray = npabs(wave)**2 / scale[:, None]
    # Period from frequency.
    period: ndarray = 1 / frequency
    # Pc5 period: indices where period is 150s – 600s.
    pc5_period: ndarray = find((period >= 150) & (period < 600))

    # Torrence and Compo 1998 equation 24: scale averaged wavelet power.
    cdelta: float = mother.cdelta
    scaled_power_sum: float = scaled_power[pc5_period, :].sum(axis=0)
    pc5_power: ndarray = dj * dt / cdelta * scaled_power_sum

    # Return Pc5 power.
    return pc5_power


def get_Pc5(station: LazyFrame) -> ndarray:
    """Returns the Pc5 power for the given LazyFrame."""
    bh: ndarray = station.select("BH").collect().get_column("BH").to_numpy()
    return Pc5_power(bh)


def Pc5_expr(col_bh: Expr) -> Expr:
    """Returns a Polars expression that applies the Pc5 power calculation 
    to the 'BH' column."""

    def get_power(s: Series) -> DataFrame:
        """ Returns a Polars DataFrame with a single column 'Pc5_power' 
        containing the computed Pc5 power values. If all 'BH' values are NaN, 
        returns a column of NaN values."""

        bh: ndarray = s.to_numpy()
        # No values or constant BH value result in Nan power,
        if npall(isnan(bh)) or npall(bh == bh[0]): 
            pc5: ndarray = full_like(bh, nan)
        else:
            pc5: ndarray = Pc5_power(bh)
        return Series(pc5)

    return col_bh.map_batches(get_power, return_dtype=Float64)


###### remove the ones below? ###################
def get_BH(station: LazyFrame) -> Expr:
    """Returns the horizontal components of the station's magnetic field values."""
    dbn: Expr = station.get_column("dbn_nez")
    dbe: Expr = station.get_column("dbe_nez")
    return (dbn.pow(2) + dbe.pow(2)).sqrt()

def get_COI(mag_component: ndarray) -> ndarray:
    """Returns the cone of influence of the given magnetic field data."""
    
    w0: int = 6         # ω0 parameter for Morlet wavelet
    dt: float = 60.0    # Timestep in seconds
    s0: float = 2 * dt  # Starting scale: 2 * 60 s = 2 mins
    dj: float = 1 / 12  # Twelve sub-octaves per octave
    J = int(9 / dj)     # 9 octaves with 12 sub-octaves each
    mother = Morlet(w0) # Mother wavelet function

    # Get the COI and return it
    _, _, _, coi, _, _ = cwt(mag_component, dt, dj, s0, J, mother)
    return coi

def Pc5_power_old(frequency: ndarray, power: ndarray) -> ndarray:
    """
    Args:
        frequency: An ordered 1D array, shape (n_time_points,).
        power: A 2D array, shape (n_scales, n_time_points).
    Returns:
        A 1D array of power in the Pc5 frequency range 2-7 mHz.
    """
    
    # Get indices for values in the Pc5 range.
    Pc5_range: ndarray = where((2e-3 <= frequency) & (frequency <= 7e-3))
    
    # Get the first and the last index.
    first: int = Pc5_range[0][0]
    last: int = Pc5_range[0][-1]

    # Return a 1D array of Pc5 powers for each time point.
    return npsum(power[first:last+1, :], axis=0)