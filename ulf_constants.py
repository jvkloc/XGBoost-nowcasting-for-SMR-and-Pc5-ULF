"""Constants for ULF prediction."""

from typing import Callable


# Project folder.
FOLDER: str = ""

# SuperMAG .netcdf files path.
SUPERMAG_DATA_PATH: str = f"{FOLDER}/"

# SuperMAG .netcdf columns for loading data.
COLUMNS: list[str] = [
    "index", "station", "mlt", "dbn_nez", "dbe_nez", "dbz_nez",
    "sza", "decl", "mcolat", "glon", "glat", 
]

# Sectors for computing Pc5 power.
SECTORS: list[tuple[str, Callable]] = [
    ("Night", lambda mlt: (mlt >= 21) | (mlt < 3)),
    ("Dawn", lambda mlt: (3 <= mlt) & (mlt < 9)),
    ("Day", lambda mlt: (9 <= mlt) & (mlt < 15)),
    ("Dusk", lambda mlt: (15 <= mlt) & (mlt < 21))
]

# Features.
CDAWEB_FEATURES: list[str] = [
    "V_GSM_x", "V_GSM_y", "V_GSM_z", "BGSM_x", "BGSM_y", "BGSM_z",
    "SC_pos_GSM_Re_x", "SC_pos_GSM_Re_y", "SC_pos_GSM_Re_z", 
    "Tpr", "Magnitude", "Np", "L1_delay", "T",
]
CDAWEB_LAGS: list[str] = [
    "Np_lag1", "Np_lag2", "Np_lag3", "Np_lag4", 
    "BGSM_z_lag1", "BGSM_z_lag2", "BGSM_z_lag3", "BGSM_z_lag4", 
    "V_GSM_x_lag1", "V_GSM_x_lag2", "V_GSM_x_lag3", "V_GSM_x_lag4", 
    "Magnitude_lag1", "Magnitude_lag2", "Magnitude_lag3", "Magnitude_lag4",
    "SC_pos_GSM_Re_x_lag1", "SC_pos_GSM_Re_x_lag2", "SC_pos_GSM_Re_x_lag3", "SC_pos_GSM_Re_x_lag4", 
    "SC_pos_GSM_Re_y_lag1", "SC_pos_GSM_Re_y_lag2", "SC_pos_GSM_Re_y_lag3", "SC_pos_GSM_Re_y_lag4", 
    "SC_pos_GSM_Re_z_lag1", "SC_pos_GSM_Re_z_lag2", "SC_pos_GSM_Re_z_lag3", "SC_pos_GSM_Re_z_lag4",
]
SUPERMAG_FEATURES: list[str] = [
    "mlt", "glon", "glat", "mcolat", "sza", "decl", 
    "dbn_nez", "dbe_nez", "dbz_nez", "bh", "Pc5", 
    # "bh" and "Pc5" are derived from SuperMAG data.
]
LAGGED_FEATURES: list[str] = [
    "Pc5_lag1", "Pc5_lag2", "Pc5_lag3", "Pc5_lag4",
    "Pc5_global_lag1", "Pc5_global_lag2", "Pc5_global_lag3", "Pc5_global_lag4",
    "Pc5_Dawn_lag1", "Pc5_Dawn_lag2", "Pc5_Dawn_lag3", "Pc5_Dawn_lag4", 
    "Pc5_Day_lag1", "Pc5_Day_lag2", "Pc5_Day_lag3", "Pc5_Day_lag4", 
    "Pc5_Dusk_lag1", "Pc5_Dusk_lag2", "Pc5_Dusk_lag3", "Pc5_Dusk_lag4", 
    "Pc5_Night_lag1", "Pc5_Night_lag2", "Pc5_Night_lag3", "Pc5_Night_lag4",
]
TIME_FEATURES: list[str] = ["unix_time"]
FEATURES: list[str] = CDAWEB_FEATURES + SUPERMAG_FEATURES + LAGGED_FEATURES + TIME_FEATURES

# Model targets.
TARGETS: list[str] = ["Pc5_global", "Pc5_Night", "Pc5_Dawn", "Pc5_Day", "Pc5_Dusk"]

# XGBoost model parameters.
N_ESTIMATORS: dict[str, int] = {
    "Pc5_global": 925,
    "Pc5_Dawn": 1625,
    "Pc5_Day": 1025,
    "Pc5_Dusk": 1525,
    "Pc5_Night": 2025
}
EARLY_STOPPING_ROUNDS: dict[str, int] = {
    "Pc5_global": 110,
    "Pc5_Dawn": 150,
    "Pc5_Day": 170,
    "Pc5_Dusk": 205,
    "Pc5_Night": 120,
}
XGB_METRIC: str = "rmse"
GLOBAL_PARAMS: dict = {
    "max_depth": 9,
    "min_child_weight": 9,
    "learning_rate": 0.011945770138329289,
    "lambda": 1.1878352965990269,
    "alpha": 6.3684745445729005,
    "gamma": 19.94093641161669,
    "subsample": 0.5199906338821163,
    "colsample_bytree": 0.9655404382692282,
    "seed": 1,
    "device": "cuda",
    "sampling_method": "gradient_based",
}
DAWN_PARAMS: dict = {
    "max_depth": 9,
    "min_child_weight": 4,
    "learning_rate": 0.01629303655517379,
    "lambda": 3.322898211963578,
    "alpha": 1.8628972928343903,
    "gamma": 19.971801849093325,
    "subsample": 0.7105490716193089,
    "colsample_bytree": 0.9489847518512788,
    "seed": 1,
    "device": "cuda",
    "sampling_method": "gradient_based",
}
DAY_PARAMS: dict = {
    "max_depth": 9,
    "min_child_weight": 5,
    "learning_rate": 0.062301631007632814,
    "lambda": 1.484498583825845,
    "alpha": 3.307554786485117,
    "gamma": 8.149539102414185,
    "subsample": 0.7112300813997279,
    "colsample_bytree": 0.816387569418687,
    "seed": 1,
    "device": "cuda",
    "sampling_method": "gradient_based",
}
DUSK_PARAMS: dict = {
    "max_depth": 9,
    "min_child_weight": 10,
    "learning_rate": 0.02000653986039571,
    "lambda": 29.693522685083053,
    "alpha": 1.6997559485982108,
    "gamma": 3.8083425712458836,
    "subsample": 0.9128661900326286,
    "colsample_bytree": 0.6118598683759804,
    "seed": 1,
    "device": "cuda",
    "sampling_method": "gradient_based",
}
NIGHT_PARAMS: dict = {
    "max_depth": 9,
    "min_child_weight": 11,
    "learning_rate": 0.023614166614077944,
    "lambda": 7.082222730206318,
    "alpha": 11.300728225565793,
    "gamma": 13.807312496670251,
    "subsample": 0.5458598494712519,
    "colsample_bytree": 0.7758763721680384,
    "seed": 1,
    "device": "cuda",
    "sampling_method": "gradient_based",
}
XGB_PARAMS: dict[str, dict] = {
    "Pc5_global": GLOBAL_PARAMS,
    "Pc5_Dawn": DAWN_PARAMS,
    "Pc5_Day": DAY_PARAMS,
    "Pc5_Dusk": DUSK_PARAMS,
    "Pc5_Night": NIGHT_PARAMS,
}

# Lagged features. Notice the added "Pc5" for adding lagged Pc5 values.
CDAWEB_LAGS: dict[str, list[str]] = {
    "both": [
        "V_GSM_x", "BGSM_z", "Magnitude", "Np", "Pc5",
        "SC_pos_GSM_Re_x", "SC_pos_GSM_Re_y", "SC_pos_GSM_Re_z"
    ],
    "ace": ["ace_swe_V_GSM_x", "ace_mfi_BGSM_z", "ace_mfi_Magnitude", "ace_swe_Np"],
    "wind": ["wind_swe_V_GSM_x", "wind_mfi_BGSM_z", "wind_mfi_Magnitude", "wind_swe_Np"]
}

# ArgParser description.
DESCRIPTION: str = "Download/check data and/or train a model or predict with a loaded model."
