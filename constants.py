"""Constants for the SMR prediction script."""

# CDAWeb .cdf folders path.
CDAWEB_PATH: str = ""
# Project folder.
FOLDER: str = f"{CDAWEB_PATH}/"

# SuperMAG data paths and filename.
SMAG_PATH: str = f"{FOLDER}/"
PATH: str = f"{FOLDER}/"

# Target variables.
TARGETS: list[str] = ["SMR", "SMR00", "SMR06", "SMR12", "SMR18"]

# Mission features, assuming both missions.
MISSION_FEATURES: list[str] = [
    "Magnitude", "BGSM_z", "SC_pos_GSM_Re_y", "V_GSM_z", "SC_pos_GSM_Re_z", 
    "T", "Tpr", "BGSM_x", "V_GSM_y", "SC_pos_GSM_Re_x", "Np", "V_GSM_x", 
]
FEATURE_LAGS: list[str] = [
    "SMR_lag1", "SMR_lag2", "SMR_lag3", "SMR_lag4", 
    "SMR00_lag1", "SMR00_lag2", "SMR00_lag3", "SMR00_lag4", 
    "SMR06_lag1", "SMR06_lag2", "SMR06_lag3", "SMR06_lag4", 
    "SMR12_lag1", "SMR12_lag2", "SMR12_lag3", "SMR12_lag4", 
    "SMR18_lag1", "SMR18_lag2", "SMR18_lag3", "SMR18_lag4",  
    "Magnitude_lag1", "Magnitude_lag2", "Magnitude_lag3", "Magnitude_lag4", 
    "T_lag1", "T_lag2", "T_lag3", "T_lag4", 
    "BGSM_z_lag1", "BGSM_z_lag2", "BGSM_z_lag3", "BGSM_z_lag4", 
    "V_GSM_x_lag1", "V_GSM_x_lag2", "V_GSM_x_lag3", "V_GSM_x_lag4",  
    "SC_pos_GSM_Re_x_lag1", "SC_pos_GSM_Re_x_lag2", "SC_pos_GSM_Re_x_lag3", "SC_pos_GSM_Re_x_lag4", 
    "SC_pos_GSM_Re_y_lag1", "SC_pos_GSM_Re_y_lag2", "SC_pos_GSM_Re_y_lag3", "SC_pos_GSM_Re_y_lag4", 
    "SC_pos_GSM_Re_z_lag1", "SC_pos_GSM_Re_z_lag2", "SC_pos_GSM_Re_z_lag3", "SC_pos_GSM_Re_z_lag4",
]
TIME_FEATURES: list[str] = ["unix_time"]
FEATURES: list[str] = MISSION_FEATURES + FEATURE_LAGS + TIME_FEATURES

# XGBoost parameters.
N_ESTIMATORS: dict[str, int] = {
    "SMR": 750,
    "SMR00": 1250,
    "SMR06": 1250,
    "SMR12": 1000,
    "SMR18": 1000
}
EARLY_STOPPING_ROUNDS: dict[str, int] = {
    "SMR": 150,
    "SMR00": 400,
    "SMR06": 400,
    "SMR12": 200,
    "SMR18": 200,
}
XGB_METRIC: str = "rmse"
SMR_PARAMS: dict = {
    "max_depth": 4,
    "min_child_weight": 12,
    "learning_rate": 0.05,
    "lambda": 20.0,          
    "alpha": 1.0,               
    "gamma": 0.5,      
    "subsample": 0.8,        
    "colsample_bytree": 0.8,
    "seed": 1,
    "device": "cuda", 
    "sampling_method": "gradient_based",
}
SMR00_PARAMS = {
    "max_depth": 5,
    "min_child_weight": 11,          
    "learning_rate": 0.02,   
    "lambda": 35.0,          
    "alpha": 6.0,   
    "gamma": 3.0,        
    "subsample": 0.75,        
    "colsample_bytree": 0.75,
    "seed": 1,    
    "device": "cuda",
    "sampling_method": "gradient_based",
}
SMR06_PARAMS = {
    "max_depth": 5,
    "min_child_weight": 11,          
    "learning_rate": 0.02,   
    "lambda": 30.0,          
    "alpha": 5.0,              
    "gamma": 3.0,            
    "subsample": 0.75,        
    "colsample_bytree": 0.75,
    "seed": 1,
    "device": "cuda",
    "sampling_method": "gradient_based",
}
SMR12_PARAMS = {
    "max_depth": 4,
    "min_child_weight": 9,
    "learning_rate": 0.025,
    "lambda": 25.0,
    "alpha": 3.0,
    "gamma": 3.0,
    "subsample": 0.75,        
    "colsample_bytree": 0.75,
    "seed": 1,
    "device": "cuda", 
    "sampling_method": "gradient_based",
}
SMR18_PARAMS = {
    "max_depth": 3,          
    "min_child_weight": 10,
    "learning_rate": 0.03,   
    "lambda": 20.0,
    "alpha": 2.0,
    "gamma": 1.0,
    "subsample": 0.7,        
    "colsample_bytree": 0.7,
    "seed": 1,
    "device": "cuda", 
    "sampling_method": "gradient_based",
}
XGB_PARAMS: dict[str, dict] = {
    "SMR": SMR_PARAMS,
    "SMR06": SMR06_PARAMS,
    "SMR12": SMR12_PARAMS,
    "SMR18": SMR18_PARAMS,
    "SMR00": SMR00_PARAMS,
}

# SuperMAG data fill value.
FILL: int = 999999

# CDAWeb datasets' (keys) and parameters (values). Tpr is thermal speed in km/s.
CDAWEB_PARAMS: dict[str, list[str]] = {
    'ace_mfi': ["Magnitude", "BGSM"],                # AC_H0_MFI
    'ace_swe': ["Np", "Tpr", "V_GSM", "SC_pos_GSM"], # AC_H0_SWE
    'wind_mfi': ["Magnitude", "BGSM", "SC_pos_GSM"], # WI_H0_MFI
    'wind_swe': ["V_GSM", "Np", "Tpr"],              # WI_K0_SWE
}

# Lagged features.
CDAWEB_LAGS: dict[str, list[str]] = {
    "both": [
        "V_GSM_x", "BGSM_z", "Magnitude", "Np", 
        "SC_pos_GSM_Re_x", "SC_pos_GSM_Re_y", "SC_pos_GSM_Re_z"
    ],
    "ace": ["ace_swe_V_GSM_x", "ace_mfi_BGSM_z", "ace_mfi_Magnitude", "ace_swe_Np"],
    "wind": ["wind_swe_V_GSM_x", "wind_mfi_BGSM_z", "wind_mfi_Magnitude", "wind_swe_Np"]
}

# Earth radius (km).
RE: int = 6371
# Boltzmann constant (J/K).
K: float = 1.380649e-23
# Proton mass (kg).
MP: float = 1.6726219e-27

# Features for plotting.
PLOTTING_FEATURES: dict = {
    'BGSM Components': ["BGSM_x", "BGSM_y", "BGSM_z"],
    'Solar Wind Parameters': ["Magnitude", "Np", "T"],
    'Velocity Components': ["V_GSM_x", "V_GSM_y", "V_GSM_z"]
}

# Physical limits for each measurement.
CDAWEB_LIMITS: dict[str, tuple] = {
    "Magnitude": (0, 100),             # nT
    "BGSM_x": (-100, 100),             # nT
    "BGSM_y": (-100, 100),             # nT
    "BGSM_z": (-100, 100),             # nT
    "Np": (0, 100),                    # cm^-3
    "T": (1e3, 1e8),                   # K
    "V_GSM_x": (-1500, 1500),          # km/s
    "V_GSM_y": (-1000, 1000),          # km/s
    "V_GSM_z": (-1000, 1000),          # km/s
    "SC_pos_GSM_Re_x": (200, 300),     # Re
    "SC_pos_GSM_Re_y": (-100, 100),    # Re
    "SC_pos_GSM_Re_z": (-100, 100),    # Re
}

# ArgParser description.
DESCRIPTION: str = "Process/load data or train models or predict with loaded models."
