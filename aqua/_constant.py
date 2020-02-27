import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso
from xgboost import XGBRegressor

random_seed = 7
np.random.seed(random_seed)

# metrics -------------------
variables_description = {
    "Height": "Participant's height",
    "Weight": "Participant's weight",
    "ADD": "Adduction",
    "ABD": "Abduction",
    "ER": "External rotation",
    "IR": "Internal rotation",
    "EXT": "Extension",
    "FLEX": "Flexion",
}

targets_description = {
    "EB max force": "Max. force during egg-beater",
    "EB mean force": "Mean force during egg-beater",
    "EB sd force": "Force standard deviation during egg-beater",
    "EB max height": "Max. height during egg-beater",
    "EB min height": "Min. height during egg-beater",
    "EB mean height": "Mean height during egg-beater",
    "BB": "Body-boost",
}

available_targets = list(targets_description.keys())
default_targets = ["BB", "EB mean height", "EB mean force"]

normalization_strategies = ["None", "Weight", "Weight x Height", "IMC"]
forces_order = ["ADD", "ABD", "ER", "IR", "EXT", "FLEX"]

# models --------------------
models_functions = {
    "Random Forest": RandomForestRegressor,
    "XGBoost": XGBRegressor,
    "Linear Regression": LinearRegression,
    "Lasso Regression": Lasso,
    "Gradient Tree Boosting": GradientBoostingRegressor,
    "Histogram Gradient Boosting": HistGradientBoostingRegressor,
}
available_models = list(models_functions.keys())
default_models = ["Random Forest", "Linear Regression"]
