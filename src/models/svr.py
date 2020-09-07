# =============================================================================
# 
# =============================================================================


# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)

from scipy.stats import  loguniform
from sklearn.model_selection import  RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from src.utils.help_func import results_searchcv,plot_predictions,\
    errors_distribution, plot_visualizations, get_model_data
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import seaborn as sns
sns.set()
import time
import numpy as np


# Get the data
df_train_val = get_model_data(40000)
seed=42

# Feature selection
features = [
    'Tr',
    'inf_pow_1',
    'inf_pow_2',
    'mort_pow_1',
    'mort_pow_2',
    'mort_pow_3',
    'n_closed',
    'react_time',
    'total_deceased',
    'betweenness',
    'degree',
    'closeness',
    'country_pop',
    'country_departures',
    'exposed_pop',
    'inf_pow_1_log',
    'inf_pow_2_log',
    'mort_pow_1_log',
    'mort_pow_2_log',
    'mort_pow_3_log',
    ]


df_train_val = df_train_val[features]

print("=" * 20)
print(f"Train_validation size: {df_train_val.shape}")
print("=" * 20)


X_train_val = df_train_val.drop('total_deceased', axis=1)
y_train_val = df_train_val['total_deceased']
y_train_val = y_train_val.replace(0, np.finfo(float).eps)
X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                  y_train_val,
                                                  random_state=seed)

# Path naming
samples = df_train_val.shape[0]
features = df_train_val.shape[1]
run_time = time.strftime("run_%d_%m_%Y-%H_%M_%S")
MODEL_NAME = 'support_vector_regessor'
# Path to save the model
PATH = f"{root_project}/models/{MODEL_NAME}-{samples}-samples-{features}-feat-{run_time}"
LOAD_PATH = f"{root_project}/models/{MODEL_NAME}.pkl"
MODEL_PATH = f"{PATH}/{MODEL_NAME}.pkl"
RESULTS_PATH = f"{PATH}/results.txt"
if not os.path.exists(PATH):
    os.makedirs(PATH)
    

pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('preprocess', StandardScaler()),
    ('estimator', SVR())
])


pipe = TransformedTargetRegressor(pipe, func=np.log, inverse_func=np.exp )



# param_dist = dict(
#     regressor__imputer__strategy=['median', 'mean'],
#     regressor__estimator__kernel = ['rbf'],
#     regressor__estimator__C= loguniform(10, 2000),
#     regressor__estimator__gamma= loguniform(1e-8, 1e-1)
# )

# Zoom param_dist
param_dist = dict(
    regressor__imputer__strategy=['median'],
    regressor__estimator__kernel=['rbf'],
    regressor__estimator__C=loguniform(10, 200),
    regressor__estimator__gamma=loguniform(1e-4, 1e-1),
    regressor__estimator__epsilon=loguniform(1e-3, 1)
)



scoring = {'R2': 'r2', 'RMSE': 'neg_root_mean_squared_error',
           'MAE': 'neg_mean_absolute_error'}

random_search = RandomizedSearchCV(pipe, param_distributions=param_dist,
                                   scoring=scoring,
                                   refit='R2',                                                   
                                   verbose=1, n_iter=50, cv=3,
                                   random_state=seed, n_jobs=-1)


random_search.fit(X_train_val, y_train_val)
joblib.dump(random_search, MODEL_PATH)

# Load a model
# random_search = joblib.load(LOAD_PATH)

results_searchcv(random_search, RESULTS_PATH)

pipe_plot = clone(pipe) # prevents yellowbrics from change pipe
plot_visualizations(PATH, pipe_plot, X_train_val,
                    y_train_val, X_train, y_train, X_val, y_val,
                    featureimportance=False )

# Train the pipe with only train data and best parameters of random search
pipe.set_params(**random_search.best_params_)
pipe.fit(X_train, y_train)


plot_predictions(pipe, X_val, y_val, samples=50)

errors_distribution(pipe, X_val, y_val, df_train_val, n=100)


