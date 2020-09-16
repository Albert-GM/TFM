# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)


from src.utils.help_func import get_model_data, results_estimator
from keras import backend as K
from kerastuner.tuners import RandomSearch
from kerastuner import Objective
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import  Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import numpy as np
from tensorflow import keras
import pandas as pd
sns.set()



def coeff_determination(y_true, y_pred):
    """
    Implements the coefficient of determination to be used in a Keras model.

    Parameters
    ----------
    y_true : np.array
        Ground truth.
    y_pred : np.array
        Predictions.

    Returns
    -------
    float
        Coefficient of determination.

    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# print(device_lib.list_local_devices())

# Get the data
df_train_val = get_model_data()
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
X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                  y_train_val,
                                                  random_state=42)

size_data = int(len(X_train) / 1000)
num_features = len(X_train.columns)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


X_train_scaled = pipe.fit_transform(X_train.astype(np.float64))
X_val_scaled = pipe.transform(X_val.astype(np.float64))

root_logdir_tensorboard = f"{root_project}/models/tests/my_logs"
root_logdir_checkpoints = f"{root_project}/models/tests/checkpoints"

def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=X_train_scaled.shape[1:]))
    units=hp.Int('units_layer', min_value=10, max_value=100, step=5)
        
    for i in range(hp.Int('num_layers', 3, 12)):
        model.add(Dense(units=units,
                               activation='selu',
                               kernel_initializer='lecun_normal'))
    model.add(Dense(1))
    model.compile(
        optimizer=Nadam(),
        loss='mean_squared_error',
            metrics=[
                'mean_absolute_error',
                'mean_absolute_percentage_error',
                coeff_determination])
    return model


tuner = RandomSearch(
    build_model,
    objective=Objective('val_coeff_determination', direction='max'),
    max_trials=20,
    executions_per_trial=1,
    directory=f"{root_project}/models/tests/neural_networks",
    project_name="tfm")

tuner.search_space_summary()

tensorboard_cb = keras.callbacks.TensorBoard(root_logdir_tensorboard)
early_stopping_cb = keras.callbacks.EarlyStopping(
                            patience=5, restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=root_logdir_checkpoints, save_best_only=True, verbose=1)

# Uncomment netxt lines to train the models
# tuner.search(X_train_scaled, y_train,
#              epochs=100,
#              validation_data=(X_val_scaled, y_val),
#              callbacks=[tensorboard_cb,
#                         early_stopping_cb,
#                         checkpoint_cb])


# tuner.results_summary()


# Reload best model from project
# tuner.reload()
# estimator = tuner.get_best_models()[0]

# Reload best model from .h5
estimator = load_model(f"{root_project}/models/neural_network.h5",
                       custom_objects={'coeff_determination': coeff_determination})



# Score in validation set
results_estimator(estimator, X_val_scaled, y_val)

# Score in test set
df_test = pd.read_pickle(
    f"{root_project}/data/processed/test_set.pickle")
df_test = df_test[features]
X_test = df_test.drop('total_deceased', axis=1)
y_test = df_test['total_deceased']

X_test_scaled = pipe.transform(X_test.astype(np.float64))

results_estimator(estimator, X_test_scaled, y_test)