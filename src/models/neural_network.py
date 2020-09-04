# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]
sys.path.append(root_project)


import time
from src.utils.help_func import plot_predictions, plot_visualizations,\
    errors_distribution, get_model_data
from keras import backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from keras import backend as K
import csv
import seaborn as sns
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
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

# Path naming for saving tensorboard output and checkpoints
root_logdir_tensorboard = f"{root_project}/models/my_logs"
root_logdir_checkpoints = f"{root_project}/models/checkpoints"


# Train a combination of different number of layers - number of nodes
# dense_layers = [8, 6, 4]
# layer_sizes = [80, 60, 40, 20]

# Train a unique neural network
dense_layers = [7]
layer_sizes = [80]

revision = '14'

# Uncomment following lines to train
# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         time.sleep(10)
#         run_time = time.strftime("run_%d_%m_%Y-%H_%M_%S")
#         NAME = f"{dense_layer}-layers-{layer_size}-nodes-{size_data}k-samples-{num_features}-feat-rev{revision}-{run_time}"
#         print(NAME)
#         model = Sequential()
#         inputs = Input(X_train_scaled.shape[1:])

#         for l in range(dense_layer - 1):
#             model.add(
#                 Dense(
#                     layer_size,
#                     activation='selu',
#                     kernel_initializer='lecun_normal'))

#         model.add(Dense(1))

#         tensorboard_cb = keras.callbacks.TensorBoard(
#             f"{root_logdir_tensorboard}/{NAME}")
#         early_stopping_cb = keras.callbacks.EarlyStopping(
#             patience=10, restore_best_weights=True)
#         checkpoint_cb = keras.callbacks.ModelCheckpoint(
#             filepath=f"{root_logdir_checkpoints}/{NAME}", save_best_only=True, verbose=1)

#         model.compile(
#             optimizer=Nadam(),
#             loss='mean_squared_error',
#             metrics=[
#                 'mean_absolute_error',
#                 'mean_absolute_percentage_error',
#                 coeff_determination])

#         history = model.fit(
#             X_train_scaled,
#             y_train,
#             epochs=500,
#             verbose=0,
#             validation_data=(
#                 X_val_scaled,
#                 y_val),
#             callbacks=[
#                 tensorboard_cb,
#                 early_stopping_cb,
#                 checkpoint_cb])

#         score_train = model.evaluate(X_train_scaled, y_train, verbose=0)
#         score_val = model.evaluate(X_val_scaled, y_val, verbose=0)

#         print('Train score:', score_train[0])
#         print('Train MAE:', score_train[1])
#         print('Train MAPE:', score_train[2])
#         print('Train R2:', score_train[3])
#         print('Test score:', score_val[0])
#         print('Test MAE:', score_val[1])
#         print('Test MAPE:', score_val[2])
#         print('Train R2:', score_val[3])
#         print('========')


# model.save(f"{NAME}.h5")

# Load a model in LOAD PATH
NAME = '7-layers-80-nodes-377k-samples-19-feat-rev14-run_03_09_2020-23_56_30.h5'
LOAD_PATH = f"{root_project}/models/{NAME}"
model = load_model(LOAD_PATH,
                   custom_objects={'coeff_determination': coeff_determination})

score_train = model.evaluate(X_train_scaled, y_train, verbose=0)
score_val = model.evaluate(X_val_scaled, y_val, verbose=0)

print('Train score:', score_train[0])
print('Train MAE:', score_train[1])
print('Train MAPE:', score_train[2])
print('Train R2:', score_train[3])
print('Val score:', score_val[0])
print('Val MAE:', score_val[1])
print('Val MAPE:', score_val[2])
print('Val R2:', score_val[3])
print('========')


errors_distribution(model, X_val_scaled, y_val, df_train_val, n=1000)


# Score in test set
df_test = pd.read_pickle(
    f"{root_project}/data/processed/test_set.pickle")

df_test = df_test[features]

X_test = df_test.drop('total_deceased', axis=1)
y_test = df_test['total_deceased']

X_test_scaled = pipe.transform(X_test.astype(np.float64))

score_test = model.evaluate(X_test_scaled, y_test, verbose=0)

plot_predictions(model, X_test_scaled, y_test, samples=50)


print('Test score:', score_test[0])
print('Test MAE:', score_test[1])
print('Test MAPE:', score_test[2])
print('Test R2:', score_test[3])
print('========')
