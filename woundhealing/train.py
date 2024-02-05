#!venv/bin/python3

"""
Contains functions for loading the wound dataset.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""


import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, Bidirectional, BatchNormalization,Conv2D, MaxPooling2D, Dense, Flatten, LSTM, TimeDistributed, Input, ConvLSTM2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler

from keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import set_gpu
import utils
import ast

import matplotlib.pyplot as plt



import argparse


import pandas as pd
import numpy as np

from keras import backend as K

def average_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def create_metric(metric_instance, output_index):
    def metric(y_true, y_pred):
        return metric_instance(y_true[:, :, output_index], y_pred[:, :, output_index])
    metric.__name__ = f'{metric_instance.__class__.__name__}_output_{output_index + 1}'
    return metric




def load_data(path="results/synthetic.csv", images_path="data/real_segmentations/", restructure=False):
    df = pd.read_csv(path)

    unique_ids = df['ID'].unique()

    MONOLAYER = [0, 3, 6, 9, 12, 24, 27]
    SPHERES = [0, 3, 6, 9, 12, 15]
    max_time_steps = max(len(MONOLAYER), len(SPHERES))  # Calculate max time steps

    X = []
    y = []
    groups = []

    images = []


    for id in unique_ids:
        id_data = df[df['ID'] == id].copy()
        cell_type_name = id.split("_")[0]
        print("ID", id)
        print(id_data)
        '''
        ID Monolayer_1
            ID  CellType  Time          Area     Perimeter
0  Monolayer_1         0     0  8.406077e+07  15946.419343
1  Monolayer_1         0     3  7.179768e+07  14689.680508
2  Monolayer_1         0     6  6.813636e+07  15347.883318
3  Monolayer_1         0     9  5.550134e+07  16800.007752
4  Monolayer_1         0    12  5.096698e+07  17002.345173
5  Monolayer_1         0    24  2.352333e+07  13657.139548
6  Monolayer_1         0    27  5.367241e+06   6574.288226


ID Sphere_1
          ID  CellType  Time          Area     Perimeter
84  Sphere_1         1     0  1.217541e+08  15780.042189
85  Sphere_1         1     3  9.195739e+07  16362.598251
86  Sphere_1         1     6  8.448796e+07  15013.852029
87  Sphere_1         1     9  2.403001e+07  13237.289645
88  Sphere_1         1    12  4.749137e+07  13168.520567
89  Sphere_1         1    15  7.648313e+06   5456.861163
        '''

        #------------------------------------
        # Load images
        pt =  f'{images_path}{cell_type_name}/{id}/gray/'
        images_loaded = [cv2.resize(cv2.imread(os.path.join(pt, f)), (64, 64)) for f in os.listdir(pt)]
        print("loaded images:", len(images_loaded))
        # x = np.expand_dims(x, axis=0)
        if "Spehere" in id and len(images_loaded) < max_time_steps:
            n_blacks = max_time_steps - len(images_loaded)
            for nb in range(n_blacks):
                black_frame = np.zeros_like(images_loaded[0])
                images_loaded.append(black_frame)
        images.append(images_loaded)
        print("len images:", len(images))
        #------------------------------------

        # id_data['Edge_Pixels'] = id_data['Edge_Pixels'].apply(ast.literal_eval) #TODO in./gen_segments.py change to scalar
        # id_data['Edge_Pixels'] = id_data['Edge_Pixels'].apply(lambda x: x[0])

        id_data['Perimeter'] = id_data['Perimeter'].apply(lambda x: 0 if x < 0 else x)

        #------------------------------------
        # Extract features from segmentation
        # Area, relative wound area, percentage of closure, and healing speed
        area_time_0 = id_data.loc[id_data['Time'] == 0, 'Area'].values[0] 
        area = id_data['Area'] 
        id_data['Relative_Wound_Area'] = area / area_time_0 
        id_data['Percentage_Wound_Closure'] = np.where(id_data['Time'] != 0, ((area_time_0 - area) / area_time_0) * 100, np.nan)
        id_data['Healing_Speed'] = np.where(id_data['Time'] != 0, (area_time_0 - area) / id_data['Time'], np.nan)
        
        id_data['Relative_Wound_Area'].fillna(0, inplace=True)
        id_data['Percentage_Wound_Closure'].fillna(0, inplace=True)
        id_data['Healing_Speed'].fillna(0, inplace=True)


        # Para el tipo de célula con 7 intervalos de tiempo
        if (id_data['CellType'] == 0).all():
            for t in range(len(utils.MONOLAYER)-1):
                print(id_data.loc[id_data['Time'] == utils.MONOLAYER[t], 'Area'])

                current_time = id_data.loc[id_data['Time'] == utils.MONOLAYER[t], 'Time'].values[0]
                time_to_close = id_data.loc[id_data['Time'] == utils.MONOLAYER[-1], 'Time'].values[0]
                relative_change = time_to_close - current_time
                id_data.loc[id_data['Time'] == utils.MONOLAYER[t], 'Time_To_Close'] = relative_change
            id_data.loc[id_data['Time'] == utils.MONOLAYER[-1], 'Time_To_Close'] = 0

        # Para el tipo de célula con 6 intervalos de tiempo
        if (id_data['CellType'] == 1).all():
            for t in range(len(utils.SPHERES)-1):
                print(id_data.loc[id_data['Time'] == utils.SPHERES[t], 'Area'])
                current_time = id_data.loc[id_data['Time'] == utils.SPHERES[t], 'Time'].values[0]
                print("current_area", current_time)
                time_to_close = id_data.loc[id_data['Time'] == utils.SPHERES[-1], 'Time'].values[0]
                print("next_area", time_to_close)
                relative_change = time_to_close - current_time
                print("time to close", relative_change)
                id_data.loc[id_data['Time'] == utils.SPHERES[t], 'Time_To_Close'] = relative_change
            id_data.loc[id_data['Time'] == utils.SPHERES[-1], 'Time_To_Close'] = 0
        #------------------------------------


     

        #------------------------------------
        # resctructure in shape: samples, timesteps, #features: samples, 7, 3
        if restructure:
            pad_length = max_time_steps - len(id_data)

            if pad_length > 0:
                padding = pd.DataFrame({'CellType': [id_data['CellType'].iloc[0]] * pad_length,
                                        'Time': [MONOLAYER[i] if i < len(MONOLAYER) else MONOLAYER[-1] for i in range(len(id_data), len(id_data) + pad_length)],
                                        'Area': [0] * pad_length,
                                        'Perimeter': [0] * pad_length,
                                        'Edge_Pixels':[0] * pad_length,
                                        'H_HSV': [0] * pad_length,
                                        'S_HSV': [0] * pad_length,
                                        'V_HSV': [0] * pad_length,
                                        'Relative_Wound_Area': [0] * pad_length,
                                        'Percentage_Wound_Closure': [0] * pad_length,
                                        'Healing_Speed': [0] * pad_length,
                                        'Time_To_Close': [0] * pad_length,
                                        'Energy': [0] * pad_length,
                                        'Entropy': [0] * pad_length,
                                        # 'Relative_Wound_Perimeter': [0] * pad_length,
                                        # 'Percentage_Wound_Closure_Perimeter': [0] * pad_length,
                                        # 'Healing_Speed_Perimeter': [0] * pad_length
                                        })

                id_data = pd.concat([id_data, padding])
        #------------------------------------


        #------------------------------------
        # Build the X array with cell type, time and area
        # and the y array with  the relative wound are, percentage of closure and healing speed
    
        input_features = ['CellType', 'Time', 'Area', 'Perimeter', 'Edge_Pixels', 'H_HSV', 'S_HSV', 'V_HSV', 'Energy', 'Entropy']
        output_features = ['Time_To_Close', 'Relative_Wound_Area', 'Percentage_Wound_Closure', 'Healing_Speed']
        X.append(id_data[input_features].values)
        y.append(id_data[output_features].values)

        print("input_features", id_data[input_features])
        print("output_features", id_data[output_features])

        #------------------------------------
    
    # Concatenate all data
    groups = None
    X = np.array([np.array(x) for x in X])
    y =  np.array([np.array(y_) for y_ in y])
    images = np.array([np.array(i) for i in images])

    return X, y, groups, images

def preprocess_data(X):
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X



def create_model(image_shape, ninputs=3, noutputs=3, architecture=0, hidden_units=18, activation_function="sigmoid", learning_rate=0.05):
    model = Sequential()
    print("X input shape", ninputs)
    print("images input shape", image_shape)
    print("n outputs", noutputs)

    if architecture==0:
        input_tabular = Input(shape=ninputs)
        dense_layer = Dense(hidden_units, activation=activation_function, kernel_regularizer=l2(0.01))(input_tabular)
        output = Dense(noutputs, activation='linear')(dense_layer)
        model = Model(inputs=input_tabular, outputs=output)
    elif architecture==1:
        print("X input shape", ninputs)
        print("images input shape", image_shape)
        print("n outputs", noutputs)

        input_tabular = Input(shape=(ninputs[0], ninputs[1]))
        input_image = Input(shape=(image_shape[0], image_shape[1], image_shape[2], image_shape[3])) 

        # Tabular input processing
        lstm = LSTM(50, return_sequences=True)(input_tabular)
        lstm = BatchNormalization()(lstm)
        flatten1 = TimeDistributed(Flatten())(lstm)


        # Image input processing
        conv2d1 = Conv2D(64, (3,3), activation='relu', padding='same')(input_image)
        conv2d2 = Conv2D(32, (3,3), activation='relu', padding='same')(conv2d1)
        convlstm = ConvLSTM2D(16, (3,3), activation='relu', padding='same', return_sequences=True)(conv2d2)
        flatten2 = TimeDistributed(Flatten())(convlstm)

        combined = concatenate([flatten1, flatten2])
        output = TimeDistributed(Dense(noutputs, activation='linear'))(combined)

        model = Model(inputs=[input_tabular, input_image], outputs=output)
        # dropout_rate = 0.2
        # model.add(Dropout(dropout_rate))
    elif architecture == 2:
        input_image = Input(shape=(ninputs[0], ninputs[1]))
        conv1 = Conv2D(32, (3, 3), activation='relu')(input_image)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
        pool2 = MaxPooling2D((2, 2))(conv2)
        flat = Flatten()(pool2)
        dense1 = Dense(hidden_units, activation=activation_function)(flat)
        output = Dense(noutputs, activation='linear')(dense1)
        model = Model(inputs=input_image, outputs=output)
    # Architecture with Single LSTM Layer
    elif architecture == 3:
        input_tabular = Input(shape=(ninputs[0], ninputs[1]))
        lstm = LSTM(hidden_units, activation=activation_function)(input_tabular)
        output = Dense(noutputs, activation='linear')(lstm)
        model = Model(inputs=input_tabular, outputs=output)


    optimizer = "adam" # Adam(learning_rate=learning_rate)

    #metrics for every output

    # this order must be the same as METRICS list constant
    mse_metrics = [create_metric(MeanSquaredError(), i) for i in range(noutputs)]
    mae_metrics = [create_metric(MeanAbsoluteError(), i) for i in range(noutputs)]
    rmse_metrics = [create_metric(RootMeanSquaredError(), i) for i in range(noutputs)]

    metrics = mse_metrics + mae_metrics + rmse_metrics

    model.compile(loss=average_mse, optimizer=optimizer, metrics=metrics)

    model.summary()

    return model


METRICS = ['MeanSquaredError', 'MeanAbsoluteError', 'RootMeanSquaredError']  # The names of the metrics (this order is important)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, images_train, images_test, original_shape_y_test, scaler_y, timesteps_test, epochs, architecture):
    # Define the checkpoint path and filename
    checkpoint_path = f"results/train/architecture_{architecture}/best_model.h5"

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss", #"val_loss",
        save_best_only=True,
        mode="min", #"min",
        verbose=1
    )


    if architecture == 1: #LSTM+Conv2D
        history = model.fit([X_train, images_train], y_train, validation_data=([X_test, images_test], y_test), 
                            epochs=epochs, batch_size=16, callbacks=[checkpoint], verbose=1)#, validation_data=(X_test, y_test))
        score = model.evaluate([X_test, images_test], y_test, verbose=1)
    else:
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                            epochs=epochs, batch_size=16, callbacks=[checkpoint], verbose=1)#, validation_data=(X_test, y_test))
        score = model.evaluate(X_test, y_test, verbose=1)

    # Extract metric names and indices from the model's metrics_names
    metric_names = model.metrics_names
    output_indices = [int(name.split('_')[-1]) for name in metric_names if 'output' in name]
    print("output_indices", output_indices)

    # Calculate average metrics for each output
    avg_metrics = {}
    for i, output_index in enumerate(output_indices):
        for j, metric in enumerate(METRICS):
            metric_name = f'{metric}_output_{output_index}'
            metric_index = metric_names.index(metric_name)
            avg_metrics[f'{metric}_{output_index}'] = score[metric_index]
    if architecture == 1: #LSTM+Conv2D
        predictions = model.predict([X_test, images_test])
    else:
        predictions = model.predict(X_test)
    y_test = y_test.reshape((original_shape_y_test[0]*original_shape_y_test[1], original_shape_y_test[2]))
    y_test = scaler_y.inverse_transform(y_test)

    predictions = predictions.reshape((original_shape_y_test[0]*original_shape_y_test[1], original_shape_y_test[2]))
    predictions_original_scale = scaler_y.inverse_transform(predictions)

    print("timesteps_test", timesteps_test.shape, "y_test[:, 0]", y_test[:, 0].shape)
    # Creamos un dataframe de pandas con los datos



    results = pd.DataFrame({
        'Timestep': timesteps_test,
        'True Time_To_Close': y_test[:, 0],
        'Predicted Time_To_Close': predictions_original_scale[:, 0],
        'True - Relative_Wound_Area': y_test[:, 1],
        'Predicted - Relative_Wound_Area': predictions_original_scale[:, 1],
        'True - Percentage_Wound_Closure': y_test[:, 2],
        'Predicted - Percentage_Wound_Closure': predictions_original_scale[:, 2],
        'True - Healing_Speed': y_test[:, 3],
        'Predicted - Healing_Speed': predictions_original_scale[:, 3],
        
        
    })

    results.to_csv(f'results/train/architecture_{architecture}/predictions_vs_true.csv', index=False)


    history_dict = history.history
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))  # Change to whatever size fits best
    N = len(history_dict["loss"])
    plt.plot(np.arange(0, N), history_dict["loss"], label="training_loss")
    plt.plot(np.arange(0, N), history_dict["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), history_dict["accuracy"], label="training_acc")
    # plt.plot(np.arange(0, N), history_dict["val_accuracy"], label="val_acc")
    plt.title("Performance Metrics of Valence Emotion State")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="best", bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.tight_layout()
    plt.savefig(f"results/train/architecture_{architecture}/history_loss_plot.png", dpi=300)

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))  # Change to whatever size fits best
    N = len(history_dict["loss"])
    plt.plot(np.arange(0, N), history_dict["MeanSquaredError_output_1"], label="training_MSE_1")
    plt.plot(np.arange(0, N), history_dict["val_MeanSquaredError_output_1"], label="val_MSE_1")
    plt.plot(np.arange(0, N), history_dict["MeanSquaredError_output_2"], label="training_MSE_2")
    plt.plot(np.arange(0, N), history_dict["val_MeanSquaredError_output_2"], label="val_MSE_2")
    plt.plot(np.arange(0, N), history_dict["MeanSquaredError_output_3"], label="training_MSE_3")
    plt.plot(np.arange(0, N), history_dict["val_MeanSquaredError_output_3"], label="val_MSE_3")
    plt.plot(np.arange(0, N), history_dict["MeanSquaredError_output_4"], label="training_MSE_4")
    plt.plot(np.arange(0, N), history_dict["val_MeanSquaredError_output_4"], label="val_MSE_4")
    # plt.plot(np.arange(0, N), history_dict["accuracy"], label="training_acc")
    # plt.plot(np.arange(0, N), history_dict["val_accuracy"], label="val_acc")
    plt.title("Performance Metrics")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"results/train/architecture_{architecture}/history_MSE_loss_plot.png", dpi=300)

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))  # Change to whatever size fits best
    N = len(history_dict["loss"])
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_1"], label="training_MAE_1")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_1"], label="val_MAE_1")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_2"], label="training_MAE_2")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_2"], label="val_MAE_2")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_3"], label="training_MAE_3")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_3"], label="val_MAE_3")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_4"], label="training_MAE_4")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_4"], label="val_MAE_4")
    # plt.plot(np.arange(0, N), history_dict["accuracy"], label="training_acc")
    # plt.plot(np.arange(0, N), history_dict["val_accuracy"], label="val_acc")
    plt.title("Performance Metrics")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"results/train/architecture_{architecture}/history_MAE_loss_plot.png", dpi=300)

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))  # Change to whatever size fits best
    N = len(history_dict["loss"])
    plt.plot(np.arange(0, N), history_dict["RootMeanSquaredError_output_1"], label="training_RMSE_1")
    plt.plot(np.arange(0, N), history_dict["val_RootMeanSquaredError_output_1"], label="val_RMSE_1")
    plt.plot(np.arange(0, N), history_dict["RootMeanSquaredError_output_2"], label="training_RMSE_2")
    plt.plot(np.arange(0, N), history_dict["val_RootMeanSquaredError_output_2"], label="val_RMSE_2")
    plt.plot(np.arange(0, N), history_dict["RootMeanSquaredError_output_3"], label="training_RMSE_3")
    plt.plot(np.arange(0, N), history_dict["val_RootMeanSquaredError_output_3"], label="val_RMSE_3")
    plt.plot(np.arange(0, N), history_dict["RootMeanSquaredError_output_4"], label="training_RMSE_4")
    plt.plot(np.arange(0, N), history_dict["val_RootMeanSquaredError_output_4"], label="val_RMSE_4")
    # plt.plot(np.arange(0, N), history_dict["accuracy"], label="training_acc")
    # plt.plot(np.arange(0, N), history_dict["val_accuracy"], label="val_acc")
    plt.title("Performance Metrics of Valence Emotion State")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"results/train/architecture_{architecture}/history_RMSE_loss_plot.png", dpi=300)

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))  # Change to whatever size fits best
    N = len(history_dict["loss"])
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_1"], label="training_MAE_1")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_1"], label="val_MAE_1")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_2"], label="training_MAE_2")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_2"], label="val_MAE_2")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_3"], label="training_MAE_3")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_3"], label="val_MAE_3")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_4"], label="training_MAE_4")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_4"], label="val_MAE_4")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_1"], label="training_MAE_1")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_1"], label="val_MAE_1")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_2"], label="training_MAE_2")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_2"], label="val_MAE_2")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_3"], label="training_MAE_3")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_3"], label="val_MAE_3")
    plt.plot(np.arange(0, N), history_dict["MeanAbsoluteError_output_4"], label="training_MAE_4")
    plt.plot(np.arange(0, N), history_dict["val_MeanAbsoluteError_output_4"], label="val_MAE_4")
    plt.plot(np.arange(0, N), history_dict["RootMeanSquaredError_output_1"], label="training_RMSE_1")
    plt.plot(np.arange(0, N), history_dict["val_RootMeanSquaredError_output_1"], label="val_RMSE_1")
    plt.plot(np.arange(0, N), history_dict["RootMeanSquaredError_output_2"], label="training_RMSE_2")
    plt.plot(np.arange(0, N), history_dict["val_RootMeanSquaredError_output_2"], label="val_RMSE_2")
    plt.plot(np.arange(0, N), history_dict["RootMeanSquaredError_output_3"], label="training_RMSE_3")
    plt.plot(np.arange(0, N), history_dict["val_RootMeanSquaredError_output_3"], label="val_RMSE_3")
    plt.plot(np.arange(0, N), history_dict["RootMeanSquaredError_output_4"], label="training_RMSE_4")
    plt.plot(np.arange(0, N), history_dict["val_RootMeanSquaredError_output_4"], label="val_RMSE_4")
    plt.title("Performance Metrics")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"results/train/architecture_{architecture}/history_all_loss_plot.png", dpi=300)


    return avg_metrics




def main():
    '''
    Examples of use:
    
    nohup ./woundhealing/train.py -r -p results/real_segments.csv -a 0 -g 3 > logs/train-arch-0.log &
    '''
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-p', '--path', type=str, default='results/synthetic_segments.csv', help='the segmentation path')
    p.add_argument('-r', '--real', action='store_true', help="Flag to indicate using real; else synthetic")
    p.add_argument('-n', '--n-folds', type=int, default=3, help='the number of folds')
    p.add_argument('-a', '--a', type=int, default=0, help='the architecture: 0-simple cnn; 1-simple LSTM')
    p.add_argument('-e', '--epochs', type=int, default=1000, help='the architecture: 0-simple cnn; 1-simple LSTM')
    p.add_argument('-et', '--eval-type', type=int, default=0, help='0:hold-out; 1:k-fold; 2:-group-kfold')
    p.add_argument('-lr', '--learning-rate', type=float, default=0.05, help='learning rate for optimizer')
    p.add_argument('-af', '--activation-function', type=str, default='sigmoid', choices=['sigmoid', 'tanh'], help='activation function for hidden layers')
    p.add_argument('-g', '--gpu-id', type=int, default=1, help='the GPU device ID')
    args = p.parse_args()

    set_gpu(args.gpu_id)

    architecture =args.a #1:LSTM
    n_folds = args.n_folds
    epochs = args.epochs
    learning_rate = args.learning_rate
    activation_function = args.activation_function

    # Load data
    X, y, groups, images = load_data(args.path, restructure=True)
    print("X shape",X.shape)
    print("y shape", y.shape)
    print("images shape", images.shape)

    # pd.DataFrame(X).to_csv("results/train/X.csv")
    # pd.DataFrame(X).to_csv("results/train/y.csv")


    # Normalize the data
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # Define k-fold cross-validation
    if args.eval_type == 0:
        print("hold-out")
        X = X.reshape(X.shape[0]*X.shape[1], 1, X.shape[2])
        y = y.reshape(y.shape[0]*y.shape[1], 1, y.shape[2])
        images = images.reshape(images.shape[0]*images.shape[1], 1, images.shape[2], images.shape[3], images.shape[4])

        print("X shape",X.shape)
        print("y shape", y.shape)
        print("images shape", images.shape)
        
        
        assert len(X) % 7 == 0
        num_sequences = len(X) // 7 # Calcular el número total de secuencias completas de longitud 7
        print("num_sequences", num_sequences)
        
        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
        # Dividir los datos por tipo celular
        X_type1, X_type2 = X[:(num_sequences//2) * 7], X[(num_sequences//2) * 7:]
        y_type1, y_type2 = y[:(num_sequences//2) * 7], y[(num_sequences//2) * 7:]
        images_type1, images_type2 = images[:(num_sequences//2) * 7], images[(num_sequences//2) * 7:]
        print("X_type1", X_type1.shape)
        print("X_type2", X_type2.shape)



        # Calcular el número de secuencias para cada tipo celular
        num_sequences_type1 = len(X_type1) // 7
        num_sequences_type2 = len(X_type2) // 7
        print("num_sequences_type1", num_sequences_type1)
        print("num_sequences_type2", num_sequences_type2)

        # Calcular el número de secuencias para el conjunto de entrenamiento y prueba
        num_train_sequences_type1 = int(num_sequences_type1 * 0.8)
        num_train_sequences_type2 = int(num_sequences_type2 * 0.8)
        print("num_train_sequences_type1", num_train_sequences_type1)
        print("num_train_sequences_type2", num_train_sequences_type2)
        num_test_sequences_type1 = num_sequences_type1 - num_train_sequences_type1
        num_test_sequences_type2 = num_sequences_type2 - num_train_sequences_type2
        print("num_test_sequences_type1", num_test_sequences_type1)
        print("num_test_sequences_type2", num_test_sequences_type2)

        # Calcular el índice de corte para cada tipo celular
        cutoff_type1_train = num_train_sequences_type1 * 7
        cutoff_type1_test = cutoff_type1_train + num_test_sequences_type1 * 7
        cutoff_type2_train = num_train_sequences_type2 * 7
        cutoff_type2_test = cutoff_type2_train + num_test_sequences_type2 * 7

        # Dividir en conjuntos de entrenamiento y prueba para cada tipo celular
        X_train_type1, X_test_type1 = X_type1[:cutoff_type1_train], X_type1[cutoff_type1_train:cutoff_type1_test]
        y_train_type1, y_test_type1 = y_type1[:cutoff_type1_train], y_type1[cutoff_type1_train:cutoff_type1_test]
        images_train_type1, images_test_type1 = images_type1[:cutoff_type1_train], images_type1[cutoff_type1_train:cutoff_type1_test]

        X_train_type2, X_test_type2 = X_type2[:cutoff_type2_train], X_type2[cutoff_type2_train:cutoff_type2_test]
        y_train_type2, y_test_type2 = y_type2[:cutoff_type2_train], y_type2[cutoff_type2_train:cutoff_type2_test]
        images_train_type2, images_test_type2 = images_type2[:cutoff_type2_train], images_type2[cutoff_type2_train:cutoff_type2_test]

        # Combinar los conjuntos de entrenamiento y prueba de cada tipo celular
        X_train = np.concatenate([X_train_type1, X_train_type2])
        y_train = np.concatenate([y_train_type1, y_train_type2])
        images_train = np.concatenate([images_train_type1, images_train_type2])

        X_test = np.concatenate([X_test_type1, X_test_type2])
        y_test = np.concatenate([y_test_type1, y_test_type2])
        images_test = np.concatenate([images_test_type1, images_test_type2])

        print("X_test_type1", X_test_type1.shape)
        print("X_test_type2", X_test_type2.shape)

        timesteps_test_type1 = []
        print("X_test_type1.shape[0] // num_sequences_type1", X_test_type1.shape[0] // 7)
        for i in range(X_test_type1.shape[0] // 7):
            # print("X_test_type1 ", i)
            for j in range(7):
                timesteps_test_type1.append(utils.MONOLAYER[j] if j<len(utils.MONOLAYER) else utils.MONOLAYER[-1]+(3))
                # print("len timesteps_test_type1", len(timesteps_test_type1))

        timesteps_test_type2 = []
        for i in range(X_test_type2.shape[0] // 7):
            # print("X_test_type2 ", i)
            for j in range(7):
                timesteps_test_type2.append(utils.SPHERES[j] if j<len(utils.SPHERES) else utils.SPHERES[-1]+(3))
                # print("len timesteps_test_type2", len(timesteps_test_type2))


        timesteps_test = np.array(timesteps_test_type1 + timesteps_test_type2)
        print("timesteps_test shape", timesteps_test.shape)



        print("X_train shape", X_train.shape, "y_train shape", y_train.shape, "images_train shape", images_train.shape)
        print("X_test shape", X_test.shape, "y_test shape", y_test.shape, "images_test shape", images_test.shape)

        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------

        images_train_normalized = images_train / 255.0
        images_test_normalized = images_test / 255.0

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        # Save the original shape
        original_shape_X_train = X_train.shape
        original_shape_X_test = X_test.shape
        original_shape_y_train = y_train.shape
        original_shape_y_test = y_test.shape

        # Flatten X_train to 2D, fit the scaler, transform the data, and then reshape it back to the original shape
        X_train = X_train.reshape((original_shape_X_train[0]*original_shape_X_train[1], original_shape_X_train[2]))
        X_train_normalized = scaler.fit_transform(X_train)
        X_train_normalized = X_train_normalized.reshape(original_shape_X_train)

        # Flatten X_test to 2D, use the same scaler that was fit on the training data, 
        # transform the data, and then reshape it back to the original shape
        X_test = X_test.reshape((original_shape_X_test[0]*original_shape_X_test[1], original_shape_X_test[2]))
        X_test_normalized = scaler.transform(X_test)
        X_test_normalized = X_test_normalized.reshape(original_shape_X_test)

        
        # Flatten y_train to 2D, fit the scaler, transform the data, and then reshape it back to the original shape
        y_train = y_train.reshape((original_shape_y_train[0]*original_shape_y_train[1], original_shape_y_train[2]))
        y_train_normalized = scaler_y.fit_transform(y_train)
        y_train_normalized = y_train_normalized.reshape(original_shape_y_train)

        # Flatten y_test to 2D, use the same scaler that was fit on the training data, 
        # transform the data, and then reshape it back to the original shape
        y_test = y_test.reshape((original_shape_y_test[0]*original_shape_y_test[1], original_shape_y_test[2]))
        y_test_normalized = scaler_y.transform(y_test)
        y_test_normalized = y_test_normalized.reshape(original_shape_y_test)


        # Create model
        model = create_model(image_shape=images.shape[1:], ninputs=X.shape[1:], noutputs=y.shape[2], architecture=architecture, hidden_units=18, learning_rate=learning_rate, activation_function=activation_function)

        print("X_train_normalized shape", X_train_normalized.shape, "y_train_normalized shape", y_train_normalized.shape, "images_train_normalized shape", images_train_normalized.shape)
        print("X_test_normalized shape", X_test_normalized.shape, "y_test_normalized shape", y_test_normalized.shape, "images_test_normalized shape", images_test_normalized.shape)
        avg_metrics = train_and_evaluate_model(model, X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, images_train_normalized, images_test_normalized, original_shape_y_test, scaler_y, timesteps_test, epochs, architecture)
        print(f"METRICS: {avg_metrics}")
        results_df = pd.DataFrame(avg_metrics, index=[0]).T.reset_index()
        results_df.columns = ['Metric', 'Value']


    else:
        if args.eval_type == 1:
            cv = KFold(n_splits=n_folds)
            tq = tqdm(cv.split(X, y), total=n_folds, desc='Training...')
        else:
            cv = GroupKFold(n_splits=n_folds) 
            tq = tqdm(cv.split(X, y, groups), total=n_folds, desc='Training...')
        
        # Collect scores from folds
        scores = []
        
        f_i = 0
        for train_index, test_index in tq:
            # Create model
            model = create_model(image_shape=images.shape[1:], ninputs=X.shape[1:], noutputs=y.shape[2], architecture=architecture, hidden_units=18, learning_rate=learning_rate, activation_function=activation_function)

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            images_train, images_test = images[train_index], images[test_index]
            print("X_train shape", X_train.shape, "y_train shape", y_train.shape, "images_train shape", images_train.shape)
            print("X_test shape", X_test.shape, "y_test shape", y_test.shape, "images_test shape", images_test.shape)


            # # Preprocess data
            # X_train = preprocess_data(X_train)
            # X_test = preprocess_data(X_test)

            # print("fold", f_i, "train shapes:", X_train.shape, y_train.shape, "test shapes:", X_test.shape, y_test.shape)
                                               
            avg_metrics = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, images_train, images_test, original_shape_y_test=None, scaler_y=None, timesteps_test=None, epochs=epochs, architecture=architecture)
            print(f"fold-{f_i}. {avg_metrics}")
            scores.append({**{'Fold': len(scores) + 1}, **avg_metrics})
        
        results_df = pd.DataFrame(scores)

        # Calculate average and append to the DataFrame
        avg_metrics = results_df.mean().to_dict()
        avg_metrics['Fold'] = 'Average'
        results_df = pd.concat([results_df, pd.DataFrame(avg_metrics, index=[0])], ignore_index=True)

    
    path = f"results/train/architecture_{architecture}/"
    os.makedirs(path, exist_ok=True)
    path = f"{path}real" if args.real else f"{path}synth"
    results_df.to_csv(f'{path}_train_arch_{architecture}_eval_type_{args.eval_type}.csv', index=False)

    print(f"Average MSE: {avg_metrics}")


if __name__ == "__main__":
    main()
