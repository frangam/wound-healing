#!venv/bin/python3


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError

from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import set_gpu

import argparse


import pandas as pd
import numpy as np

from keras import backend as K

def average_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def create_metric(metric_instance, output_index):
    def metric(y_true, y_pred):
        return metric_instance(y_true[:, output_index], y_pred[:, output_index])
    metric.__name__ = f'{metric_instance.__class__.__name__}_output_{output_index + 1}'
    return metric




def load_data(path="data/synthetic.csv"):
    df = pd.read_csv(path)

    unique_ids = df['ID'].unique()

    MONOLAYER = [0, 3, 6, 9, 12, 24, 27]
    SPHERES = [0, 3, 6, 9, 12, 15]

    # Calculate real times based on CellType
    df.loc[df['CellType'] == 0, 'Time'] = df.loc[df['CellType'] == 0, 'Time'].map(lambda x: MONOLAYER[x])
    df.loc[df['CellType'] == 1, 'Time'] = df.loc[df['CellType'] == 1, 'Time'].map(lambda x: SPHERES[x])


    X = []
    y = []
    groups = []

    for id in unique_ids:
        id_data = df[df['ID'] == id].copy()
       
        area_time_0 = id_data.loc[id_data['Time'] == 0, 'Area'].values[0]

        id_data['Relative_Wound_Area'] = id_data['Area'] / area_time_0
        id_data['Percentage_Wound_Closure'] = np.where(id_data['Time'] != 0, ((area_time_0 - id_data['Area']) / area_time_0) * 100, np.nan)
        id_data['Healing_Speed'] = np.where(id_data['Time'] != 0, (area_time_0 - id_data['Area']) / id_data['Time'], np.nan)
        
        id_data['Relative_Wound_Area'].fillna(0, inplace=True)
        id_data['Percentage_Wound_Closure'].fillna(0, inplace=True)
        id_data['Healing_Speed'].fillna(0, inplace=True)

        X.append(id_data[['CellType', 'Time', 'Area']])
        y.append(id_data[['Relative_Wound_Area', 'Percentage_Wound_Closure', 'Healing_Speed']])
        groups.extend([id] * len(id_data))

    X = pd.concat(X)
    y = pd.concat(y)
    groups = pd.Series(groups, name='ID')

    df_processed = pd.concat([X, y, groups], axis=1)

    df_processed.to_csv('processed_data.csv', index=False)

    return np.array(X), np.array(y), np.array(groups)

def preprocess_data(X):
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X



def create_model(ninputs=3, noutputs=3, architecture=0, hidden_units=18, activation_function="sigmoid", learning_rate=0.05):
    model = Sequential()

    if architecture==0:
        model.add(Dense(hidden_units, input_dim=ninputs, activation=activation_function, kernel_regularizer=l2(0.01)))
    elif architecture==1:
        model.add(LSTM(50, activation='relu', input_shape=(1, ninputs)))

    model.add(Dense(noutputs, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate)

    #metrics for every output

    # this order must be the same as METRICS list constant
    mse_metrics = [create_metric(MeanSquaredError(), i) for i in range(noutputs)]
    mae_metrics = [create_metric(MeanAbsoluteError(), i) for i in range(noutputs)]
    rmse_metrics = [create_metric(RootMeanSquaredError(), i) for i in range(noutputs)]

    metrics = mse_metrics + mae_metrics + rmse_metrics

    model.compile(loss=average_mse, optimizer=optimizer, metrics=metrics)

    return model


N_OUTPUTS = 3  # The number of output neurons
METRICS = ['MeanSquaredError', 'MeanAbsoluteError', 'RootMeanSquaredError']  # The names of the metrics (this order is important)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs):
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=1)#, validation_data=(X_test, y_test))
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

    return avg_metrics




def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n-folds', type=int, default=3, help='the number of folds')
    p.add_argument('--a', type=int, default=0, help='the architecture: 0-simple cnn; 1-simple LSTM')
    p.add_argument('--epochs', type=int, default=1000, help='the architecture: 0-simple cnn; 1-simple LSTM')
    p.add_argument('--eval-type', type=int, default=0, help='0:k-fold; 1-group-kfold')
    p.add_argument('--learning-rate', type=float, default=0.05, help='learning rate for optimizer')
    p.add_argument('--activation-function', type=str, default='sigmoid', choices=['sigmoid', 'tanh'], help='activation function for hidden layers')
    p.add_argument('--gpu-id', type=int, default=1, help='the GPU device ID')
    args = p.parse_args()

    set_gpu(args.gpu_id)

    architecture =args.a #1:LSTM
    n_folds = args.n_folds
    epochs = args.epochs
    learning_rate = args.learning_rate
    activation_function = args.activation_function

    # Load data
    X, y, groups = load_data()

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define k-fold cross-validation
    if args.eval_type == 0:
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
        model = create_model(architecture=architecture, hidden_units=18, learning_rate=learning_rate, activation_function=activation_function)


        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if architecture == 1:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Preprocess data
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)

        print("fold", f_i, "train shapes:", X_train.shape, y_train.shape, "test shapes:", X_test.shape, y_test.shape)

        avg_metrics = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs)
        print(f"fold-{f_i}. {avg_metrics}")
        scores.append({**{'Fold': len(scores) + 1}, **avg_metrics})

    results_df = pd.DataFrame(scores)

    # Calculate average and append to the DataFrame
    avg_metrics = results_df.mean().to_dict()
    avg_metrics['Fold'] = 'Average'
    results_df = pd.concat([results_df, pd.DataFrame(avg_metrics, index=[0])], ignore_index=True)

    results_df.to_csv(f'results_eval_type_{args.eval_type}.csv', index=False)

    print(f"Average MSE: {avg_metrics}")


if __name__ == "__main__":
    main()
