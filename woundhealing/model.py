"""
Contains functions for building deep learning models.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""

import numpy as np
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import ConvLSTM2D, GRU, BatchNormalization, Conv3D, Input, Reshape, UpSampling2D, Concatenate, MaxPooling2D, Conv2D, TimeDistributed, Lambda, MaxPooling3D, Add, LSTM, Conv2DTranspose, Flatten, Dense, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from keras.activations import softmax

from skimage.metrics import structural_similarity as ssim

import keras.backend as K

def scale_weights(output, scale):
    return output * K.constant(scale)

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def psnr(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 / tf.math.log(10.0) * (tf.math.log(max_pixel ** 2 / K.mean(K.square(y_pred - y_true), axis=-1)))

def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def create_model(input_shape, architecture, num_layers=3):
    """
    Create and compile a Convolutional LSTM or GRU model based on the selected architecture and number of layers.

    Parameters:
    input_shape (tuple): The shape of the input data (num_frames, width, height, channels).
    architecture (int): The index representing the desired architecture (0 for ConvLSTM2D, 1 for GRU).
    num_layers (int): The number of layers to include in the model.

    Returns:
    model (keras.Model): The compiled Convolutional LSTM or GRU model.
    """

    if architecture == 0: #ConvLSTM-1
        model = architecture_conv_lstm(input_shape[2:], num_layers=1) # architecture_conv_lstm(input_shape[2:], num_layers)
    elif architecture == 1: #ConvLSTM-3
        model = architecture_conv_lstm(input_shape[2:], num_layers=3)
    elif architecture == 2: #ConvLSTM-6
        model = architecture_conv_lstm(input_shape[2:], num_layers=6)
    elif architecture == 3: #ConvLSTM-2
        model = architecture_conv_lstm(input_shape[2:], num_layers=2) 
    elif architecture == 4: #CasConvLSTM
        model = arch_2(input_shape[2:]) 
    elif architecture == 5: #ConvLSTM2D-Conv2D-3D
        model = arch_3(input_shape[2:]) 
    elif architecture == 6: #MultiPathConvLSTM
        model = arch_5(input_shape[2:]) 
    elif architecture == 7: #ResConvLSTM-2x64
        model = residual_conv_lstm(input_shape[2:], num_layers=2, num_filters=64, dropout_rate_list=[0.3,0.3])
    elif architecture == 8: #ResConvLSTM-3x64
        model = residual_conv_lstm(input_shape[2:], num_layers=3, num_filters=64, dropout_rate_list=[0.3,0.3,0.3])
    elif architecture == 9: #ResConvLSTM-6x64
        model = residual_conv_lstm(input_shape[2:], num_layers=6, num_filters=64, dropout_rate_list=[0.3,0.3,0.3,0.3,0.3,0.3])
    elif architecture == 10: #ResConvLSTM-9x64
        model = residual_conv_lstm(input_shape[2:], num_layers=9, num_filters=64, dropout_rate_list=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3])
    elif architecture == 11: #ResConvLSTM-3x32
        model = residual_conv_lstm(input_shape[2:], num_layers=3, num_filters=32, dropout_rate_list=[0.1,0.1,0.3])
    elif architecture == 12: #ResConvLSTM-3x128
        model = residual_conv_lstm(input_shape[2:], num_layers=3, num_filters=128, dropout_rate_list=[0.1,0.1,0.1])
    elif architecture == 13: #ResConvLSTM-4x64
        model = residual_conv_lstm(input_shape[2:], num_layers=4, num_filters=64, dropout_rate_list=[0.3,0.3,0.3,0.3])
    elif architecture == 14: #ResConvLSTM-4x128
        model = residual_conv_lstm(input_shape[2:], num_layers=4, num_filters=128, dropout_rate_list=[0.3,0.3,0.3,0.3])
        # model = residual_conv_lstm_par(input_shape[2:], num_layers=4, num_filters_list=[64,64,64,64], dropout_rate=0.3)
    elif architecture == 15: #ResConvLSTM-3x256
        model = residual_conv_lstm(input_shape[2:], num_layers=3, num_filters=256, dropout_rate_list=[0.3,0.3,0.3])
    elif architecture == 16: #ConvLSTMEncDec-3x64
        model = architecture_fcn_2p_lstm(input_shape[2:], c=64)
    elif architecture == 17: #ConvLSTM-2T
        model = arch_6(input_shape[2:])
    elif architecture == 18: #TD-Dense-LSTM
        model = arch_7(input_shape[2:])
    elif architecture == 19: #Transformer
        model = arch_transformer(input_shape[2:])
    elif architecture == 20: #Transformer-4
        model = arch_transformer_2(input_shape[2:])
    elif architecture == 21: #ConvLSTM-8
        model = architecture_conv_lstm(input_shape[2:], num_layers=8)
    elif architecture == 22: #ConvLSTM-10
        model = architecture_conv_lstm(input_shape[2:], num_layers=10)
    elif architecture == 23: #ConvLSTM-12
        model = architecture_conv_lstm(input_shape[2:], num_layers=12)
    elif architecture == 24: #ResConvLSTM-4x256
        model = residual_conv_lstm(input_shape[2:], num_layers=4, num_filters=256, dropout_rate_list=[0.3,0.3,0.3,0.3])
    elif architecture == 25: #ResConvLSTM-4x512
        model = residual_conv_lstm(input_shape[2:], num_layers=4, num_filters=512, dropout_rate_list=[0.3,0.3,0.3,0.3])
    elif architecture == 30: # Conv2D-1x32
        model = arch_Conv2D(input_shape[2:], n_layers=1, filters=32)
    elif architecture == 31: # Conv2D-2x32
        model = arch_Conv2D(input_shape[2:], n_layers=1, filters=32)
    elif architecture == 32: #LSTM-1x32
        model = arch_LSTM(input_shape[2:], 1, 32)
    elif architecture == 33: #LSTM-2x32
        model = arch_LSTM(input_shape[2:], 2, 32)
    elif architecture == 34: #LSTM-4x32
        model = arch_LSTM(input_shape[2:], 4, 32)
    elif architecture == 40: #ConvLSTMEncDec-3x32
        model = architecture_fcn_2p_lstm(input_shape[2:], c=32)
    elif architecture == 41: #ConvLSTMEncDec-3x128
        model = architecture_fcn_2p_lstm(input_shape[2:], c=128)
    elif architecture == 42: #ConvLSTMEncDec-3x256
        model = architecture_fcn_2p_lstm(input_shape[2:], c=256)
    elif architecture == 43: # ConvLSTMEncDec-2x64
        model = configurable_architecture(input_shape[2:], c=64, num_convlstm_layers=3)
    elif architecture == 44: # ConvLSTMEncDec-2x128
        model = configurable_architecture(input_shape[2:], c=128, num_convlstm_layers=3)
    elif architecture == 45: # ConvLSTMEncDec-2x256
        model = configurable_architecture(input_shape[2:], c=256, num_convlstm_layers=3)
    elif architecture == 50: #
        model = res_convlstm_enc_dec(input_shape[2:], c=64, num_convlstm_layers=3)
    elif architecture == 60:
        model = architecture_fcn_2p_lstm_4(input_shape[2:], c=64)
    else:
        raise ValueError("Invalid architecture index.")

    model.summary()
    
    # Compile the model
    model.compile(
        loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=[mse, psnr, ssim]
    )

    return model

def configurable_architecture(input_shape, c=64, num_convlstm_layers=3):
    input_image = Input(shape=(None, *input_shape), name="input")
    
    # Encoder
    x = input_image
    for i in range(num_convlstm_layers):
        x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c1 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)

    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(c1)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c2 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)

    # Decoder
    x = TimeDistributed(UpSampling2D((2, 2)))(c2)
    x = Concatenate()([c1, x])
    x = TimeDistributed(Conv2D(c, (3, 3), padding="same", activation="relu"))(x)
    x = TimeDistributed(Conv2D(3, (3, 3), padding="same", activation="relu"))(x)

    output = TimeDistributed(Conv2D(input_shape[-1], (3, 3), padding="same", activation="sigmoid", name="output"))(x)
    output = Lambda(scale_weights, arguments={'scale': 1.0})(output)

    model = Model(inputs=input_image, outputs=[output])
    
    return model



def architecture_fcn_2p_lstm(input_shape, c=64):
    input_image = Input(shape=(None, *input_shape), name="input")

    # ------- Encoder starts here -------
    # First ConvLSTM layer

    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(input_image)
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c1 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    
    # Second ConvLSTM layer
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(c1)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c2 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    
    # Third ConvLSTM layer
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(c2)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c3 = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    # ------- Encoder ends here -------

    # ------- Decoder starts here -------
    # First Upsampling layer
    x = TimeDistributed(UpSampling2D((2, 2)))(c3)
    x = Concatenate()([c2, x])
    x = TimeDistributed(Conv2D(c, (3, 3), padding="same", activation="relu"))(x)

    # Second Upsampling layer
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = Concatenate()([c1, x])
    x = TimeDistributed(Conv2D(3, (3, 3), padding="same", activation="relu"))(x)
    # ------- Decoder ends here -------

    output = TimeDistributed(Conv2D(input_shape[-1], (3, 3), padding="same", activation="sigmoid", name="output"))(x)
    output = Lambda(scale_weights, arguments={'scale': 1.0})(output)

    model = Model(inputs=input_image, outputs=[output])
    
    return model

def architecture_fcn_2p_lstm_4(input_shape, c=64):
    input_image = Input(shape=(None, *input_shape), name="input")

    # ------- Encoder starts here -------
    # First ConvLSTM layer

    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(input_image)
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c1 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    
    # Second ConvLSTM layer
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(c1)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c2 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    
    # Third ConvLSTM layer
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(c2)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c3 = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)

    # Fourth ConvLSTM layer
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(c3)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c4 = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)


    # Fifth ConvLSTM layer
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(c4)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c5 = ConvLSTM2D(filters=2*c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    # ------- Encoder ends here -------

    # ------- Decoder starts here -------
    # First Upsampling layer
    x = TimeDistributed(UpSampling2D((2, 2)))(c5)
    x = Concatenate()([c4, x])
    x = TimeDistributed(Conv2D(c, (3, 3), padding="same", activation="relu"))(x)

    # Second Upsampling layer
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = Concatenate()([c3, x])
    x = TimeDistributed(Conv2D(3, (3, 3), padding="same", activation="relu"))(x)

    # Third Upsampling layer
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = Concatenate()([c2, x])
    x = TimeDistributed(Conv2D(3, (3, 3), padding="same", activation="relu"))(x)

     # Fourth Upsampling layer
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = Concatenate()([c1, x])
    x = TimeDistributed(Conv2D(3, (3, 3), padding="same", activation="relu"))(x)
    # ------- Decoder ends here -------

    output = TimeDistributed(Conv2D(input_shape[-1], (3, 3), padding="same", activation="sigmoid", name="output"))(x)
    output = Lambda(scale_weights, arguments={'scale': 1.0})(output)

    model = Model(inputs=input_image, outputs=[output])
    
    return model

def arch_0(input_shape):
    '''Arquitectura básica Conv3DLSTM'''
    input_layer = Input(shape=(None, *input_shape))
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3, 3), padding='same', return_sequences=True)(input_layer)
    x = BatchNormalization()(x)
    x = Conv3D(filters=input_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)
    x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

def arch_1(input_shape):
    '''Arquitectura con capas adicionales de reducción dimensional'''
    input_layer = Input(shape=(None, *input_shape))
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(input_layer)
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = Conv3D(filters=input_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)
    x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

def arch_2(input_shape):
    '''Arquitectura de red en cascada'''
    input_layer = Input(shape=(None, *input_shape))
    x1 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(input_layer)
    x2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(x1)
    x = Concatenate(axis=-1)([x1, x2])
    x = Conv3D(filters=input_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)
    x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

def arch_3(input_shape):
    '''Arquitectura con capas ConvLSTM2D intercaladas con capas Conv2D'''
    input_layer = Input(shape=(None, *input_shape))
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = Conv3D(filters=input_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)
    x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

# def arch_4(input_shape):
#     '''Arquitectura con residuos (Residual Network)'''
#     input_layer = Input(shape=(None, *input_shape))
#     x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(input_layer)
#     x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
#     res = Add()([input_layer, x])
#     x = Conv3D(filters=input_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(res)
#     x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

#     model = Model(inputs=input_layer, outputs=x)

#     return model


def arch_5(input_shape):
    '''Arquitectura de múltiples caminos (Multi-Path Network)'''
    input_layer = Input(shape=(None, *input_shape))
    x1 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(input_layer)
    x2 = ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', return_sequences=True)(input_layer)
    x = Concatenate(axis=-1)([x1, x2])
    x = Conv3D(filters=input_shape[-1], kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)
    x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

    model = Model(inputs=input_layer, outputs=x)

    return model

def arch_6(input_shape):
    '''Arquitectura con Conv2DTranspose en lugar de Conv3D'''

    '''ConvLSTM2D con Conv2DTranspose: En lugar de utilizar capas Conv3D 
    para la salida, usar capas Conv2DTranspose para incrementar 
    la dimensión espacial de la salida, que puede ser útil 
    si las capas intermedias reducen la dimensión espacial.'''
    input_layer = Input(shape=(None, *input_shape))
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(input_layer)
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = TimeDistributed(Conv2DTranspose(filters=input_shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))(x)
    x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

def arch_7(input_shape):
    '''Arquitectura con TimeDistributed(Dense) y LSTM al final'''

    '''TimeDistributed(Dense()): probar con un enfoque diferente y utilizar 
    una red completamente conectada en lugar de una convolucional 
    para procesar cada frame del video por separado, 
    luego combinar la información temporal al final con capas LSTM.'''
    input_layer = Input(shape=(None, *input_shape))
    x = TimeDistributed(Flatten())(input_layer)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = LSTM(32, return_sequences=True)(x)
    x = TimeDistributed(Dense(input_shape[0]*input_shape[1]*input_shape[2], activation='sigmoid'))(x)
    x = TimeDistributed(Reshape((input_shape[0], input_shape[1], input_shape[2])))(x)
    x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

def arch_Conv2D(input_shape, n_layers=2, filters=32):
    input_layer = Input(shape=(None, *input_shape))
    x = input_layer
    for _ in range(n_layers):
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(input_shape[0]*input_shape[1]*input_shape[2], activation='sigmoid'))(x)
    output_layer = TimeDistributed(Reshape((input_shape[0], input_shape[1], input_shape[2])))(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def arch_LSTM(input_shape, n_layers=2, filters=32):
    input_layer = Input(shape=(None, *input_shape))
    x = TimeDistributed(Flatten())(input_layer)
    for _ in range(n_layers-1):
        x = LSTM(filters, return_sequences=True)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = TimeDistributed(Dense(input_shape[0]*input_shape[1]*input_shape[2], activation='sigmoid'))(x)
    x = TimeDistributed(Reshape((input_shape[0], input_shape[1], input_shape[2])))(x)
    output_layer = Lambda(scale_weights, arguments={'scale': 1.0})(x)


    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def residual_conv_lstm(input_shape, num_layers=3, num_filters=64, dropout_rate_list=[0.3,0.3,0.3], activation="relu"):
    assert len(dropout_rate_list) == num_layers, "Debe proporcionar un valor para cada capa"

    inp = Input(shape=(None, *input_shape))

    # Primer bloque ConvLSTM
    x = ConvLSTM2D(
        filters=num_filters,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation=activation,
        dropout=dropout_rate_list[0],
        recurrent_dropout=dropout_rate_list[0], 
        name="conv_lstm2d_1"
    )(inp)

    for i in range(1, num_layers):
        # Bloques residuales
        x_residual = x

        x = ConvLSTM2D(
            filters=num_filters,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation=activation,
            dropout=dropout_rate_list[i],
            recurrent_dropout=dropout_rate_list[i], 
            name=f"conv_lstm2d_{i+1}"
        )(x)

        # # Adaptar las dimensiones de los canales para la conexión residual
        # x_residual = Conv2D(filters=num_filters_list[i], kernel_size=(1, 1), padding='same', activation='relu')(x_residual)


        # Conexión residual
        x = Add()([x, x_residual])

    # Capa de salida
    x = Conv3D(
        filters=input_shape[-1],
        kernel_size=(3, 3, 3),
        activation="sigmoid",
        padding="same",
    )(x)
    x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

    model = Model(inp, x)
    return model

def residual_conv_lstm_par(input_shape, num_layers, num_filters_list, dropout_rate=0.3, activation="relu"):
    assert len(num_filters_list) == num_layers, "Debe proporcionar un valor para cada capa"

    inp = Input(shape=(None, *input_shape))

    # Primer bloque ConvLSTM
    x = ConvLSTM2D(
        filters=num_filters_list[0],
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation=activation,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name="conv_lstm2d_1"
    )(inp)

    for i in range(1, num_layers):
        # Bloques residuales
        x_residual = x
        x = ConvLSTM2D(
            filters=num_filters_list[i],
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation=activation,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name=f"conv_lstm2d_{i+1}"
        )(x)

        # Conexión residual solo en ciertos bloques
        if i % 2 == 0:
            x = Add()([x, x_residual])

    # Capa de salida
    x = Conv3D(
        filters=input_shape[-1],
        kernel_size=(3, 3, 3),
        activation="sigmoid",
        padding="same"
    )(x)
    x = Lambda(scale_weights, arguments={'scale': 1.0})(x)

    model = Model(inp, x)
    return model


def architecture_conv_lstm(input_shape, num_layers, scaling_factor=1.0):
    """
    Combined Architecture: Multiple ConvLSTM2D layers followed by Conv3D layers.

    Parameters:
    input_shape (tuple): The shape of the input data (num_frames, width, height, channels).
    num_layers (int): The number of layers to include in the model.

    Returns:
    model (keras.Model): The combined Convolutional LSTM model.
    """
    inp = Input(shape=(None, *input_shape))
    x = inp

    for i in range(num_layers):
        x = ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
            # dropout=0.3,
            # recurrent_dropout=0.3, 
            name=f"conv_lstm2d_{i+1}"
        )(x)
        # x = BatchNormalization()(x)

    x = Conv3D(
        filters=input_shape[-1],
        kernel_size=(3, 3, 3),
        activation="sigmoid",
        padding="same",
    )(x)

    # Aplicar weight scaling a la última capa
    x = Lambda(scale_weights, arguments={'scale': scaling_factor})(x)

    model = Model(inp, x)
    return model

# def conv_lstm_block(inputs, filters):
#     x = layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding='same', return_sequences=True)(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
#     x = layers.BatchNormalization()(x)
#     return x

# def unet_block(inputs, filters):
#     x = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     return x

# def architecture_bidirectional_conv_lstm_unet(input_shape, num_layers):
#     """
#     Bi-directional ConvLSTM U-Net Architecture: Multiple Bi-directional ConvLSTM2D layers followed by Conv2D layers.

#     Parameters:
#     input_shape (tuple): The shape of the input data (num_frames, width, height, channels).
#     num_layers (int): The number of layers to include in the model.

#     Returns:
#     model (keras.Model): The Bi-directional ConvLSTM U-Net model.
#     """
#     inp = Input(shape=(None, *input_shape))
#     x = inp

#     # Encoder
#     encoder_outputs = []
#     for _ in range(num_layers):
#         x = conv_lstm_block(x, filters=64)
#         encoder_outputs.append(x)
#         x = tf.nn.pool(x, window_shape=(1, 2, 2), pooling_type='MAX', padding='SAME', strides=(1, 2, 2))

#     # Bi-directional ConvLSTM
#     x = conv_lstm_block(x, filters=128)
#     x = layers.Bidirectional(layers.ConvLSTM2D(128, kernel_size=(3, 3), padding='same', return_sequences=True))(x)

#     # Decoder
#     for i in range(num_layers - 1, -1, -1):
#         x = layers.Concatenate()([x, encoder_outputs[i]])
#         x = conv_lstm_block(x, filters=64)
#         x = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(x)

#     # Final Conv2D layer
#     x = layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

#     model = Model(inp, x)
#     return model

# def architecture_gru(input_shape, num_layers):
#     """
#     GRU Architecture: Multiple GRU layers followed by Conv3D layers.

#     Parameters:
#     input_shape (tuple): The shape of the input data (num_frames, width, height, channels).
#     num_layers (int): The number of layers to include in the model.

#     Returns:
#     model (keras.Model): The GRU model.
#     """
#     # Reshape input data
#     inp =  Input(shape=(None, *input_shape))
#     x = Reshape((input_shape[0], input_shape[1]* input_shape[2]))(inp)

#     for _ in range(num_layers):
#         x = GRU(
#             units=64,
#             return_sequences=True,
#             activation="relu"
#         )(x)
#         x = BatchNormalization()(x)

#     x = Conv3D(
#         filters=input_shape[-1],
#         kernel_size=(3, 3, 3),
#         activation="sigmoid",
#         padding="same",
#     )(x)

#     model = Model(inp, x)
#     return model



# def architecture_bidirectional_gru(input_shape, num_layers):
#     """
#     Bidirectional GRU Architecture: Multiple Bidirectional GRU layers followed by Conv2D layers.

#     Parameters:
#     input_shape (tuple): The shape of the input data (num_frames, width, height, channels).
#     num_layers (int): The number of layers to include in the model.

#     Returns:
#     model (keras.Model): The Bidirectional GRU model.
#     """
#     inp = Input(shape=(None, *input_shape))
#     x = inp

#     for _ in range(num_layers):
#         x = layers.Bidirectional(layers.GRU(64, return_sequences=True, activation='relu'))(x)
#         x = layers.BatchNormalization()(x)

#     x = layers.Conv2D(
#         filters=input_shape[-1],
#         kernel_size=(3, 3),
#         activation='sigmoid',
#         padding='same',
#     )(x)

#     model = Model(inp, x)
#     return model


def architecture_conv3d_conv2d(input_shape, num_layers):
    """
    Conv3D-Conv2D Architecture: Multiple Conv3D and Conv2D layers.

    Parameters:
    input_shape (tuple): The shape of the input data (num_frames, width, height, channels).
    num_layers (int): The number of layers to include in the model.

    Returns:
    model (keras.Model): The Conv3D-Conv2D model.
    """
    inp = Input(shape=(None, *input_shape))
    x = inp

    for _ in range(num_layers):
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=input_shape[-1],
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same',
    )(x)

    model = Model(inp, x)
    return model





import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Capa de atención multi-cabeza
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, ff_dim, rate=0.1, embed_dim=65536, name="transformer_block"):
        super(TransformerBlock, self).__init__(name=name)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        # Additional layer to adjust the shape of out1
        self.reshape_out1 = layers.TimeDistributed(layers.Dense(embed_dim))

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        out1 = self.reshape_out1(out1)  # Adjust the shape of out1
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Print tensor shapes
        print(f"out1 shape: {out1.shape}")
        print(f"ffn_output shape: {ffn_output.shape}")

        return self.layernorm2(out1 + ffn_output)






def arch_transformer(input_shape):
    embed_dim = 64  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    # Input layer
    inputs = layers.Input(shape=(None, *input_shape))

    # Use a Conv2D layer to reduce the dimension of the image
    x = layers.TimeDistributed(layers.Conv2D(embed_dim, kernel_size=(3, 3), strides=(2, 2), padding='same'))(inputs)
    x = layers.Reshape((-1, np.prod(x.shape[2:])))(x)

    # Transformer block
    transformer_block = TransformerBlock(num_heads, ff_dim, 0.1, embed_dim)
    x = transformer_block(x)

    # Decoding layers to reconstruct the image
    x = layers.Dense(16*16*64, activation='relu')(x)  # adjust these dimensions as needed
    x = layers.Reshape((-1, 16, 16, 64))(x)

    x = layers.TimeDistributed(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid'))(x)

    model = keras.Model(inputs=inputs, outputs=x)

    return model


def arch_transformer_2(input_shape):
    embed_dim = 64  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 128  # Hidden layer size in feed forward network inside transformer
    num_blocks = 4  # Number of transformer blocks

    # Input layer
    inputs = layers.Input(shape=(None, *input_shape))

    # Use a Conv2D layer to reduce the dimension of the image
    x = layers.TimeDistributed(layers.Conv2D(embed_dim, kernel_size=(3, 3), strides=(2, 2), padding='same'))(inputs)
    x = layers.Reshape((-1, np.prod(x.shape[2:])))(x)

    # Transformer blocks
    for i in range(num_blocks):
        transformer_block = TransformerBlock(num_heads, ff_dim, 0.1, embed_dim, f"transformer_block_{i+1}")
        x = transformer_block(x)

    # Decoding layers to reconstruct the image
    x = layers.Dense(16*16*64, activation='relu')(x)  # adjust these dimensions as needed
    x = layers.Reshape((-1, 16, 16, 64))(x)

    x = layers.TimeDistributed(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid'))(x)

    model = keras.Model(inputs=inputs, outputs=x)

    return model




def res_convlstm_enc_dec(input_shape, c=64, num_convlstm_layers=3):
    input_image = Input(shape=(None, *input_shape), name="input")
    
    # Encoder
    x = input_image
    for i in range(num_convlstm_layers - 1):
        x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c1 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)

    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(c1)
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    c2 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)

    # Decoder
    x = TimeDistributed(UpSampling2D((2, 2)))(c2)
    x = Concatenate()([c1, x])
    x = TimeDistributed(Conv2D(c, (3, 3), padding="same", activation="relu"))(x)
    x = TimeDistributed(Conv2D(3, (3, 3), padding="same", activation="relu"))(x)

    output = TimeDistributed(Conv2D(input_shape[-1], (3, 3), padding="same", activation="sigmoid", name="output"))(x)
    output = Lambda(scale_weights, arguments={'scale': 1.0})(output)

    model = Model(inputs=input_image, outputs=[output])
    
    return model

