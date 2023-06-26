"""
Contains functions for building deep learning models.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno, Miguel Ángel Gutiérrez-Naranjo. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import ConvLSTM2D, GRU, BatchNormalization, Conv3D, Input, Reshape, UpSampling2D, Concatenate, MaxPooling2D, Conv2D, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import layers




def create_model(input_shape, architecture, num_layers):
    """
    Create and compile a Convolutional LSTM or GRU model based on the selected architecture and number of layers.

    Parameters:
    input_shape (tuple): The shape of the input data (num_frames, width, height, channels).
    architecture (int): The index representing the desired architecture (0 for ConvLSTM2D, 1 for GRU).
    num_layers (int): The number of layers to include in the model.

    Returns:
    model (keras.Model): The compiled Convolutional LSTM or GRU model.
    """

    if architecture == 0:
        model = architecture_conv_lstm(input_shape, num_layers)
    elif architecture == 1:
        # model = architecture_gru(input_shape, num_layers)
        model = architecture_bidirectional_conv_lstm_unet(input_shape, num_layers)
    elif architecture == 3:
        model = architecture_bidirectional_gru(input_shape, num_layers)
    elif architecture == 4:
        model = architecture_conv3d_conv2d(input_shape, num_layers)
    else:
        raise ValueError("Invalid architecture index.")

    # Compile the model
    model.compile(
        loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam()
    )

    return model


def architecture_conv_lstm(input_shape, num_layers):
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

    for _ in range(num_layers):
        x = ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
            dropout=0.3,
            recurrent_dropout=0.3
        )(x)
        x = BatchNormalization()(x)

    x = Conv3D(
        filters=input_shape[-1],
        kernel_size=(3, 3, 3),
        activation="sigmoid",
        padding="same",
    )(x)

    model = Model(inp, x)
    return model

def conv_lstm_block(inputs, filters):
    x = layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding='same', return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    return x

def unet_block(inputs, filters):
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def architecture_bidirectional_conv_lstm_unet(input_shape, num_layers):
    """
    Bi-directional ConvLSTM U-Net Architecture: Multiple Bi-directional ConvLSTM2D layers followed by Conv2D layers.

    Parameters:
    input_shape (tuple): The shape of the input data (num_frames, width, height, channels).
    num_layers (int): The number of layers to include in the model.

    Returns:
    model (keras.Model): The Bi-directional ConvLSTM U-Net model.
    """
    inp = Input(shape=(None, *input_shape))
    x = inp

    # Encoder
    encoder_outputs = []
    for _ in range(num_layers):
        x = conv_lstm_block(x, filters=64)
        encoder_outputs.append(x)
        x = tf.nn.pool(x, window_shape=(1, 2, 2), pooling_type='MAX', padding='SAME', strides=(1, 2, 2))

    # Bi-directional ConvLSTM
    x = conv_lstm_block(x, filters=128)
    x = layers.Bidirectional(layers.ConvLSTM2D(128, kernel_size=(3, 3), padding='same', return_sequences=True))(x)

    # Decoder
    for i in range(num_layers - 1, -1, -1):
        x = layers.Concatenate()([x, encoder_outputs[i]])
        x = conv_lstm_block(x, filters=64)
        x = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(x)

    # Final Conv2D layer
    x = layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inp, x)
    return model

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

