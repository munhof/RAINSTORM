"""Reconstructed binary model builders from the original RAINSTORM sources.

The TensorFlow import stays inside these functions so STORM, Studio, and the
Python 3.9 inference worker can import package metadata without loading TF.
"""

def build_simple(input_features=12, learning_rate=1e-4):
    """Original 12-feature Dense(32,16,8,1) binary classifier recipe."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam

    if input_features != 12:
        raise ValueError('The recovered simple recipe expects 12 features')
    model = Sequential([
        Input(shape=(input_features,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_wide(timesteps=7, features=12, *, units=(32, 16, 8), dropout_rate=.2,
               learning_rate=1e-5):
    """Original Conv1D + bidirectional LSTM recipe from model_building.py."""
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (Input, Dense, Bidirectional, LSTM,
        BatchNormalization, Dropout, GlobalMaxPooling1D, Conv1D)
    from tensorflow.keras.optimizers import Adam

    if timesteps != 7 or features != 12:
        raise ValueError('The recovered wide recipe expects (7,12) windows')
    inputs = Input(shape=(timesteps, features), name='input_sequence')
    values = Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu',
                    name='conv1d_motion')(inputs)
    values = BatchNormalization(name='bn_conv')(values)
    values = Dropout(dropout_rate, name='dropout_conv')(values)
    for index, count in enumerate(units):
        values = Bidirectional(LSTM(count, return_sequences=True,
                                    name=f'bilstm_{index}'))(values)
        values = BatchNormalization(name=f'bn_{index}')(values)
        values = Dropout(dropout_rate, name=f'dropout_{index}')(values)
    values = GlobalMaxPooling1D(name='global_max_pooling')(values)
    values = Dense(8, activation='relu', name='dense_8')(values)
    values = Dropout(dropout_rate, name='dropout_dense_8')(values)
    values = Dense(4, activation='relu', name='dense_4')(values)
    values = Dropout(dropout_rate, name='dropout_dense_4')(values)
    output = Dense(1, activation='sigmoid', name='binary_out')(values)
    model = Model(inputs, output, name='CleanedBidirectionalRNN')
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model
