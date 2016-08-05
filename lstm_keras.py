from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, BatchNormalization

def LSTMnet(nb_lstm_layer, nb_frame, input_size, output_size, hidden_size, time_distributed = False, weights_path = None, dropout = 0.0, batchnorm = False, gpu = False):
    first_layer_return_sequences = False
    device = 'cpu'
    if nb_lstm_layer == 1:
        first_layer_return_sequences = time_distributed
    else:
        first_layer_return_sequences = True

    if gpu:
        device = 'gpu'

    model = Sequential()

    # num_lstm_layer LSTM layers
    model.add(LSTM(hidden_size[0], input_shape = (nb_frame, input_size), return_sequences = first_layer_return_sequences, consume_less = device))
    for i in range(nb_lstm_layer-1):
        if i == nb_lstm_layer-2:
            model.add(LSTM(hidden_size[i], return_sequences = time_distributed, consume_less = device))
        else :
            model.add(LSTM(hidden_size[i], return_sequences = True, consume_less = device))

    if batchnorm:
        # Per frame normalization
        model.add(BatchNormalization(mode = 0, axis = 1))

    # Final dense layers
    model.add(Dropout(dropout))
    if time_distributed:
        model.add(TimeDistributed(Dense(4096, activation='relu')))
        model.add(Dropout(dropout))
        model.add(TimeDistributed(Dense(output_size, activation='linear')))
    else:
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(output_size, activation='linear'))

    if weights_path:
        model.load_weights(weights_path)

    print 'LSTM Model loaded'

    return model
