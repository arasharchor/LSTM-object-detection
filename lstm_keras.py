from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed

def LSTMnet(nb_lstm_layer, nb_frame, input_size, output_size, hidden_size, time_distributed = False, weights_path = None):
    first_layer_return_sequences = False
    if nb_lstm_layer == 1:
        first_layer_return_sequences = time_distributed
    else:
        first_layer_return_sequences = True

    model = Sequential()

    # num_lstm_layer LSTM layers
    model.add(LSTM(hidden_size, input_shape = (nb_frame, input_size), return_sequences = first_layer_return_sequences))
    for i in range(nb_lstm_layer-1):
        if i == nb_lstm_layer-2:
            model.add(LSTM(hidden_size, return_sequences = time_distributed))
        else :
            model.add(LSTM(hidden_size, return_sequences = True))

    # Final dense layer
    if time_distributed:
        model.add(TimeDistributed(Dense(4096, activation='relu')))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(output_size, activation='linear')))
    else:
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_size, activation='linear'))

    if weights_path:
        model.load_weights(weights_path)

    print 'LSTM Model loaded'

    return model
