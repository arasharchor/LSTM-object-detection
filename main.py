import re, sys, numpy as np
from parameters import *

from keras import backend as K
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, SGD

from vgg16_keras import VGG_16
from lstm_keras import LSTMnet
from get_data import get_data
from process_predictions import process_predictions

# sys.setrecursionlimit(1000)

###############################
# Loading synset dictionaries #
###############################

synset_raw = [l.strip() for l in open(synset_path).readlines()]
synset_wnet2id = {}
synset_wnet2name = {}
synset_id2name = {}
regexp = "(n[0-9]*) ([0-9]*) ([a-z]*)"
for s in synset_raw:
    synset_wnet2id[re.match(regexp,s).group(1)] = re.match(regexp,s).group(2)
    synset_wnet2name[re.match(regexp,s).group(1)] = re.match(regexp,s).group(3)
    synset_id2name[re.match(regexp,s).group(2)] = re.match(regexp,s).group(3)
print 'Synset dictionaries loaded'

##################################
# Parameters of the LSTM network #
##################################

# Parameters
print 'Number of categories :', num_categories
print 'Side :', S

# LSTM parameters
nb_lstm_layer = 3
input_size = 512*7*7
hidden_size = 256
time_distributed = True
nb_frame = 16

# Fitting parameters
batch_size = 1
bucket_size = 1000 / batch_size
nb_epoch = 20

#####################################
# Creating the custom loss function #
#####################################

def custom_loss(y_true, y_pred):
	# y_true and y_pred are
	# (nb_frame, S * S * (x,y,w,h,p1, p2, ..., p30, objectness))
    nb_features = 4 + num_categories + 1
    y1 = y_pred
    y2 = y_true
    loss = 0.0
    lambda_coord = 5.0
    lambda_noobj = 0.5

    for i in range(S*S):
        y1_probs = y1[:,:, i*nb_features+4:((i+1)*nb_features - 1)]
        y2_probs = y2[:,:, i*nb_features+4:((i+1)*nb_features - 1)]

        y1_coords = y1[:,:, i*nb_features:i*nb_features + 4]
        y2_coords = y2[:,:, i*nb_features:i*nb_features + 4]

        noobj = ((y2[:,:, ((i+1)*35 - 1)] == 0))

        # Only penalizing classification if an object is present
        loss_probs = K.sum(K.square(y1_probs - y2_probs),axis=2) * y2[:,:, ((i+1)*nb_features - 1)]
        # SSE weighted by lambda_coord, increasing localization loss
        loss_coords = K.sum(K.square(y1_coords - y2_coords),axis=2)
        # SSE weighted by lambda_noobj if no object is present
        lambda_conf = lambda_noobj * noobj + (1-noobj)
        loss_conf = K.square(y1[:,:, ((i+1)*nb_features - 1)] - y2[:,:, ((i+1)*nb_features - 1)])

        loss = loss + K.sum(loss_probs + lambda_coord * loss_coords + lambda_conf * loss_conf, axis = 1)

    loss = K.sum(loss)
    return loss

#############################
# Loading the VGG16 network #
#############################

vgg16_network = VGG_16(vgg16_weights_path)
vgg16_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
vgg16_network.compile(optimizer=vgg16_optimizer, loss='categorical_crossentropy')

############################
# Loading the LSTM network #
############################

# w_path = lstm_weights_directory + 'checkpoint-002.hdf5'
w_path = None

lstm_network = LSTMnet(nb_lstm_layer, nb_frame, input_size, (4 + num_categories + 1) * S * S, hidden_size, time_distributed, w_path)
lstm_optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
lstm_network.compile(loss=custom_loss, optimizer=lstm_optimizer)

############################
# Creating a batch generator
#############################

def batchGenerator():
    while 1:
        (X_train, y_train, _, _, _) = get_data(image_net_directory, batch_size, nb_frame, 'train', synset_wnet2id, bucket_id = '0000', verbose = False)
        X_train = np.reshape(X_train, (-1, 3, 224, 224))
        X_train = vgg16_network.predict(X_train)
        X_train = np.reshape(X_train, (batch_size, nb_frame, 512 * 7 * 7))
        y_train = np.reshape(y_train, (batch_size, nb_frame, S * S * (4 + num_categories + 1)))

        yield X_train, y_train

batchGenerator = batchGenerator()

print 'Batch generator loaded'

#####################
# Fitting the model #
#####################

#checkpoint = ModelCheckpoint(filepath=lstm_weights_directory + 'checkpoint-{epoch:03d}.hdf5') #, save_weights_only = True)
#print 'Initializing the model...'
#lstm_network.fit_generator(batchGenerator, samples_per_epoch = bucket_size, nb_epoch = nb_epoch, verbose = 1, callbacks = [checkpoint])

###########################################
# Testing the model and show some results #
###########################################

(X_train, y_train, image_paths, label_paths, indexes) = get_data(image_net_directory, batch_size, nb_frame, 'val', synset_wnet2id, bucket_id = '0000', verbose = False)
print 'Processing video', image_paths[0], 'from frame', indexes[0]
X_train = np.reshape(X_train, (-1, 3, 224, 224))
X_train = vgg16_network.predict(X_train)
X_train = np.reshape(X_train, (batch_size, nb_frame, 512 * 7 * 7))
y_pred = lstm_network.predict(X_train)
y_train = np.reshape(y_train, (batch_size, nb_frame, -1))

process_predictions(y_pred, y_train, image_paths[0], label_paths[0], indexes[0], nb_frame, synset_id2name, synset_wnet2name, show = True)
print 'Loss',custom_loss(y_train, y_pred).eval()
