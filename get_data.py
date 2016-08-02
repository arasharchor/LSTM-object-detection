import cv2, os, re, random
import numpy as np
from ILSVRC_parsing import parse_ILSVRCXML
from process_objects import process_objects
from parameters import *

def get_x_frame(path, frame_id):
    image_path = path + frame_id + '.JPEG'

    im = cv2.resize(cv2.imread(image_path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    return im

def get_y_frame(path, frame_id, synset_wnet2id):
    label_path = path + frame_id + '.xml'
    folder, filename, database, width, height, objects = parse_ILSVRCXML(label_path)
    y = process_objects(int(width), int(height), objects, synset_wnet2id)
    return y

def get_x(path, snippet_id, nb_frame, type):
    image_path = path + 'ILSVRC2015_' + type + '_' + snippet_id + '/'
    regexp = "([0-9]*).JPEG"
    x = np.zeros((nb_frame, 3, 224, 224))
    count_file = 0
    for file in sorted(os.listdir(image_path)):
        if count_file < nb_frame:
            frame_id = re.match(regexp,file).group(1)
            x[count_file,:,:,:] = get_x_frame(image_path, frame_id)
        count_file += 1

    return x, image_path

# YOLO method
def get_y(path, snippet_id, nb_frame, type, synset_wnet2id):
    label_path = path + 'ILSVRC2015_' + type + '_' + snippet_id + '/'
    regexp = "([0-9]*).xml"
    y = np.zeros((nb_frame, S, S, 4+num_categories+1))
    count_file = 0
    for file in os.listdir(label_path):
        if count_file < nb_frame:
            frame_id = re.match(regexp,file).group(1)
            y[count_file, :, :, :] = get_y_frame(label_path, frame_id, synset_wnet2id)
            count_file += 1

    return y, label_path

# def get_y(path, snippet_id, nb_frame, type, synset_wnet2id):
#     label_path = path + 'ILSVRC2015_' + type + '_' + snippet_id + '/'
#     regexp = "([0-9]*).xml"
#     y = np.zeros((nb_frame, num_categories))
#     count_file = 0
#     for file in sorted(os.listdir(label_path)):
#         if count_file < nb_frame:
#             frame_id = re.match(regexp,file).group(1)
#             y[count_file, :] = get_y_frame(label_path, frame_id, synset_wnet2id)
#             count_file += 1

#     return y

def get_data(image_net_directory, batch_size, nb_frame, type, synset_wnet2id, bucket_id = None, verbose = False):
    if type == 'train':
        path_x = image_net_directory + 'Data/VID/' + type + '/ILSVRC2015_VID_' + type + '_' + bucket_id + '/'
        path_y = image_net_directory + 'Annotations/VID/' + type + '/ILSVRC2015_VID_' + type + '_' + bucket_id + '/'

    elif type == 'val':
        path_x = image_net_directory + 'Data/VID/' + type + '/'
        path_y = image_net_directory + 'Annotations/VID/' + type + '/'

    elif type == 'test':
        path_x = image_net_directory + 'Data/VID/' + type + '/'

    batch = random.sample(os.listdir(path_x), batch_size)
    X = np.zeros((batch_size, nb_frame, 3, 224, 224))
    Y = np.zeros((batch_size, nb_frame, S, S, 4+num_categories+1))

    image_paths = []
    label_paths = []

    regexp = "ILSVRC2015_"+ type +"_([0-9]*)"
    count_dir = 0
    for dir in batch:
        snippet_id = re.match(regexp,dir).group(1)
        if verbose:
            print dir, str(count_dir + 1) + '/' + str(batch_size)
        X[count_dir, :, :, :, :], image_path = get_x(path_x, snippet_id, nb_frame, type)
        image_paths.append(image_path)
        if type != 'test':
            Y[count_dir, :, :, :, :], label_path = get_y(path_y, snippet_id, nb_frame, type, synset_wnet2id)
            # Y[count_dir, :, :] = get_y(path_y, snippet_id, nb_frame, type, synset_wnet2id)
            label_paths.append(label_path)
        count_dir += 1

    return X, Y, image_paths, label_paths

def get_one_random_frame_path(image_net_directory, type, bucket_id = None):
    if type == 'train':
        path_x = image_net_directory + 'Data/VID/' + type + '/ILSVRC2015_VID_' + type + '_' + bucket_id + '/'
        path_y = image_net_directory + 'Annotations/VID/' + type + '/ILSVRC2015_VID_' + type + '_' + bucket_id + '/'

    elif type == 'val':
        path_x = image_net_directory + 'Data/VID/' + type + '/'
        path_y = image_net_directory + 'Annotations/VID/' + type + '/'

    elif type == 'test':
        path_x = image_net_directory + 'Data/VID/' + type + '/'
        path_y = None

    random_dir = random.sample(os.listdir(path_x), 1)

    regexp = "ILSVRC2015_"+ type +"_([0-9]*)"
    snippet_id = re.match(regexp,random_dir[0]).group(1)

    random_frame = random.sample(os.listdir(path_x + 'ILSVRC2015_' + type + '_' + snippet_id + '/'), 1)
    regexp = "([0-9]*).JPEG"
    frame_id = re.match(regexp,random_frame[0]).group(1)

    path_x = path_x + 'ILSVRC2015_' + type + '_' + snippet_id + '/' + frame_id + '.JPEG'
    if type != 'test':
        path_y = path_y + 'ILSVRC2015_' + type + '_' + snippet_id + '/' + frame_id + '.xml'

    return path_x, path_y

def get_one_random_video_path(image_net_directory, type, bucket_id = None):
    if type == 'train':
        path_x = image_net_directory + 'Data/VID/' + type + '/ILSVRC2015_VID_' + type + '_' + bucket_id + '/'
        path_y = image_net_directory + 'Annotations/VID/' + type + '/ILSVRC2015_VID_' + type + '_' + bucket_id + '/'

    elif type == 'val':
        path_x = image_net_directory + 'Data/VID/' + type + '/'
        path_y = image_net_directory + 'Annotations/VID/' + type + '/'

    elif type == 'test':
        path_x = image_net_directory + 'Data/VID/' + type + '/'
        path_y = None

    random_dir = random.sample(os.listdir(path_x), 1)

    regexp = "ILSVRC2015_"+ type +"_([0-9]*)"
    snippet_id = re.match(regexp,random_dir[0]).group(1)

    path_x = path_x + 'ILSVRC2015_' + type + '_' + snippet_id + '/'
    if type != 'test':
        path_y = path_y + 'ILSVRC2015_' + type + '_' + snippet_id + '/'

    return path_x, path_y

if __name__ == "__main__":
    pass
