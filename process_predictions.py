from parameters import *
import numpy as np
from ILSVRC_parsing import parse_ILSVRCXML
from matplotlib import pyplot as plt
import cv2

def process_predictions(y_pred, y_true, image_path, label_path, nb_frame, synset_id2name, synset_wnet2name, show = False):
    predictions = np.reshape(y_pred, (nb_frame, S, S, 4 + num_categories + 1))
    truth = np.reshape(y_true, (nb_frame, S, S, 4 + num_categories + 1))
    treshold = 0.2

    for f in range(nb_frame):
        boxes = []
        path = label_path + str(f).zfill(6) + '.xml'
        folder, filename, database, width, height, objects = parse_ILSVRCXML(path)

        x_unit = width / S
        y_unit = height / S

        for i in range(S*S):
            col = i / S
            row = i % S

            x = min(max((predictions[f, row, col, 0] + col) * x_unit, 0), x_unit *(col + 1))
            y = min(max((predictions[f, row, col, 1] + row) * y_unit, 0), y_unit *(row + 1))
            w = min(max(predictions[f, row, col, 2] * width,0), width)
            h = min(max(predictions[f, row, col, 3] * height,0), height)
            
            x_true = min(max((truth[f, row, col, 0] + col) * x_unit, 0), x_unit *(col + 1))
            y_true = min(max((truth[f, row, col, 1] + row) * y_unit, 0), y_unit *(row + 1))
            w_true = min(max(truth[f, row, col, 2] * width,0), width)
            h_true = min(max(truth[f, row, col, 3] * height,0), height)
            
            scale = predictions[f, row, col, 4 + num_categories]

            probs = predictions[f, row, col, 4 : 4 + num_categories] * scale
            filter_mat_probs = np.array(probs >= treshold, dtype = 'bool')
            filter_mat_boxes = np.nonzero(filter_mat_probs)
            
            if f==10:
                print x,y,w,h,scale
                print x_true, y_true, w_true, h_true
                print '\n'
            
            top = np.argmax(predictions[f, row, col, 4 : (4 + num_categories)])
          
            boxes.append([x,y,w,h,top])
        if show:
            show_predictions(image_path + str(f).zfill(6) + '.JPEG', width, height, objects, boxes, f, synset_id2name, synset_wnet2name)

def show_predictions(image_path,width, height, objects, boxes, frame_id, synset_id2name, synset_wnet2name):
    im = cv2.imread(image_path)
    x_unit = width / S
    y_unit = height / S
    for x,y,w,h,top in boxes:
        cv2.rectangle(im, (int(x), int(y)) , (int(x), int(y)), color = (255,0,0), thickness = 5)

        xmin = max(int(x - w/2), 0)
        xmax = min(int(x + w/2), width)
        ymin = max(int(y - h/2), 0)
        ymax = min(int(y + h/2), height)

        cv2.rectangle(im, (xmin, ymin) , (xmax, ymax), color = (0,0,255), thickness = 3)
        cv2.rectangle(im, (xmin, ymin-40) , (xmax, ymin), color = (255,255,255), thickness = -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, synset_id2name[str(top+1)], (xmin, ymin-10), font, 1, (0,0,0))
    for trackid, name, xmax, xmin, ymax, ymin, occluded, generated in objects:
        #cv2.rectangle(im, (xmin, ymin-40) , (xmax, ymin), color = (255,255,255), thickness = -1)
        #cv2.putText(im, synset_wnet2name[name], (xmin, ymin-10), font, 1, (0,0,0))
        cv2.rectangle(im, (xmin, ymin) , (xmax, ymax), color = (0,255,0), thickness = 3)
    plt.imshow(im)
    plt.savefig(examples_path + str(frame_id) + '.jpg')
