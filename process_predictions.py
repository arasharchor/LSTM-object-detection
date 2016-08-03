import cv2, math, numpy as np
from parameters import *
from ILSVRC_parsing import parse_ILSVRCXML
from matplotlib import pyplot as plt

def process_predictions(y_pred, y_true, image_path, label_path, index, nb_frame, synset_id2name, synset_wnet2name, show = False):
    predictions = np.reshape(y_pred, (nb_frame, S, S, 4 + num_categories + 1))
    truth = np.reshape(y_true, (nb_frame, S, S, 4 + num_categories + 1))
    treshold = 0.2
    for f in range(nb_frame):
        boxes = []
        path = label_path + str(index + f).zfill(6) + '.xml'
        folder, filename, database, width, height, objects = parse_ILSVRCXML(path)

        x_unit = width / S
        y_unit = height / S

        for i in range(S*S):
            col = i / S
            row = i % S

            x = (predictions[f, row, col, 0] + col) * x_unit
            y = (predictions[f, row, col, 1] + row) * y_unit
            w = pow(predictions[f, row, col, 2], 2) * width
            h = pow(predictions[f, row, col, 3], 2) * height

            x_true = (truth[f, row, col, 0] + col) * x_unit
            y_true = (truth[f, row, col, 1] + row) * y_unit
            w_true = pow(truth[f, row, col, 2], 2) * width
            h_true = pow(truth[f, row, col, 3], 2) * height

            scale = predictions[f, row, col, 4 + num_categories]

            top = np.argmax(predictions[f, row, col, 4 : (4 + num_categories)])

            boxes.append([x,y,w,h,top])
        if show:
            show_predictions(image_path + str(index + f).zfill(6) + '.JPEG', width, height, objects, boxes, f, synset_id2name, synset_wnet2name)

def show_predictions(image_path,width, height, objects, boxes, frame_id, synset_id2name, synset_wnet2name):
    im = cv2.imread(image_path)
    x_unit = width / S
    y_unit = height / S

    # Drawing the grid
    for s in range(S):
        cv2.line(im, (s*x_unit, 0), (s*x_unit, height), color = (0,0,0), thickness = 2)
        cv2.line(im, (0, s*y_unit), (width, s*y_unit),  color = (0,0,0), thickness = 2)

    # Drawing the predicted boxes
    for x,y,w,h,top in boxes:
        cv2.rectangle(im, (int(x), int(y)) , (int(x), int(y)), color = (255,0,0), thickness = 5)

        xmin = max(int(x - w/2), 0)
        xmax = min(int(x + w/2), width)
        ymin = max(int(y - h/2), 0)
        ymax = min(int(y + h/2), height)

        cv2.rectangle(im, (xmin, ymin) , (xmax, ymax), color = (0,0,255), thickness = 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, synset_id2name[str(top+1)], (xmin, ymin+30), font, 1, (0,0,0))

    # Drawing the ground truth boxes
    for trackid, name, xmax, xmin, ymax, ymin, occluded, generated in objects:
        cv2.putText(im, synset_wnet2name[name], (xmin, ymin+30), font, 1, (0,0,0))
        cv2.rectangle(im, ((xmin+xmax)/2, (ymin+ymax)/2) , ((xmin+xmax)/2, (ymin+ymax)/2), color = (0,255,0), thickness = 5)
        cv2.rectangle(im, (xmin, ymin) , (xmax, ymax), color = (0,255,0), thickness = 3)

    # Saving the image
    plt.imshow(im)
    plt.savefig(examples_path + str(frame_id) + '.jpg')
