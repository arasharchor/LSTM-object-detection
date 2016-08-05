import cv2, math, numpy as np
from parameters import *
from ILSVRC_parsing import parse_ILSVRCXML
from matplotlib import pyplot as plt
from image_preprocessing import shift

def process_predictions(y_pred, y_true, image_path, label_path, index, nb_frame, synset_id2name, show = False):
    predictions = np.reshape(y_pred, (nb_frame, S, S, 4 + num_categories + 1))
    truth = np.reshape(y_true, (nb_frame, S, S, 4 + num_categories + 1))
    thresh = threshold

    for f in range(nb_frame):
        boxes = []
        path = label_path + str(index + f).zfill(6) + '.xml'
        folder, filename, database, width, height, objects = parse_ILSVRCXML(path)

        x_unit = width / S
        y_unit = height / S

        for i in range(S*S):
            row = i / S
            col = i % S

            x = (predictions[f, row, col, 0] + row) * x_unit
            y = (predictions[f, row, col, 1] + col) * y_unit
            w = pow(predictions[f, row, col, 2], 2) * width
            h = pow(predictions[f, row, col, 3], 2) * height

            x_true = (truth[f, row, col, 0] + row) * x_unit
            y_true = (truth[f, row, col, 1] + col) * y_unit
            
            w_true = pow(truth[f, row, col, 2], 2) * width
            h_true = pow(truth[f, row, col, 3], 2) * height

            scale = predictions[f, row, col, 4 + num_categories]
            
            top = np.argmax(predictions[f, row, col, 4 : (4 + num_categories)])
            top_true = np.argmax(truth[f, row, col, 4 : (4 + num_categories)])
            
            if truth[f, row, col, 4 + num_categories] == .5:
                print int(x),int(y),int(w),int(h), top
                print predictions[f, row, col, 4 : (4 + num_categories)]
                print x_true,y_true, w_true, h_true, top_true

            boxes.append([x,y,w,h,top,x_true,y_true,w_true,h_true,top_true, truth[f, row, col, 4 + num_categories]])
        if show:
            show_predictions(image_path + str(index + f).zfill(6) + '.JPEG', width, height, objects, boxes, f, synset_id2name)

def show_predictions(image_path,width, height, objects, boxes, frame_id, synset_id2name):
    im = cv2.imread(image_path)

    x_unit = width / S
    y_unit = height / S

    # Drawing the grid
    for s in range(S):
        cv2.line(im, (s*x_unit, 0), (s*x_unit, height), color = (0,0,0), thickness = 2)
        cv2.line(im, (0, s*y_unit), (width, s*y_unit),  color = (0,0,0), thickness = 2)

    # Drawing the predicted boxes
    for x,y,w,h,top,x_true,y_true,w_true,h_true,top_true, objectness in boxes:
        cv2.rectangle(im, (int(x), int(y)) , (int(x), int(y)), color = (255,0,0), thickness = 5)

        xmin = min(max(int(x - w/2), 0), width)
        xmax = min(max(int(x + w/2), 0), width)
        ymin = min(max(int(y - h/2), 0), height)
        ymax = min(max(int(y + h/2), 0), height)

        #cv2.rectangle(im, (xmin,ymin), (xmax, ymin+30), color = (255,255,255), thickness=-1)
        cv2.rectangle(im, (xmin, ymin) , (xmax, ymax), color = (0,0,255), thickness = 3)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(im, synset_id2name[str(top+1)], (xmin, ymin+30), font, 1, (0,0,0))

    # Drawing the ground truth boxes
    #for trackid, name, xmax, xmin, ymax, ymin, occluded, generated in objects:    
        if objectness == 1:
            xmin = min(max(int(x_true - w_true/2), 0), width)
            xmax = min(max(int(x_true + w_true/2), 0), width)
            ymin = min(max(int(y_true - h_true/2), 0), height)
            ymax = min(max(int(y_true + h_true/2), 0), height)
            #cv2.rectangle(im, (xmin,ymin), (xmax, ymin+30), color = (255,255,255), thickness=-1)
            #cv2.putText(im, synset_id2name[str(top_true+1)], (xmin, ymin+30), font, 1, (0,0,0))
            cv2.rectangle(im, ((xmin+xmax)/2, (ymin+ymax)/2) , ((xmin+xmax)/2, (ymin+ymax)/2), color = (0,255,0), thickness = 5)
            cv2.rectangle(im, (xmin, ymin) , (xmax, ymax), color = (0,255,0), thickness = 3)

    # Saving the image
    plt.imshow(im)
    plt.savefig(examples_path + str(frame_id) + '.jpg')
