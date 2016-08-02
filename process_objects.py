import re, numpy as np
from parameters import *

def process_objects(width, height, objects, synset_wnet2id):
    # y : (S * S * (x,y,w,h,p1...p30,obj))
    y = np.zeros((S,S,4 + num_categories + 1))
    boxes = []
    for i in range(S):
        rows = []
        for j in range(S):
            rows.append(False)
        boxes.append(rows)

    x_unit = width / S
    y_unit = height / S

    for obj in objects:
        trackid, name, xmax, xmin, ymax, ymin, occluded, generated = obj
        x_center = (xmax + xmin) / 2
        y_center = (ymax + ymin) / 2

        xbox = x_center / x_unit
        ybox = y_center / y_unit

        if(boxes[xbox][ybox] == False):
            idbox = int(synset_wnet2id[name]) - 1
            y[xbox, ybox, 4 + idbox] = 1
            y[xbox, ybox, 4 + num_categories] = 1

            xcoord = (float(x_center) / x_unit) - xbox
            ycoord = (float(y_center) / y_unit) - ybox

            w = float(xmax - xmin) / width
            h = float(ymax - ymin) / height

            y[xbox, ybox, 0] = xcoord
            y[xbox, ybox, 1] = ycoord
            y[xbox, ybox, 2] = w
            y[xbox, ybox, 3] = h

            boxes[xbox][ybox] = True

    return y

# def process_objects(width, height, objects, synset_wnet2id):
#     y = np.zeros((num_categories,))
#     already_filled = False

#     for obj in objects:
#         trackid, name, xmax, xmin, ymax, ymin, occluded, generated = obj
#         idbox = int(synset_wnet2id[name]) - 1
#         if already_filled == False:
#             y[idbox,] = 1
#             # already_filled = True

#     return y

if __name__ == "__main__":
    pass
