import re, math, numpy as np
from parameters import *

def process_objects(width, height, objects, wrg, hrg, synset_wnet2id):
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
        xmax = min(max(int(xmax - hrg * width), 0), width)
        xmin = min(max(int(xmin - hrg * width), 0), width)
        ymax = min(max(int(ymax - wrg * height), 0), height)
        ymin = min(max(int(ymin - wrg * height), 0), height)
        
        x_center = min((xmax + xmin) / 2, width)
        y_center = min((ymax + ymin) / 2, height)

        xbox = min(x_center / x_unit, S-1)
        ybox = min(y_center / y_unit, S-1)

        if(boxes[xbox][ybox] == False): # Just one object per cell
            idbox = int(synset_wnet2id[name]) - 1
            y[xbox, ybox, 4 + idbox] = 1
            y[xbox, ybox, 4 + num_categories] = 1

            xcoord = (float(x_center) / x_unit) - xbox
            ycoord = (float(y_center) / y_unit) - ybox

            w = float(xmax - xmin) / width
            h = float(ymax - ymin) / height

            y[xbox, ybox, 0] = xcoord
            y[xbox, ybox, 1] = ycoord
            y[xbox, ybox, 2] = math.sqrt(w)
            y[xbox, ybox, 3] = math.sqrt(h)

            boxes[xbox][ybox] = True

    return y

if __name__ == "__main__":
    pass
