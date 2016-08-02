from matplotlib import pyplot as plt
from ILSVRC_parsing import parse_ILSVRCXML
from get_data import get_one_random_frame_path, get_one_random_video_path
from parameters import *
import cv2, os, re

def show_frame_boxes(path_to_frame, path_to_label):
	im = cv2.imread(path_to_frame)
	folder, filename, database, width, height, objects = parse_ILSVRCXML(path_to_label)
	for obj in objects:
		trackid, name, xmax, xmin, ymax, ymin, occluded, generated = obj
		cv2.rectangle(im, (int(xmin),int(ymin)), (int(xmax), int(ymax)), color = (0,255,0), thickness = 5)
	plt.imshow(im)
	plt.show()

def save_video_boxes(path_to_video, path_to_label, nb_frame = float('Inf')):
	count_file = 0
	for file in sorted(os.listdir(path_to_video)):
		if count_file < nb_frame:
			regexp = "([0-9]*).JPEG"
			frame_id = re.match(regexp,file).group(1)

			im = cv2.imread(path_to_video + file)
			folder, filename, database, width, height, objects = parse_ILSVRCXML(path_to_label + frame_id + '.xml')
			for obj in objects:
				trackid, name, xmax, xmin, ymax, ymin, occluded, generated = obj
				cv2.rectangle(im, (int(xmin),int(ymin)), (int(xmax), int(ymax)), color = (0,255,0), thickness = 5)
				plt.imshow(im)
				plt.savefig(examples_path + file)
			count_file += 1

if __name__ == "__main__":
    # path_x,path_y = get_one_random_frame_path(image_net_directory, 'train', '0000')
    # show_frame_boxes(path_x, path_y)

    # path_x,path_y = get_one_random_video_path(image_net_directory, 'train', '0000')
    # save_video_boxes(path_x, path_y, 50)