import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from encoders import *
from utils import *
from tbe import TemporalBinaryEncoding
from cl_parser import CLParser

import sys
sys.path.insert(0, '../prophesee-automotive-dataset-toolbox/')
from src.io.psee_loader import PSEELoader


def setupDirectories(root_dir):
    '''
        @brief: Setup directories as requested in YOLOV3
                implementation. 
        @return: Dict of useful directories:
                "train": train_images_path
                "valid": valid_images_path
                "test_file": test_file_path,
                "labels": labels_path
                "completed": completed_file_path
    '''

    start_folder = 'data'
    data_path = root_dir + '/' + start_folder
    custom_path = data_path + '/custom'
    images_path = custom_path + '/images'
    labels_path = custom_path + '/labels'
    classes_file_path = custom_path + '/classes.names'
    train_file_path = custom_path + '/train.txt'
    valid_file_path = custom_path + '/valid.txt'
    test_file_path = custom_path + '/test.txt'
    completed_file_path = data_path + "/completed_videos"
    evaluated_enc_path = data_path + "/evaluated_enc"

    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    start_folder_abs = os.path.abspath(data_path)
    list_path = start_folder_abs + '/custom/images/'

    if not os.path.isdir(custom_path):
        os.mkdir(custom_path)

    if not os.path.isdir(images_path):
        os.mkdir(images_path)

    if not os.path.isdir(labels_path):
        os.mkdir(labels_path)

    if not os.path.isdir(evaluated_enc_path):
        os.mkdir(evaluated_enc_path)

    ## Setup classes
    f = open(classes_file_path, "w")
    f.write("vehicle\n")
    f.write("pedestrian\n")
    f.close

    if not os.path.isfile(completed_file_path):
        f = open(completed_file_path, "x")
        f.close()

    if not os.path.isfile(train_file_path):
        f = open(train_file_path, "x")
        f.close()

    if not os.path.isfile(valid_file_path):
        f = open(valid_file_path, "x")
        f.close()

    if not os.path.isfile(test_file_path):
        f = open(test_file_path, "x")
        f.close()

    return {
            "images": images_path,
            "labels": labels_path,
            "train_file": train_file_path,
            "valid_file": valid_file_path,
            "test_file": test_file_path,
            "list": list_path,
            "completed": completed_file_path,
            "enc": evaluated_enc_path
            }


def getEventList(directory):
    '''
        @brief: Check in directory for events and bbox annotations. 
                An event is valid if annotation file 
                with same basename exists.
        @return: List of basenames of valid event files.
    '''

    file_list_npy = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and 
                                                                os.path.splitext(os.path.join(directory, file))[1] == '.npy']
    file_list_dat = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and 
                                                                os.path.splitext(os.path.join(directory, file))[1] == '.dat']
    filtered_file_list = []
    for td in file_list_dat:
        td_split = td.split('_')
        td = td_split[0] + "_" + td_split[1] + "_" + td_split[2] + "_" + td_split[3]
        for bbox in file_list_npy:
            bbox_split = bbox.split('_')
            bbox = bbox_split[0] + "_" + bbox_split[1] + "_" + bbox_split[2] + "_" + bbox_split[3]
            if td == bbox:
                filtered_file_list.append(td)

    return filtered_file_list


## Parsing arguments
print("Event to frame converter")

parser = CLParser()
args = parser.parse()
save_encoding = True if args.save_enc > 0 else False
use_stored_encoding = True if args.use_stored_enc > 0 else False
show_video = True if args.show_video > 0 else False
tbr_bits_requested = True if args.tbr_bits != None else False
src_video_requested = True if args.src_video != None else False
dest_path_requested = True if args.dest_path != None else False
event_type_requested = True if args.event_type != None else False
save_path_requested = True if args.save_bb_img != None else False
accumulation_time_requested = True if args.accumulation_time != None else False
encoder_type_requested = True if args.encoder != None else False

dest_root_folder = '..'
if dest_path_requested:
    dest_root_folder = args.dest_path[0]

save_path = ""
if save_path_requested:
    save_path = args.save_bb_img[0] + '/'

if save_path_requested and show_video:
    print("Unable to save frames and showing video, using save frame as option")

## Setup data directory to save files (images, bboxes and labels)
dir_paths = setupDirectories(dest_root_folder)

video_dir = "../train_events/"
if src_video_requested:
    video_dir = args.src_video[0] + '/'

data_type = 'train'
if event_type_requested:
    if args.event_type[0] == 'train' or args.event_type[0] == 'validation' or args.event_type[0] == 'test':
        data_type = args.event_type[0]
    else:
        print("Invalid event type requested. Supported: <train | validation | test>.")
        exit()

## Encoder
requested_encoder = 'tbe'
if encoder_type_requested:
    if args.encoder[0] == 'tbe' or args.encoder[0] == 'polarity':
        requested_encoder = args.encoder[0]
    else:
        print("Invalid encoder requested. Using default encoder (tbe)")

## Number of events to be accumulated
N = 16
if tbr_bits_requested and args.tbr_bits[0] > 0:
    N = args.tbr_bits[0]

## Accumulation time (microseconds)
delta_t = 1000
if accumulation_time_requested and args.accumulation_time[0] > 0:
    delta_t = args.accumulation_time[0]

## Print some info
print("===============================")
print("Encoder: " + requested_encoder)
print("Requested encoded array saving: " + str(save_encoding))
print("Requested saved encoded array loading: " + str(use_stored_encoding))
print("Requested video show during processing: " + str(show_video))
if requested_encoder == 'tbe':
    print("Accumulating {:d} events".format(N))
print("Accumulation time: " + str(delta_t))
print("Source event path: " + video_dir)
print("Destination path: " + dest_root_folder + '/data')
print("Event data type: " + data_type)
print("===============================")

if data_type == "train":
    txt_list_file = 'train_file'
elif data_type == 'validation':
    txt_list_file = 'valid_file'
else:
    txt_list_file = 'test_file'

## Iterate through videos in video_dir to get list 
video_names = getEventList(video_dir)

## Iterate videos
for video_name in video_names:
    video_path = video_dir + video_name

    with open(dir_paths['completed']) as completed_videos:
        if video_name in completed_videos.read():
            print("Skipping completed video: " + video_name)
            continue

    print("Processing video: " + video_name)

    arr = np.load(video_path + "_bbox.npy")

    ## Load video
    video = PSEELoader(video_path + "_td.dat")

    width = video.get_size()[1]
    height = video.get_size()[0]
    encoder = TemporalBinaryEncoding(N, width, height)

    if not use_stored_encoding:
        ## Convert event video to a Temporal Binary Encoded frames array
        if requested_encoder == 'tbe':
            encoded_array = encode_video_tbe(N, width, height, video, encoder, delta_t)
        elif requested_encoder == 'polarity' and accumulation_time_requested:
            encoded_array = encode_video_polarity(width, height, video, delta_t)
        else:
            encoded_array = encode_video_polarity(width, height, video)

        if save_encoding:
            np.save(dir_paths["enc"] + video_name + "_enc.npy", encoded_array)
    else:
        ## Use the pre-evaluated encoded (tbe or else) array
        encoded_array = np.load(dir_paths["enc"] + video_name + "_enc.npy")

    ## Iterate through video frames
    img_count = 0
    bbox_count = 0
    print("Saving encoded frames and bounding boxes...")
    for f in encoded_array:
        bboxes = get_frame_BB(f, arr)

        filename = video_name + str("_" + str(f["startTs"]))
        ## Save images that have at least a bbox
        if len(bboxes) != 0:
            ## Save image
            plt.imsave(dir_paths["images"] + "/" + filename + ".jpg", f['frame'], vmin=0, vmax=1, cmap='gray')

            ## Update train or validation txt file (append if not existing)
            with open(dir_paths[txt_list_file], "r+") as list_txt_file:
                file_string = dir_paths["list"] + filename + ".jpg"
                for line in list_txt_file:
                    ## Search for image file path in this file
                    if file_string in line:
                        break
                else:   # Note: this indentation is intentional
                    ## If entered, the string does not exist in this file
                    ## Append file path
                    list_txt_file.write(file_string + "\n")

            ## Write BBoxes in labels
            label_file = open(dir_paths["labels"] + "/" + filename + ".txt", "w")
            for b in bboxes:
                conv_bbox = convertBBoxCoords(b, width, height)
                label_file.write(str("%d" % conv_bbox[0]) + " ")
                label_file.write(str("%.8f" % conv_bbox[1]) + " ")
                label_file.write(str("%.8f" % conv_bbox[2]) + " ")
                label_file.write(str("%.8f" % conv_bbox[3]) + " ")
                label_file.write(str("%.8f" % conv_bbox[4]) + "\n")
                bbox_count += 1
            label_file.close()

            if save_path_requested:
                save_bb_image(f['frame'], bboxes, save_path + filename + "_bb.jpg")

            img_count += 1

        if show_video:
            show_image(f['frame'], bboxes)
            plt.pause(0.05)

    print("Saved {:d} encoded frames in path: {:s}".format(img_count, dir_paths["images"]))
    print("Saved {:d} bounding boxes annotations in path: {:s}".format(bbox_count, dir_paths["labels"]))

    completed_file = open(dir_paths['completed'], "a")
    completed_file.write(video_name + "\n")
    completed_file.close()

print("Done")
