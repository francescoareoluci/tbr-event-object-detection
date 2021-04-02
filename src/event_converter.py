import math
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm
from tbe import TemporalBinaryEncoding

import sys
sys.path.insert(0, '../prophesee-automotive-dataset-toolbox/')
from src.io.psee_loader import PSEELoader


def encode_video(N, width, height, video, encoder, show_frame=False):
    '''
        @brief: Encode an event video in a sequence of frame
                using the Temporal Binary Representation
    '''
    
    # Each encoded frame will have a start/end timestamp (ms) in order
    # to associate bounding boxes later.
    # Note: If videos are longer than 1 minutes, 16 bits per ts are not sufficient.
    data_type = np.dtype([('startTs', np.uint16), 
                            ('endTs', np.uint16), 
                            ('frame', np.float64, (height, width))])
    
    samplePerVideo = math.ceil((video.total_time() / 1000) / N)
    accumulation_mat = np.zeros((N, height, width))
    tbe_array = np.zeros(samplePerVideo, dtype=data_type)

    i = 0
    j = 0
    startTimestamp = 0
    endTimestamp = 0

    with tqdm(total = samplePerVideo, file = sys.stdout) as pbar:
        while not video.done:
            i = (i + 1) % N
            # Load next 1ms events from the video
            events = video.load_delta_t(1000)
            f = np.zeros(video.get_size())
            #f = np.zeros((width, height))
            for e in events:
                # Evaluate presence/absence of event for
                # a certain pixel
                f[e['y'], e['x']] = 1

            accumulation_mat[i, ...] = f

            if i == N - 1:
                endTimestamp += N
                tbe = encoder.encode(accumulation_mat)
                tbe_array[j]['startTs'] = startTimestamp
                tbe_array[j]['endTs'] = endTimestamp
                tbe_array[j]['frame'] = tbe
                j += 1
                startTimestamp += N
                pbar.update(1)

                if show_frame:
                    plt.imshow(tbe); plt.colorbar()
                    plt.show()

    return tbe_array


def get_frame_BB(frame, BB_array):
    '''
        @brief: Associates to Temporal Binary Encoded video frame
                a list of bounding boxes with timestamp included in 
                start/end timestamp of the frame. 
        @return: The associated BB.
    '''

    associated_bb = []
    for bb in BB_array:
        # Convert timestamp to milliseconds
        timestamp = bb[0] / 1000
        startTime = frame['startTs']
        endTime = frame['endTs']
        if timestamp >= startTime and timestamp <= endTime:
            associated_bb.append(bb)
        # Avoid useless iterations
        if timestamp > endTime:
            break
    
    return np.array(associated_bb)


def show_image(frame, bboxes):
    '''
        @brief: show video of TBR frames and their bboxes
                during processing
    '''

    plt.figure(1)
    plt.clf()
    plt.imshow(frame, animated=True, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()

    # Get the current reference
    ax = plt.gca()

    # Create Rectangle boxes
    for b in bboxes:
        if b[5] == 1:
            # Person
            rect = Rectangle((b[1], b[2]), b[3], b[4], linewidth=2, edgecolor='g', facecolor='none')
        else:
            # Vehicle
            rect = Rectangle((b[1], b[2]), b[3], b[4], linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)


def save_bb_image(frame, bboxes, save_path):
    '''
        @brief: save TBR frames with their bboxes
    '''

    plt.imshow(frame, cmap='gray', vmin=0, vmax=1)

    # Get the current reference
    ax = plt.gca()

    # Create Rectangle boxes
    for b in bboxes:
        if b[5] == 1:
            # Person
            rect = Rectangle((b[1], b[2]), b[3], b[4], linewidth=2, edgecolor='g', facecolor='none')
        else:
            # Vehicle
            rect = Rectangle((b[1], b[2]), b[3], b[4], linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    
    if len(bboxes) != 0:
        plt.savefig(save_path)
        plt.close()
        

def convertBBoxCoords(bbox, image_width, image_height):
    '''
        @brief: Converts top-left starting coordinates to
                rectangle-centered coordinates. Moreover,
                coordinates and size are normalized.
        @return: np array compliant to YOLOV3 implementation.
    '''

    top_left_x = bbox[1]
    top_left_y = bbox[2]
    width = bbox[3]
    height = bbox[4]
    norm_center_x = float((top_left_x + (width / 2)) / image_width)
    norm_center_y = float((top_left_y + (height / 2)) / image_height)
    norm_width = float(width / image_width)
    norm_height = float(height / image_height)

    new_bbox = np.array([int(bbox[5]), norm_center_x, norm_center_y, norm_width, norm_height])
    
    return new_bbox


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
    evaluated_tbe_path = data_path + "/evaluated_tbe"

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

    if not os.path.isdir(evaluated_tbe_path):
        os.mkdir(evaluated_tbe_path)

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
            "tbe": evaluated_tbe_path
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


def setupArgParser():
    '''
        @brief: Setup command line arguments 
        @return: parsed arguments
    '''

    parser = argparse.ArgumentParser(description='Convert events to frames and associates bboxes')
    parser.add_argument('--use_stored_tbe', '-l', action='count', default=0,
                        help='use_stored_tbe: instead of evaluates TBR, uses pre-evaluated TBR array. Default: false')
    parser.add_argument('--save_tbe', '-s', action='count', default=0,
                        help='save_tbe: save the intermediate Temporal Binary Represented frame array. Default: false')
    parser.add_argument('--show_video', '-v', action='count', default=0,
                        help='show_video: show video with evaluated TBR frames and their bboxes during processing. Default: false')
    parser.add_argument('--accumulate', '-n', type=int, nargs=1,
                        help='accumulator: set the number of events to be accumulated. Default: 16')
    parser.add_argument('--src_video', '-t', type=str, nargs=1,
                        help='src_video: path to event videos')
    parser.add_argument('--dest_path', '-d', type=str, nargs=1,
                    help='dest_path: path where images and bboxes will be stored')
    parser.add_argument('--event_type', '-e', type=str, nargs=1,
                    help='event_type: specify data type: <train | validation | test>')
    parser.add_argument('--save_bb_img', '-b', type=str, nargs=1,
                    help='save_bb_img: save frame with bboxes to path')

    return parser.parse_args()


## Parsing arguments
args = setupArgParser()
save_tbe = True if args.save_tbe > 0 else False
use_stored_tbe = True if args.use_stored_tbe > 0 else False
show_video = True if args.show_video > 0 else False
accumulate_requested = True if args.accumulate != None else False
src_video_requested = True if args.src_video != None else False
dest_path_requested = True if args.dest_path != None else False
event_type_requested = True if args.event_type != None else False
save_path_requested = True if args.save_bb_img != None else False

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

## Iterate through videos in video_dir to get list 
video_names = getEventList(video_dir)

## Number of events to be accumulated
N = 16
if accumulate_requested and args.accumulate[0] > 0:
    N = args.accumulate[0]

## Print some info
print("Event to frame converter")
print("===============================")
print("Requested TBR array saving: " + str(save_tbe))
print("Requested saved TBR array loading: " + str(use_stored_tbe))
print("Requested video show during processing: " + str(show_video))
print("Accumulating {:d} events".format(N))
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

    if not use_stored_tbe:
        ## Convert event video to a Temporal Binary Encoded frames array
        tbe_array = encode_video(N, width, height, video, encoder)
        if save_tbe:
            np.save(dir_paths["tbe"] + video_name + "_tbe.npy", tbe_array)
    else:
        ## Use the pre-evaluated TBE array
        tbe_array = np.load(dir_paths["tbe"] + video_name + "_tbe.npy")

    ## Iterate through video frames
    img_count = 0
    bbox_count = 0
    print("Saving TBE frames and bounding boxes...")
    for f in tbe_array:
        bboxes = get_frame_BB(f, arr)

        filename = video_name + str("_" + str(f["startTs"]))
        ## Save images that have at least a bbox
        if len(bboxes) != 0:
            #print(bboxes)

            ## Save image
            plt.imsave(dir_paths["images"] + "/" + filename + ".jpg", f['frame'], vmin=0, vmax=1, cmap='gray')
            #print(f['frame'].shape)

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

    print("Saved {:d} TBR frames in path: {:s}".format(img_count, dir_paths["images"]))
    print("Saved {:d} bounding boxes annotations in path: {:s}".format(bbox_count, dir_paths["labels"]))

    completed_file = open(dir_paths['completed'], "a")
    completed_file.write(video_name + "\n")
    completed_file.close()

print("Done")
