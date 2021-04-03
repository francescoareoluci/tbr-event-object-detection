import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from encoders import *
from utils import *
from dir_handler import *
from tbe import TemporalBinaryEncoding
from cl_parser import CLParser

import sys
sys.path.insert(0, '../prophesee-automotive-dataset-toolbox/')
from src.io.psee_loader import PSEELoader


if __name__ == "__main__":
    print("Event to frame converter")

    ## Parsing arguments
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
        if args.encoder[0] == 'tbe' or args.encoder[0] == 'polarity' or args.encoder[0] == 'sae':
            requested_encoder = args.encoder[0]
        else:
            print("Invalid encoder requested")
            exit()

    ## Number of bits to be used in Temporal Binary Encoding
    tbr_bits = 16
    if tbr_bits_requested and args.tbr_bits[0] > 0:
        tbr_bits = args.tbr_bits[0]

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
        print("Using {:d} bits to represent events".format(tbr_bits))
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

    ## Setup data directory to save files (images, bboxes and labels)
    dir_paths = setupDirectories(dest_root_folder)

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

        gen1_bboxes = np.load(video_path + "_bbox.npy")

        ## Load video
        gen1_video = PSEELoader(video_path + "_td.dat")

        width = gen1_video.get_size()[1]
        height = gen1_video.get_size()[0]
        encoder = TemporalBinaryEncoding(tbr_bits, width, height)

        if not use_stored_encoding:
            ## Convert event video to a Temporal Binary Encoded frames array
            if requested_encoder == 'tbe':
                encoded_array = encode_video_tbe(tbr_bits, width, height, gen1_video, encoder, delta_t)
            elif requested_encoder == 'polarity':
                encoded_array = encode_video_polarity(width, height, gen1_video, delta_t)
            else:
                encoded_array = encode_video_sae(width, height, gen1_video, delta_t)

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
            bboxes = get_frame_BB(f, gen1_bboxes)

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
