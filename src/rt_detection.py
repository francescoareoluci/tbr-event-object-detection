"""
rt_detection.py: Real Time Detection, uses YOLOv3 implementation
in order to detect objects on an GEN1 event video (.dat) using
Temporal Binary Encoding
"""

from __future__ import division

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from encoders import *
from ec_utils import *
from tbe import TemporalBinaryEncoding

import sys
sys.path.insert(0, '../PyTorch-YOLOv3/')
from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *

sys.path.insert(0, '../prophesee-automotive-dataset-toolbox/')
from src.io.psee_loader import PSEELoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_video", type=str, default="../data/event.dat", 
                        help="event_video: path to event video (.dat)")
    parser.add_argument('--tbr_bits', '-n', type=int, default=8,
                        help='tbr_bits: set the number of bits for Temporal Binary Representation. Default: 8')
    parser.add_argument('--accumulation_time', '-a', type=int, default=2500,
                        help='accumulation_time: set the quantization time of events (microseconds). Default: 2500')
    parser.add_argument("--model_def", type=str, default="../PyTorch-YOLOv3/config/yolov3.cfg",
                        help="model_def: path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../PyTorch-YOLOv3/weights/yolov3.weights", 
                        help="weights_path: path to weights file")
    parser.add_argument("--class_path", type=str, default="../data/classes.names", 
                        help="class_path: path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, 
                        help="conf_thres: object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, 
                        help="nms_thres: iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="batch_size: size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, 
                        help="m_cpu: number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, 
                        help="img_size: size of each image dimension")
    parser.add_argument('--show_video', action='count', default=0,
                        help='show_video: show video with evaluated TBR frames and their bboxes during processing. Default: false')
    parser.add_argument('--save_frames', action='count', default=0,
                        help='save_frames: save TBE frames and their detection as images')

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_path = "../rt_detections"
    if opt.save_frames > 0:
        os.makedirs(output_path, exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    # Number of bits to be used in Temporal Binary Encoding
    tbr_bits = opt.tbr_bits

    # Accumulation time (microseconds)
    delta_t = opt.accumulation_time
    
    gen1_video = PSEELoader(opt.event_video)

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    # Set up TBE vars
    accumulation_mat = np.zeros((tbr_bits, gen1_video.get_size()[0], gen1_video.get_size()[1]))
    tbe_frame = np.zeros(gen1_video.get_size())
    encoder = TemporalBinaryEncoding(tbr_bits, gen1_video.get_size()[1], gen1_video.get_size()[0])

    i = 0
    batch_count = 0
    prev_time = time.time()
    # Parse events and build TBE frames
    while not gen1_video.done:
        i = (i + 1) % tbr_bits
        # Load next 1ms events from the video
        events = gen1_video.load_delta_t(delta_t)
        f = np.zeros(gen1_video.get_size())
        for e in events:
            # Evaluate presence/absence of event for
            # a certain pixel
            f[e['y'], e['x']] = 1

        accumulation_mat[i, ...] = f

        if i == tbr_bits - 1:
            # Encode frame
            tbe_frame = encoder.encode(accumulation_mat)

            transform = transforms.Compose([            
                ToTensor(),
                Resize(opt.img_size)
            ])

            # Implemented transformations expect bbox array. Use a fake array
            # @TODO: find a better solution...
            input_img, bbox = transform([Image.fromarray(255 * tbe_frame).convert('RGB'), np.zeros((1,1))])
            # Add batch size (1)
            input_img = torch.unsqueeze(input_img, 0)
            input_img = input_img.to(device)
        
            # Detect objects on TBE frame
            with torch.no_grad():
                detections = model(input_img)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_count, inference_time))
            batch_count += 1

            bbox_list = []
            if detections[0] is not None:
                for d in detections:
                    d = d.cpu()
                    to_list = d.tolist()
                    if len(to_list) != 0:
                        bbox_list.append(to_list)
            
            bbox_list = np.array(bbox_list)
            bboxes = []
            if len(bbox_list) != 0:
                # Rescale boxes to original image
                bbox_list = rescale_boxes(bbox_list[0], opt.img_size, tbe_frame.shape)
                print(bbox_list)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in bbox_list:

                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1
                    bbox = [0, x1, y1, box_w, box_h, cls_pred]
                    bboxes.append(bbox)

            if opt.show_video > 0:
                show_image(tbe_frame, np.array(bboxes))
                plt.pause(0.001)

            if opt.save_frames > 0:
                save_bb_image(tbe_frame, np.array(bboxes), output_path + "/" + str(batch_count) + ".png", False)
