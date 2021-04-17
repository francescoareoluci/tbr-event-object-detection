"""
test_gen1.py: Fork of the YOLOv3 implementation.
This script will also output a GEN1 compliant
npy array for each test event in order to use
the prophesee COCO evaluation.
"""

from __future__ import division

import sys
sys.path.insert(0, '../PyTorch-YOLOv3/')
from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def extract_bboxes(tensor, timestamp, img_size):
    bboxes = []
    if tensor is None:
        return bboxes

    gen1_img_width = 304
    gen1_img_height = 240
    clone_tensor = tensor.clone()
    clone_tensor = clone_tensor.numpy()
    # Rescale boxes to original image
    bbox_list = rescale_boxes(clone_tensor, img_size, (gen1_img_height, gen1_img_width))
    for b in bbox_list:
        x1 = b[0]
        y1 = b[1]
        x2 = b[2]
        y2 = b[3]

        w = x2 - x1
        h = y2 - y1

        pred_cls = int(b[6])
        conf = float(b[5])

        bbox = [timestamp, int(x1), int(y1), int(w), int(h), pred_cls, conf, 0]
        bboxes.append(tuple(bbox))
    
    return bboxes

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, gen1_output, acc_time):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, 
                          img_size=img_size, 
                          multiscale=False, 
                          transform=DEFAULT_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    event_npy = []
    gen1_data_type= np.dtype([('ts', '<u8'), 
                            ('x', '<f4'), 
                            ('y', '<f4'), 
                            ('w', '<f4'), 
                            ('h', '<f4'), 
                            ('class_id', 'u1'),
                            ('confidence', '<f4'),
                            ('track_id', '<u4')])
    last_event = ""
    rel_path = gen1_output
    for batch_i, (img_names, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        
        if targets is None:
            continue
            
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        curr_event = os.path.basename(img_names[0])
        event_split = curr_event.split('_')
        curr_event = event_split[0] + "_" + event_split[1] + "_" + event_split[2] + "_" + event_split[3]
        file_ts = int(event_split[4].split('.')[0]) * 1000    # to microseconds
        # @TODO: tbd if this is an optimal approximation
        ts = (file_ts + (accumulation_time / 2))

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        if last_event == "":
            # First event
            bboxes = extract_bboxes(outputs[0], ts, img_size)
            if len(bboxes) != 0:
                event_npy += bboxes
        elif last_event == curr_event:
            # Prediction of the same event
            bboxes = extract_bboxes(outputs[0], ts, img_size)
            if len(bboxes) != 0:
                event_npy += bboxes
        else:
            # Prediction of another event
            # Save old array
            event_npy = np.array(event_npy, dtype=gen1_data_type)
            np.save(rel_path + "/" + last_event + "_bbox.npy", event_npy)
            # Create new array
            event_npy = []
            bboxes = extract_bboxes(outputs[0], ts, img_size)
            if len(bboxes) != 0:
                event_npy += bboxes

        last_event = curr_event

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    
    # Save last event
    event_npy = np.array(event_npy, dtype=gen1_data_type)
    np.save(rel_path + "/" + curr_event + "_bbox.npy", event_npy)

    if len(sample_metrics) == 0:  # no detections over whole validation set.
        return None
    
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--total_acc_time", type=int, default=20000, 
                        help="total accumulation time in microseconds (for tbe = accumulation time * nbits)")
    parser.add_argument("--gen1_output", type=str, default="../gen1_arrays", 
                        help="path where GEN1 npy file should be stored")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    gen1_output_path = opt.gen1_output
    accumulation_time = opt.total_acc_time

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=1,
        gen1_output=gen1_output_path,
        acc_time=accumulation_time
    )

    print("Class precisions:")
    for i, c in enumerate(precision):
        print(f"+ Class '{i}' ({class_names[i]}) - Precision: {c}")

    print("Class recalls:")
    for i, c in enumerate(recall):
        print(f"+ Class '{i}' ({class_names[i]}) - Recall: {c}")

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
