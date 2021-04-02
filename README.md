# Temporal Binary Represented Event Object Detection

This repository contains a framework that can be used in order to perform object detection over events acquired from Event Cameras (https://en.wikipedia.org/wiki/Event_camera).
To perform test the following technologies and tools have been employed:

* Prophesee’s GEN1 Automotive Detection Dataset. This dataset contains events and their bounding boxes for pedestrian and vehicles (https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)

* Temporal Binary Representation. This encoding have been developed in order to encode events into frames that can be feed to an object detector along with annotated bounding boxes. For further details check this paper: https://arxiv.org/pdf/2010.08946.pdf

* YOLOv3 as Object Detector

## Submodules

This repository use the following repos as submodules:
* PyTorch YOLOv3 implementation: https://github.com/eriklindernoren/PyTorch-YOLOv3.git
* Prophesee Toolbox: https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox.git

Once this repository have been cloned, run:
> git submodule update --init

## Requirements

In order to execute the conversion and the object detection, use the environment.yml file to create a dedicated Conda environment.

## Convert events to frames

Events from the Prophesee’s GEN1 dataset can be converted to frames and bounding box labeling using the code inside the src/ folder. The code performs the conversion of all the events listed in a given directory and organizes the data in a folder compliant to what the YOLOv3 implementation expect. Given a destination directory, the following directory tree is generated:
``` bash
.
└── data
    ├── completed_videos
    ├── custom
    │   ├── classes.names
    │   ├── images
    │   ├── labels
    │   ├── test.txt
    │   ├── train.txt
    │   └── valid.txt
    └── evaluated_tbe

``` 

Under custom folder, the converted frames and bounding box annotations are stored in images and labels folder. Image types are specified in test, train and valid txt files. Already converted events are specified in completed_videos text file. Temporal Binary Represented array can be stored in npy format in evaluated_tbe folder to avoid performing the conversion multiple times. 

### Conversion

The conversion can be executed using src/event_converted.py file:
> python event_converter.py -h
``` bash
usage: event_converter.py [-h] [--use_stored_tbe] [--save_tbe] [--show_video]
                          [--accumulate ACCUMULATE] [--src_video SRC_VIDEO]
                          [--dest_path DEST_PATH] [--event_type EVENT_TYPE]
                          [--save_bb_img SAVE_BB_IMG]

Convert events to frames and associates bboxes

optional arguments:
  -h, --help            show this help message and exit
  --use_stored_tbe, -l  use_stored_tbe: instead of evaluates TBR, uses pre-
                        evaluated TBR array. Default: false
  --save_tbe, -s        save_tbe: save the intermediate Temporal Binary
                        Represented frame array. Default: false
  --show_video, -v      show_video: show video with evaluated TBR frames and
                        their bboxes during processing. Default: false
  --accumulate ACCUMULATE, -n ACCUMULATE
                        accumulator: set the number of events to be
                        accumulated. Default: 16
  --src_video SRC_VIDEO, -t SRC_VIDEO
                        src_video: path to event videos
  --dest_path DEST_PATH, -d DEST_PATH
                        dest_path: path where images and bboxes will be stored
  --event_type EVENT_TYPE, -e EVENT_TYPE
                        event_type: specify data type: <train | validation |
                        test>
  --save_bb_img SAVE_BB_IMG, -b SAVE_BB_IMG
                        save_bb_img: save frame with bboxes to path
```

For example, to convert events from directory /dataset/train, store results in /dest/folder and label them as train data, run the following:
> python event_converter.py --src_video /dataset/train --dest_path /dest/folder

To convert events from directory /dataset/validation, store results in /dest/folder and label them as validation data, run the following:
> python event_converter.py --src_video /dataset/validation --dest_path /dest/folder

To convert events from directory /dataset/test, store results in /dest/folder and label them as test data, run the following:
> python event_converter.py --src_video /dataset/test --dest_path /dest/folder

Additional options are available in order to:
* Change the number of events that should be accumulated - Option: -n X
* Save converted frames with bboxes as image in a directory during processing - Option: -b /path/to/folder
* Save the resulting TBR array in npy format - Option: -s
* Load a TBR array - Option -l
* Show video of converted frames and bboxes during processing - Option: -v
