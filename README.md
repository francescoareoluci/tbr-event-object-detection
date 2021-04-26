# Temporal Binary Represented Event Object Detection

This repository contains a framework that can be used in order to perform object detection over events acquired from Event Cameras (https://en.wikipedia.org/wiki/Event_camera).
To perform tests, the following technologies and tools have been employed:

* Prophesee’s GEN1 Automotive Detection Dataset. This dataset contains events and their annotated bounding boxes for two classes: pedestrian and vehicles (https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)

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
Note: if you have a NVidia graphic card compatible with CUDA, use the environment_cuda.yml environment in order to use the GPU with YOLO.

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

Starting from the custom folder, the converted frames and bounding box annotations are stored in images and labels folder. Image types are specified in test, train and valid txt files. Already converted events are specified in completed_videos text file. Temporal Binary Represented array can be stored in npy format in evaluated_tbe folder to avoid performing the conversion multiple times.
Moreover, other types of conversion have been implemented in order to compare the results of Temporal Binary Represented event object detection (Polarity and Surface Active Events encoding).

### Conversion

The conversion can be executed using src/event_converted.py file:
> python event_converter.py -h
``` bash
Event to frame converter
usage: event_converter.py [-h] [--use_stored_enc] [--save_enc] [--show_video]
                          [--tbr_bits TBR_BITS] [--src_video SRC_VIDEO]
                          [--dest_path DEST_PATH] [--event_type EVENT_TYPE]
                          [--save_bb_img SAVE_BB_IMG]
                          [--accumulation_time ACCUMULATION_TIME]
                          [--encoder ENCODER]
                          [--export_all_frames EXPORT_ALL_FRAMES]

Convert events to frames and associates bboxes

optional arguments:
  -h, --help            show this help message and exit
  --use_stored_enc, -l  use_stored_enc: instead of evaluates TBR or other
                        encodings, uses pre-evaluated encoded array. Default:
                        false
  --save_enc, -s        save_enc: save the intermediate TBR or other encodings
                        frame array. Default: false
  --show_video, -v      show_video: show video with evaluated TBR frames and
                        their bboxes during processing. Default: false
  --tbr_bits TBR_BITS, -n TBR_BITS
                        tbr_bits: set the number of bits for Temporal Binary
                        Representation. Default: 8
  --src_video SRC_VIDEO, -t SRC_VIDEO
                        src_video: path to event videos
  --dest_path DEST_PATH, -d DEST_PATH
                        dest_path: path where images and bboxes will be stored
  --event_type EVENT_TYPE, -e EVENT_TYPE
                        event_type: specify data type: <train | validation |
                        test>
  --save_bb_img SAVE_BB_IMG, -b SAVE_BB_IMG
                        save_bb_img: save frame with bboxes to path
  --accumulation_time ACCUMULATION_TIME, -a ACCUMULATION_TIME
                        accumulation_time: set the quantization time of events
                        (microseconds). Default: 2500
  --encoder ENCODER, -c ENCODER
                        encoder: set the encoder: <tbe | polarity | sae>.
                        Default: tbe
  --export_all_frames EXPORT_ALL_FRAMES
                        export_all_frames: export all encoded frames from an
                        event video to path

```

For example, to convert events from directory /dataset/train, store results in /dest/folder and label them as train data, run the following:
> python event_converter.py --src_video /dataset/train --dest_path /dest/folder

To convert events from directory /dataset/validation, store results in /dest/folder and label them as validation data, run the following:
> python event_converter.py --src_video /dataset/validation --dest_path /dest/folder

To convert events from directory /dataset/test, store results in /dest/folder and label them as test data, run the following:
> python event_converter.py --src_video /dataset/test --dest_path /dest/folder

Additional options are available in order to:
* Change the number of bits that should be used in TBR - Option: -n X
* Save converted frames with bboxes as image in a directory during processing - Option: -b /path/to/folder
* Save the resulting encoded array in npy format - Option: -s
* Load an encoded array - Option -l
* Show video of converted frames and bboxes during processing - Option: -v
* Change the accumulation time - Option: -a
* Change encoder in order to store frames in other formats - Option: -c <tbe | polarity | sae>

### Training the object detector

Once the dataset has been built, the object detector can be trained.
To setup the detector, modify the gen1.data and gen1-test.data files inside the yolo_config folder with the absolute path of the dataset.
Two configuration files are avaible in order to use the tiny (yolo_config/yolov3-tiny.cfg) or the full (yolo_config/yolov3-gen1.cfg) YOLO implementation.

To train the detector, from the PyTorch-YOLOv3 folder launch the following command:
> python3 train.py --model_def ../yolo_config/yolov3-<tiny | gen1>.cfg --data_config ../yolo_config/gen1.data

To test the detector:

> python3 test.py --model_def ../yolo_config/yolov3-<tiny | gen1>.cfg --data_config ../yolo_config/gen1-test.data --weights_path checkpoints/preferred_ckpt.pth

To use the detector against real images:

> python3 detect.py --image_folder /path/to/images --model_def ../yolo_config/yolov3-<tiny | gen1>.cfg --weights_path checkpoints/preferred_ckpt.pth --class_path /path/to/dataset/data/custom/classes.names

Further informations are available in PyTorch-YOLOv3 repository README.

### Use the Prophesee COCO metric evaluation

In order to use the Prophesee evaluator script, compliant npy array files should be created. These arrays should contain the detected bounding boxes on test images along with their timestamp.
This has be done by forking the YOLOv3 test.py script. The new script creates for each event (frames that belong to the same video) an array of tuples, where each tuple is a detected bounding box. The timestamp associated to a bounding box is approximated as: starting_accumulation_time + (total_accumulation_time / 2), where the total_accumulation_time for Temporal Binary Encoded frames is the accumulation time * number of bits used.
In order to output the npy files enter the src/ folder and run the following command:

> python3 test_gen1.py --model_def ../yolo_config/yolov3-<tiny | gen1>.cfg --data_config ../yolo_config/gen1-test.data --weights_path ../PyTorch-YOLOv3/checkpoints/preferred_ckpt.pth --gen1_output /path/to/output/folder --total_acc_time 20000

Once the npy files have been created, the Prophesee evaluator can be used. Enter the prophesee-automotive-dataset-toolbox repository and run the following command:

> python3 psee_evaluator.py --camera GEN1 /path/to/gen1/events/npy /path/to/detection/events/npy

### Real time object detection

The YOLOv3 detect.py script has been forked in order to create a demo script that can be used to convert an event video to encoded frames and detect objects on them at the same time. In order to do that enter the src/ folder and run the following commands. 

> python3 rt_detection.py --class_path /path/to/dataset/data/custom/classes.names --event_video /path/to/event/dat --model_def ../yolo_config/yolov3-<tiny | gen1>.cfg --show_video --weights_path ../PyTorch-YOLOv3/checkpoints/preferred_ckpt.pth --conf_thres 0.8
