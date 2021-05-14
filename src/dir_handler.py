"""
dir_handlers.py: module that manages the input/output directories
"""

import os

def setupDirectories(root_dir: str) -> dict:
    """
    @brief: Setup directories as requested in YOLOV3
            implementation. 
    @param: root_dir - Root directory where the files should be
                       saved. Must be a valid folder
    @return: Dict of useful directories:
            "images": images_path,
            "labels": labels_path,
            "train_file": train_file_path,
            "valid_file": valid_file_path,
            "test_file": test_file_path,
            "list": list_path,
            "completed": completed_file_path,
            "enc": evaluated_enc_path
    """

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

    # Setup classes
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


def getEventList(directory: str) -> list:
    """
    @brief: Check in directory for events and bbox annotations. 
            An event is valid if annotation file 
            with same basename exists.
    @param: directory - Directory where the .dat and .npy files
                        are stored
    @return: List of basenames of valid event files.
    """

    file_list_npy = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and 
                                                                os.path.splitext(os.path.join(directory, file))[1] == '.npy']
    file_list_dat = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and 
                                                                os.path.splitext(os.path.join(directory, file))[1] == '.dat']
    filtered_file_list = []
    for td in file_list_dat:
        if "cut" in td:
            # Avoid files with same name but different 'cut'
            # @TODO: change split policy to handle these files
            print("Skipping video {:s}: filename not compliant".format(td))
            continue

        td_split = td.split('_')
        td = td_split[0] + "_" + td_split[1] + "_" + td_split[2] + "_" + td_split[3]
        for bbox in file_list_npy:
            bbox_split = bbox.split('_')
            bbox = bbox_split[0] + "_" + bbox_split[1] + "_" + bbox_split[2] + "_" + bbox_split[3]
            if td == bbox:
                filtered_file_list.append(td)

    return filtered_file_list