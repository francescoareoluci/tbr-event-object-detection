import os
import sys
import numpy as np

def printGen1BBoxes(directory: str):
    """
    @brief: Print GEN1 bboxes (.npy files)
    @param: directory - Directory where GEN1 .npy files are stored
    """

    file_list_npy = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and 
                                                                os.path.splitext(os.path.join(directory, file))[1] == '.npy']
    for npy_arr in file_list_npy:
        pedestrian_bboxes = 0
        vehicle_bboxes = 0
        
        gen1_bboxes = np.load(directory + "/" + npy_arr)
        for bbox in gen1_bboxes:
            if bbox[5] == 1:
                pedestrian_bboxes += 1
            else:
                vehicle_bboxes += 1

        print("==================================")
        print("Filename: " + npy_arr)
        print("Pedestrian bboxes: " + str(pedestrian_bboxes))
        print("Vehicle bboxes: " + str(vehicle_bboxes))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_gen1_bboxes.py path")
        exit()

    printGen1BBoxes(str(sys.argv[1]))