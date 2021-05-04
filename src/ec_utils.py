"""
ec_utils.py: Event Converter Utils - Utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator


def show_image(frame: np.array, bboxes: np.array):
    """
    @brief: show video of encoded frames and their bboxes
            during processing
    @param: frame - A np array containing pixel informations
    @param: bboxes - np array with the bboxes associated to the frame.
                     As loaded from the GEN1 .npy array
    """

    plt.figure(1)
    plt.clf()
    plt.axis("off")
    plt.imshow(frame, animated=True, cmap='gray', vmin=0, vmax=1)
    #plt.colorbar()

    # Get the current reference
    ax = plt.gca()

    # Create Rectangle boxes
    for b in bboxes:
        predicted_class = b[5]
        x = b[1]
        y = b[2]
        w = b[3]
        h = b[4]
        bbox_color = 'g' if predicted_class == 1 else 'r'
        # Create Rectangle
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=bbox_color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add label
        plt.text(
            b[1],
            b[2],
            s='Pedestrian' if predicted_class == 1 else 'Vehicle',
            color="white",
            verticalalignment="top",
            bbox={"color": bbox_color, "pad": 0},
        )


def save_bb_image(frame: np.array, 
                  bboxes: np.array, 
                  save_path: str, 
                  only_detection: bool = True):
    """
    @brief: save encoded frames with their bboxes
    @param: frame - A np array containing pixel informations
    @param: bboxes - np array with the bboxes associated to the frame.
                     As loaded from the GEN1 .npy array
    @param: save_path - Existing path where the resulting images should
                        be saved
    """

    plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    # Get the current reference
    ax = plt.gca()

    # Create Rectangle boxes
    for b in bboxes:
        predicted_class = b[5]
        x = b[1]
        y = b[2]
        w = b[3]
        h = b[4]
        bbox_color = 'g' if predicted_class == 1 else 'r'
        # Create Rectangle
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=bbox_color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add label
        plt.text(
            b[1],
            b[2],
            s='Pedestrian' if predicted_class == 1 else 'Vehicle',
            color="white",
            verticalalignment="top",
            bbox={"color": bbox_color, "pad": 0},
        )
    
    if not only_detection:
        # Save all frames if requested
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    elif len(bboxes) != 0:
        # Save only frames with bboxes associated
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        

def convertBBoxCoords(bbox: np.array, image_width: int, image_height: int) -> np.array:
    """
    @brief: Converts top-left starting coordinates to
            rectangle-centered coordinates. Moreover,
            coordinates and size are normalized.
    @param: bbox - A bbox as loaded from the GEN1 .npy array
    @param: image_width
    @param: image_height
    @return: np array compliant to YOLOV3 implementation.
    """

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