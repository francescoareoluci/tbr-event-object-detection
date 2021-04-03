import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_image(frame, bboxes):
    '''
        @brief: show video of encoded frames and their bboxes
                during processing
        @param: frame - A np array containing pixel informations
        @param: bboxes - np array with the bboxes associated to the frame.
                         As loaded from the GEN1 .npy array
    '''

    plt.figure(1)
    plt.clf()
    plt.imshow(frame, animated=True, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()

    ## Get the current reference
    ax = plt.gca()

    ## Create Rectangle boxes
    for b in bboxes:
        if b[5] == 1:
            ## Person
            rect = Rectangle((b[1], b[2]), b[3], b[4], linewidth=2, edgecolor='g', facecolor='none')
        else:
            ## Vehicle
            rect = Rectangle((b[1], b[2]), b[3], b[4], linewidth=2, edgecolor='r', facecolor='none')

        ## Add the patch to the Axes
        ax.add_patch(rect)


def save_bb_image(frame, bboxes, save_path):
    '''
        @brief: save encoded frames with their bboxes
        @param: frame - A np array containing pixel informations
        @param: bboxes - np array with the bboxes associated to the frame.
                         As loaded from the GEN1 .npy array
        @param: save_path - Existing path where the resulting images should
                            be saved
    '''

    plt.imshow(frame, cmap='gray', vmin=0, vmax=1)

    ## Get the current reference
    ax = plt.gca()

    ## Create Rectangle boxes
    for b in bboxes:
        if b[5] == 1:
            ## Person
            rect = Rectangle((b[1], b[2]), b[3], b[4], linewidth=2, edgecolor='g', facecolor='none')
        else:
            ## Vehicle
            rect = Rectangle((b[1], b[2]), b[3], b[4], linewidth=2, edgecolor='r', facecolor='none')

        ## Add the patch to the Axes
        ax.add_patch(rect)
    
    if len(bboxes) != 0:
        plt.savefig(save_path)
        plt.close()
        

def convertBBoxCoords(bbox, image_width, image_height):
    '''
        @brief: Converts top-left starting coordinates to
                rectangle-centered coordinates. Moreover,
                coordinates and size are normalized.
        @param: bbox - A bbox as loaded from the GEN1 .npy array
        @param: image_width
        @param: image_height
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