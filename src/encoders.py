"""
encoders.py: encode frames from event videos using
Temporal Binary, Polarity, Surface Active Events encodings
"""

import math
import sys
import numpy as np
from tqdm import tqdm

from tbe import TemporalBinaryEncoding
import sys
sys.path.insert(0, '../prophesee-automotive-dataset-toolbox/')
from src.io.psee_loader import PSEELoader


def encode_video_sae(width: int, 
                    height: int, 
                    video: PSEELoader, 
                    delta: int = 2500) -> np.array:
    """
    @brief: Encode video in a sequence of frames using
            Surface Active Event (SAE) encoding
    @param: width
    @param: height
    @param: video - loaded from PSEELoader
    @param: delta - accumulation time
    @return: encoded frames as a Numpy array with the following data type:
                [('startTs', np.uint16), 
                ('endTs', np.uint16), 
                ('frame', np.float32, (height, width))]
    """

    print("Starting Surface Active Event encoding...")

    # Each encoded frame will have a start/end timestamp (ms) in order
    # to associate bounding boxes later.
    # Note: If videos are longer than 1 minutes, 16 bits per ts are not sufficient.
    data_type = np.dtype([('startTs', np.uint16), 
                            ('endTs', np.uint16), 
                            ('frame', np.float32, (height, width))])
    
    samplePerVideo = math.ceil(video.total_time() / delta)
    sae_array = np.zeros(samplePerVideo, dtype=data_type)

    i = 0
    startTimestamp = 0  # milliseconds
    endTimestamp = 0    # milliseconds

    pbar = tqdm(total=samplePerVideo, file=sys.stdout)
    while not video.done:
        events = video.load_delta_t(delta)
        f = np.zeros(video.get_size())
        for e in events:
            # Evaluate polarity of an event 
            # for a certain pixel
            t_p = e['t']                    # microseconds
            t_0 = startTimestamp * 1000     # microseconds
            f[e['y'], e['x']] = 255 * ((t_p - t_0) / delta)

        endTimestamp += delta / 1000
        sae_array[i]['startTs'] = startTimestamp
        sae_array[i]['endTs'] = endTimestamp
        sae_array[i]['frame'] = f
        startTimestamp += delta / 1000
        i += 1
        
        pbar.update(1)

    pbar.close()
    return sae_array


def encode_video_polarity(width: int, 
                        height: int, 
                        video: PSEELoader, 
                        delta: int = 2500) -> np.array:
    """
    @brief: Encode video in a sequence of frames using
            Polarity encoding
    @param: width
    @param: height
    @param: video - loaded from PSEELoader
    @param: delta - accumulation time
    @return: encoded frames as a Numpy array with the following data type:
             [('startTs', np.uint16), 
              ('endTs', np.uint16), 
              ('frame', np.float32, (height, width))]
    """

    print("Starting Polarity Encoding...")

    # Each encoded frame will have a start/end timestamp (ms) in order
    # to associate bounding boxes later.
    # Note: If videos are longer than 1 minutes, 16 bits per ts are not sufficient.
    data_type = np.dtype([('startTs', np.uint16), 
                            ('endTs', np.uint16), 
                            ('frame', np.float32, (height, width))])
    
    samplePerVideo = math.ceil(video.total_time() / delta)
    polarity_array = np.zeros(samplePerVideo, dtype=data_type)

    i = 0
    startTimestamp = 0   # milliseconds
    endTimestamp = 0     # milliseconds

    pbar = tqdm(total=samplePerVideo, file=sys.stdout)
    while not video.done:
        events = video.load_delta_t(delta)
        f = np.full(video.get_size(), 0.5)
        for e in events:
            # Evaluate polarity of an event 
            # for a certain pixel
            if e['p'] == 1:
                f[e['y'], e['x']] = 1
            else:
                f[e['y'], e['x']] = 0

        endTimestamp += delta / 1000
        polarity_array[i]['startTs'] = startTimestamp
        polarity_array[i]['endTs'] = endTimestamp
        polarity_array[i]['frame'] = f
        startTimestamp += delta / 1000
        i += 1
        
        pbar.update(1)

    pbar.close()
    return polarity_array


def encode_video_tbe(N: int, 
                    width: int, 
                    height: int, 
                    video: PSEELoader, 
                    encoder: TemporalBinaryEncoding, 
                    delta: int = 2500) -> np.array:
    """
    @brief: Encode an event video in a sequence of frame
            using the Temporal Binary Representation
    @param: N - number of bits to be used
    @param: width
    @param: height
    @param: video - loaded from PSEELoader
    @param: encoded - TBE encoder
    @param: delta - accumulation time
    @return: encoded frames as a Numpy array with the following data type:
             [('startTs', np.uint16), 
              ('endTs', np.uint16), 
              ('frame', np.float32, (height, width))]
    """
    
    print("Starting Temporal Binary Encoding...")

    # Each encoded frame will have a start/end timestamp (ms) in order
    # to associate bounding boxes later.
    # Note: If videos are longer than 1 minutes, 16 bits per ts are not sufficient.
    data_type = np.dtype([('startTs', np.uint16), 
                            ('endTs', np.uint16), 
                            ('frame', np.float32, (height, width))])
    
    samplePerVideo = math.ceil((video.total_time() / delta) / N)
    accumulation_mat = np.zeros((N, height, width))
    tbe_array = np.zeros(samplePerVideo, dtype=data_type)

    i = 0
    j = 0
    startTimestamp = 0  # milliseconds
    endTimestamp = 0    # milliseconds

    pbar = tqdm(total = samplePerVideo, file = sys.stdout)
    while not video.done:
        i = (i + 1) % N
        # Load next 1ms events from the video
        events = video.load_delta_t(delta)
        f = np.zeros(video.get_size())
        for e in events:
            # Evaluate presence/absence of event for
            # a certain pixel
            f[e['y'], e['x']] = 1

        accumulation_mat[i, ...] = f

        if i == N - 1:
            endTimestamp += (N * delta) / 1000
            tbe = encoder.encode(accumulation_mat)
            tbe_array[j]['startTs'] = startTimestamp
            tbe_array[j]['endTs'] = endTimestamp
            tbe_array[j]['frame'] = tbe
            j += 1
            startTimestamp += (N * delta) / 1000
            pbar.update(1)
    
    pbar.close()
    return tbe_array


def get_frame_BB(frame: np.array, BB_array: np.array) -> np.array:
    """
    @brief: Associates to an encoded video frame
            a list of bounding boxes with timestamp included in 
            start/end timestamp of the frame. 
    @param: frame - Encoded frame with the following structure:
                    [{'startTs': startTs}, {'endTs': endTs}, {'frame': frame}]
                    (i.e. as the one returned from the encoders fuctions)
    @param: BB_array - Bounding Boxes array, 
                       loaded from the GEN1 .npy arrays
    @return: The associated BBoxes.
    """

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