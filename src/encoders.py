import math
import sys
import numpy as np
from tqdm import tqdm

from tbe import TemporalBinaryEncoding


def encode_video_polarity(width, height, video, delta=20000):
    '''
        @brief: Encode video in a sequence of frames using
                Polarity encoding
    '''

    # Each encoded frame will have a start/end timestamp (ms) in order
    # to associate bounding boxes later.
    # Note: If videos are longer than 1 minutes, 16 bits per ts are not sufficient.
    data_type = np.dtype([('startTs', np.uint16), 
                            ('endTs', np.uint16), 
                            ('frame', np.float64, (height, width))])
    
    samplePerVideo = math.ceil(video.total_time() / delta)
    polarity_array = np.zeros(samplePerVideo, dtype=data_type)

    i = 0
    startTimestamp = 0
    endTimestamp = 0

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


def encode_video_tbe(N, width, height, video, encoder, delta=1000):
    '''
        @brief: Encode an event video in a sequence of frame
                using the Temporal Binary Representation
    '''
    
    # Each encoded frame will have a start/end timestamp (ms) in order
    # to associate bounding boxes later.
    # Note: If videos are longer than 1 minutes, 16 bits per ts are not sufficient.
    data_type = np.dtype([('startTs', np.uint16), 
                            ('endTs', np.uint16), 
                            ('frame', np.float64, (height, width))])
    
    samplePerVideo = math.ceil((video.total_time() / delta) / N)
    accumulation_mat = np.zeros((N, height, width))
    tbe_array = np.zeros(samplePerVideo, dtype=data_type)

    i = 0
    j = 0
    startTimestamp = 0
    endTimestamp = 0

    pbar = tqdm(total = samplePerVideo, file = sys.stdout)
    while not video.done:
        i = (i + 1) % N
        # Load next 1ms events from the video
        events = video.load_delta_t(delta)
        f = np.zeros(video.get_size())
        #f = np.zeros((width, height))
        for e in events:
            # Evaluate presence/absence of event for
            # a certain pixel
            f[e['y'], e['x']] = 1

        accumulation_mat[i, ...] = f

        if i == N - 1:
            endTimestamp += N
            tbe = encoder.encode(accumulation_mat)
            tbe_array[j]['startTs'] = startTimestamp
            tbe_array[j]['endTs'] = endTimestamp
            tbe_array[j]['frame'] = tbe
            j += 1
            startTimestamp += N
            pbar.update(1)
    
    pbar.close()
    return tbe_array