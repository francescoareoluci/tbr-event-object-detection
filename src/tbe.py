import numpy as np

class TemporalBinaryEncoding:
    '''
        @brief: Class for Temporal Binary Encoding using N bits
    '''

    def __init__(self, N, width, height):
        self.N = N
        self.width = width
        self.height = height
        self._mask = np.ones((self.N, self.height, self.width))

        # Build the mask
        for i in range(N):
            self._mask[i, :, :] = 2 ** i

    def encode(self, mat):
        '''
            @brief: Encode events using binary encoding
        '''

        frame = np.sum((mat * self._mask), 0) / (2 ** self.N)
        return frame