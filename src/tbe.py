"""
tbe.py: class that manages the Temporal Binary Encoding
"""

import numpy as np

class TemporalBinaryEncoding:
    """
    @brief: Class for Temporal Binary Encoding using N bits
    """

    def __init__(self, N: int, width: int, height: int):
        self.N = N
        self.width = width
        self.height = height
        self._mask = np.ones((self.N, self.height, self.width))

        # Build the mask
        for i in range(N):
            self._mask[i, :, :] = 2 ** i

    def encode(self, mat: np.array) -> np.array:
        """
        @brief: Encode events using binary encoding
        @param: mat
        @return: Encoded frame
        """

        frame = np.sum((mat * self._mask), 0) / (2 ** self.N)
        return frame