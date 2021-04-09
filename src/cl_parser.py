"""
cl_parser.py: Command Line Parser, parses user commands
"""

import argparse

class CLParser:
    """
    @brief: Class to manage command line arguments
    """
    
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Convert events to frames and associates bboxes')
        self._parser.add_argument('--use_stored_enc', '-l', action='count', default=0,
                        help='use_stored_enc: instead of evaluates TBR or other encodings, uses pre-evaluated encoded array. Default: false')
        self._parser.add_argument('--save_enc', '-s', action='count', default=0,
                        help='save_enc: save the intermediate TBR or other encodings frame array. Default: false')
        self._parser.add_argument('--show_video', '-v', action='count', default=0,
                        help='show_video: show video with evaluated TBR frames and their bboxes during processing. Default: false')
        self._parser.add_argument('--tbr_bits', '-n', type=int, default=8,
                        help='tbr_bits: set the number of bits for Temporal Binary Representation. Default: 8')
        self._parser.add_argument('--src_video', '-t', type=str, nargs=1,
                        help='src_video: path to event videos')
        self._parser.add_argument('--dest_path', '-d', type=str, nargs=1,
                        help='dest_path: path where images and bboxes will be stored')
        self._parser.add_argument('--event_type', '-e', type=str, nargs=1,
                        help='event_type: specify data type: <train | validation | test>')
        self._parser.add_argument('--save_bb_img', '-b', type=str, nargs=1,
                        help='save_bb_img: save frame with bboxes to path')
        self._parser.add_argument('--accumulation_time', '-a', type=int, default=2500,
                        help='accumulation_time: set the quantization time of events (microseconds). Default: 2500')
        self._parser.add_argument('--encoder', '-c', type=str, nargs=1,
                        help='encoder: set the encoder: <tbe | polarity | sae>. Default: tbe')

    def parse(self):
        """
        @brief: parse the command line arguments
        @return: parsed arguments
        """

        return self._parser.parse_args()