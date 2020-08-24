import numpy as np
import matplotlib.pyplot as plt
import logging
import uuid
from baselines.constants import *

class Channel:
    def __init__(self, channel_type, fading=None, rate=None):
        self.uuid = uuid.uuid4()
        self.channel_type = channel_type
        self.bw = []
        self.max_coverage = []
        self.fading = fading
        if not rate:
            if channel_type==LTE:
                self.up = 75*MBPS
                self.down = 300*MBPS
            elif channel_type==WIFI:
                self.up = 135*MBPS
                self.down = 135*MBPS
            elif channel_type==BT:
                self.up = 22*MBPS
                self.down = 22*MBPS
            elif channel_type==NFC:
                self.up = 212*KBPS
                self.down = 212*KBPS
            else: # channel_type==WIRED:
                self.up = 0.02*GBPS
                self.down = 0.02*GBPS
        else:
            self.up = rate[0]
            self.down = rate[1]

    def get_uuid(self):
        return self.uuid.hex

    def get_channel_type(self):
        return self.channel_type

    def get_rate(self, is_up=True):
        if is_up:
            mean_rate = self.up
        else:
            mean_rate = self.down

        if not self.fading:
            return mean_rate
        elif self.fading in 'rR':
            # mode = np.sqrt(2/np.pi)*mean_rate
            return np.random.rayleigh( np.sqrt(2/np.pi)*mean_rate )
