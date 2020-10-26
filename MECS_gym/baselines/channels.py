import numpy as np
from numpy.random import randn, random, standard_normal
import matplotlib.pyplot as plt
import logging
import uuid
from baselines.constants import *

class Channel:
    def __init__(self, channel_type, fading=0, rate=None, op_freq=None):
        self.uuid = uuid.uuid4()
        self.channel_type = channel_type
        # self.bw = []
        # self.max_coverage = []
        self.fading = fading
        # self.awgn = awgn

        if not rate:
            if channel_type==LTE:
                self.up = 75*MBPS
                self.down = 300*MBPS
                self.op_freq = 2.6*GHZ
            elif channel_type==WIFI1:
                self.up = 135*MBPS
                self.down = 135*MBPS
                self.op_freq = 2.4*GHZ
            elif channel_type==WIFI2:
                self.up = 135*MBPS
                self.down = 135*MBPS
                self.op_freq = 5*GHZ
            elif channel_type==BT:
                self.up = 22*MBPS
                self.down = 22*MBPS
                self.op_freq = 2.4*GHZ
            elif channel_type==NFC:
                self.up = 212*KBPS
                self.down = 212*KBPS
                self.op_freq = 13.56*MHZ
            elif channel_type==NFC:
                self.up = 212*KBPS
                self.down = 212*KBPS
                self.op_freq = 13.56*MHZ
            else: # channel_type==WIRED:
                self.up = 0.02*GBPS
                self.down = 0.02*GBPS
        else:
            self.up = rate[0]
            self.down = rate[1]
            self.op_freq = op_freq

    def get_uuid(self):
        return self.uuid.hex

    def get_channel_type(self):
        return self.channel_type

    def get_rate(self, is_up=True, dist=0):
        # noises = 0
        gain = 1
        if is_up:
            mean_rate = self.up
        else:
            mean_rate = self.down

        if self.fading and self.channel_type!=WIRED:
            gain *= 1 + standard_normal()*np.sqrt(self.fading)
            # return np.random.rayleigh( np.sqrt(2/np.pi)*mean_rate )
        return mean_rate*gain

def main():
    import pdb; pdb.set_trace()

if __name__=='__main__':
    main()
