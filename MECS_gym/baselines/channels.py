import numpy as np
from numpy.random import randn, random, standard_normal, lognormal
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import logging
import uuid
from baselines.constants import *

class Channel:
    def __init__(self, channel_type, pathloss=False, lf=False, sf=False, rate=None, op_freq=None):
        self.uuid = uuid.uuid4()
        self.channel_type = channel_type
        self.bw = []
        self.max_coverage = []
        self.pathloss = pathloss # path loss exponent value (gamma) of the path loss model
        self.lf = lf # sigma value of exponential distributin in dB in large scale fading (shadow fading)
        self.sf = sf # m value of Nakagami (m=1 becomes Rayleigh) fading of small scale fading (multipath fading)

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
        else: # user input
            self.up = rate[0]
            self.down = rate[1]
            self.op_freq = op_freq

    def get_uuid(self):
        return self.uuid.hex

    def get_channel_type(self):
        return self.channel_type

    def get_rate(self, is_up=True, dist=1, ref_dist=1):
        # noises = 0
        gain = 1
        if is_up:
            mean_rate = self.up
        else:
            mean_rate = self.down
        
        if self.pathloss and self.channel_type!=WIRED:            
            # Applicable values references of pathloss and shadowing: 
            # 3GPP TR 38.901 or https://en.wikipedia.org/wiki/Log-distance_path_loss_model 
            
            # https://en.wikipedia.org/wiki/Log-distance_path_loss_model (pathloss with large scale shadowing)            
            # Building Type                 Frequency of Transmission	\gamma 	\sigma[dB]
            # Vacuum, infinite space                                    2.0	    0
            # Retail store	                914 MHz	                    2.2	    8.7
            # Grocery store	                914 MHz	                    1.8	    5.2
            # Office with hard partition	1.5 GHz	                    3.0	    7
            # Office with soft partition	900 MHz	                    2.4	    9.6
            # Office with soft partition	1.9 GHz	                    2.6	    14.1
            # Textile or chemical	        1.3 GHz	                    2.0	    3.0
            # Textile or chemical	        4 GHz	                    2.1	    7.0, 9.7
            # Office	                    60 GHz	                    2.2	    3.92
            # Commercial	                60 GHz	                    1.7	    7.9
            
            # https://en.wikipedia.org/wiki/Free-space_path_loss (free space path loss constant)
            # m, KHz ; const = -87.55
            # m, MHz ; const = -27.55
            # m, Hz ; const = -147.55
            # km, GHz ; const = 92.45
            # km, MHz ; 32.44

            # Pathloss part
            # let pathloss value be gamma in the pathloss expression (ex. if vacuum, gamma=self.pathloss=2)
            const = -147.55 # meter, Hz
            fspl = 20*np.log10(dist)+20*np.log10(self.op_freq)+const         
            gain *= (ref_dist/dist)**self.pathloss *10**(-fspl/10)
            
            # Large scale fading part(shadow fading)
            # follows lognormal with sigma 4~8 in dB
            # let lf value be sigma of the normal distribution
            lf = lognormal(0, self.lf)
            gain *= 10**(-lf/10)
            
            # Small scale fading part (multipath fading)
            if self.sf > 0:
                # if self.sf == RAYLEIGH:
                # rayleigh distribution is nakagami distribution with parameter 1
                # else self.sf>1, return general nakagami rv value
                gain *= stats.nakagami.rvs(self.sf)
        
        return mean_rate*gain

def main():
    import pdb; pdb.set_trace()

if __name__=='__main__':
    main()
