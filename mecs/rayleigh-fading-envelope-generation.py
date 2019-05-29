# Smiths Fading Simulator
# http://www.raymaps.com/index.php/rayleigh-fading-envelope-generation-python/

from numpy import sqrt
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt

# number of rvs
N=20;
# Doppler freq. in Hz
fm=70.0;
# freq spacing
df=(2*fm)/(N-1);
# sampling freq in Hz
fs=1000;
# total number of samples in freq. domain
M=round(fs/df);
#
T=1/df;
Ts=1/fs;

# Generating first Gaussian RV set
# (two seq. of N/2 complex Gaussian rvs)
# bins up to +fm
g=randn(int(N/2))+1j*randn(int(N/2))
# complex conjugate of g to bins up to -fm
gc=np.conj(g)
# reverse gc
gcr=gc[::-1]
# [gcr,gc]
g1=np.concatenate((gcr,g),axis=0)

# Generating second Gaussian RV set
g=randn(int(N/2))+1j*randn(int(N/2))
gc=np.conj(g)
gcr=gc[::-1]
g2=np.concatenate((gcr,g),axis=0)

# Generating the Doppler Spectrum
f=np.arange(-fm, fm+df, df)
S=1.5/(np.pi*fm*sqrt(1-(f/fm)**2))
S[0]=2*S[1]-S[2]
S[-1]=2*S[-2]-S[-3]

# Shaping the RV sequence g1 and taking IFFT
X=g1*sqrt(S);
X=np.concatenate((np.zeros(int((M-N)/2)), X), axis=0)
X=np.concatenate((X, np.zeros(int((M-N)/2))), axis=0)
x=np.abs(np.fft.ifft(X))

# Shaping the RV sequence g2 and taking IFFT
Y=g2*sqrt(S)
Y=np.concatenate((np.zeros(int((M-N)/2)), Y), axis=0)
Y=np.concatenate((Y, np.zeros(int((M-N)/2))), axis=0)
y=np.abs(np.fft.ifft(Y))

# Generating complex envelope
z=x+1j*y
r=np.abs(z)

# Plotting the envelope in the time domain
t=np.arange(0, T, Ts)
plt.plot(t, 10*np.log10(r/np.max(r)),'b')

plt.xlabel('Time(msecs)')
plt.ylabel('Envelope(dB)')
plt.grid(True)
plt.title('Rayleigh Fading')
plt.show()
