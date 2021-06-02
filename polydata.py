import numpy as np
from os.path import join as pjoin

def dataSet(istart,iend):
    x = []
    y = []
    for i in range(istart,iend):
        freq,poy = np.loadtxt(pjoin('data_fit','poynting_rms_run%d'%i+'.txt'),unpack=True)
        x.append(freq)
        y.append(np.log(poy))

    x = np.array(x)
    y = np.array(y)

    x = x.flatten()
    y = y.flatten()
    # transforming the data to include another axis
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    return x,y
