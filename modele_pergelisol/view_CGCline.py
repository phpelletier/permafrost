#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the CGC seismic line
"""

import segyio
import numpy as np
import matplotlib.pyplot as plt



def plot_seis(data, dt, clip=0.04, aspect=3):
    fig, ax = plt.subplots(1, 1, figsize=[16, 16/aspect])
    vmax = clip * np.max(data)
    vmin = -vmax
    ax.imshow(seis,
              vmin=vmin, vmax=vmax,
              aspect='auto',
              cmap=plt.get_cmap('Greys'),
              extent=[0, data.shape[1], 0, data.shape[0] * dt])
    ax.set_xlabel("CMP number")
    ax.set_ylabel("Time (ms)")
    plt.show()



if __name__ == "__main__":
    segydir = "/Users/gabrielfabien-ouellet/BF01_1sec.sgy"


    with segyio.open(segydir, "r") as segy:
        seis = np.array([segy.trace[trid] for trid in range(segy.tracecount)])
        seis = np.transpose(seis)
        dt = segyio.dt(segy) * 10**-3

    plot_seis(seis, dt=dt)




