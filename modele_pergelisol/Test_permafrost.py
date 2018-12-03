import segyio
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    segydir = "C:/Users/Luc Pelletier/Desktop/Universite/UPIR/modele_pergelisol/BF01_1sec.sgy"

    with segyio.open(segydir, "r", ignore_geometry=True) as segy:
        seis = np.array([segy.trace[trid] for trid in range(segy.tracecount)])
        seis = np.transpose(seis)
        dt = segyio.dt(segy) * 10 ** -3

    clip = 0.1
    aspect = 3
    data = seis
    fig, ax = plt.subplots(1, 1, figsize=[16, 16 / aspect])
    vmax = clip * np.max(data)
    vmin = -vmax
    ax.imshow(seis,
            vmin=vmin, vmax=vmax,
            aspect='auto',
            cmap=plt.get_cmap('Greys'),
            extent=[0, data.shape[1], 0, data.shape[0] * dt])
    ax.set_xlabel("CMP number")
    ax.set_ylabel("Time (ms)")
    ax.set_title('Seismic Section')
    plt.show()

    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        points = tuple(zip(xdata[ind], ydata[ind]))
        print('onpick points:', points)

    fig.canvas.mpl_connect('button_press_event', onpick)

plt.show()