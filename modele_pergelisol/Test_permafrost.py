import segyio
import numpy as np
import matplotlib.pyplot as plt
import time


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.pid = line.figure.canvas.mpl_connect('button_press_event', self.pick)
        self.did = line.figure.canvas.mpl_connect('key_press_event', self.delete)


    def pick(self, event):
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

    def delete(self, event):
        if event.key != "d": return
        self.xs = self.xs[:-1]
        self.ys = self.ys[:-1]
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

def plot_seis(data, dt, clip=0.04, aspect=3, horizons = []):
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
    for h in horizons:
        ax.plot(h[0], h[1])

    return ax

if __name__ == "__main__":
    segydir = "C:/Users/Luc Pelletier/Desktop/Universite/UPIR/modele_pergelisol/BF01_1sec.sgy"
    segydir = "./BF01_1sec.sgy"

    with segyio.open(segydir, "r", ignore_geometry=True) as segy:
        seis = np.array([segy.trace[trid] for trid in range(segy.tracecount)])
        seis = np.transpose(seis)
        dt = segyio.dt(segy) * 10 ** -3

    #x = plt.ginput(10)
    #print(x)

    horizons = []
    nhorizons =2

    for n in range(nhorizons):
        ax = plot_seis(seis, dt, horizons=horizons)
        line, = ax.plot([], [])  # empty line
        linebuilder = LineBuilder(line)
        plt.show()
        horizons.append([linebuilder.xs, linebuilder.ys])
        time.sleep(0.5)

