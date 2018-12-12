#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.signal import gaussian
import matplotlib.pyplot as plt



class ModelParameters(object):
    """
    This class contains all model parameters needed to generate random models
    and seismic data
    """

    def __init__(self):
        self.layer_dh_min = 50  # minimum thickness of a layer (in grid cells)
        self.layer_num_min = 5  # minimum number of layers
        self.num_layers = 10  # Fix the number of layers if not 0

        self.vp_default = 1500.0  # default value of vp (in m/s)
        self.vs_default = 0.0  # default value of vs (in m/s)
        self.rho_default = 2000  # default value of rho (in kg/m3)
        self.vp_min = 2000.0  # maximum value of vp (in m/s)
        self.vp_max = 4000.0  # minimum value of vp (in m/s)
        self.dvmax = 2000  # Maximum velocity difference between 2 layers
        self.marine = False  # if true, first layer will be at water velocity
        self.velwater = 1500  # mean velocity of water
        self.d_velwater = 30  # amplitude of random variation of water velocity
        self.water_depth = 3000  # mean water depth (m)
        self.dwater_depth = 2000  # maximum amplitude of water depth variations
        self.vp_trend_min = 0    # Minimum trend for velocity variation in z
        self.vp_trend_max = 0    # Maximum trend for velocity variation in z


        self.rho_var = False
        self.rho_min = 2000.0  # maximum value of rho
        self.rho_max = 3500.0  # minimum value of rho
        self.drhomax = 800  # Maximum velocity difference between 2 layers

        self.angle_0 = True     # It true, first layer angle is 0
        self.angle_max = 15  # Maximum dip of a layer
        self.dangle_max = 5  # Maximum dip difference between two adjacent layers
        self.max_deform_freq = 0  # Max frequency of the layer boundary function
        self.min_deform_freq = 0  # Min frequency of the layer boundary function
        self.amp_max = 25  # Maximum amplitude of boundary deformllations
        self.max_deform_nfreq = 20  # Maximum nb of frequencies of boundary
        self.prob_deform_change = 0.3  # Probability that a boundary shape will
        # change between two lahyers
        self.max_texture = 0.15  # Add random noise two a layer (% or velocity)
        self.texture_xrange = 1  # Range of the filter in x for texture creation
        self.texture_zrange = 1  # Range of the filter in z for texture creation



        self.NX = 256                      # number of grid cells in X direction
        self.NZ = 256                      # number of grid cells in Z direction
        self.dh = 10.0          # grid spacing in X, Y, Z directions (in meters)
        self.fs = False         # whether free surface is turned on the top face
        self.Npad = 16           # number of padding cells of absorbing boundary
        self.NT = 2048                                   # number of times steps
        self.dt = 0.0009             # time sampling for seismogram (in seconds)
        self.peak_freq = 10.0       # peak frequency of input wavelet (in Hertz)
        self.df = 2                # Frequency of source peak_freq +- random(df)
        self.tdelay = 2.0 / (self.peak_freq - self.df)     # delay of the source
        self.resampling = 10                 # Resampling of the shots time axis
        self.source_depth = (self.Npad + 2) * self.dh     # depth of sources (m)
        self.receiver_depth = (self.Npad + 2) * self.dh # depth of receivers (m)
        self.dg = 2                           # Receiver interval in grid points
        self.ds = 2                                    # Source interval (in 2D)
        self.gmin = None    # Minimum position of receivers (-1 = minimum of grid)
        self.gmax = None    # Maximum position of receivers (-1 = maximum of grid)
        self.minoffset = 0
        self.sourcetype = 100       # integer used by SeisCL for pressure source

def texture_1lay(NZ, NX, lz=2, lx=2):
    """
    Created a random model with bandwidth limited noise.

    @params:
    NZ (int): Number of cells in Z
    NX (int): Number of cells in X
    lz (int): High frequency cut-off size in z
    lx (int): High frequency cut-off size in x
    @returns:

    """

    noise = np.fft.fft2(np.random.random([NZ, NX]))
    noise[0, :] = 0
    noise[:, 0] = 0
    noise[-1, :] = 0
    noise[:, -1] = 0

    iz = lz
    ix = lx
    maskz = gaussian(NZ, iz)
    maskz = np.roll(maskz, [int(NZ / 2), 0])
    maskx = gaussian(NX, ix)
    maskx = np.roll(maskx, [int(NX / 2), 0])
    noise = noise * np.reshape(maskz, [-1, 1])
    noise *= maskx
    noise = np.real(np.fft.ifft2(noise))
    noise = noise / np.max(noise)

    return noise


def create_deformation(max_deform_freq, min_deform_freq,
                       amp_max, max_deform_nfreq, Nmax):
    nfreqs = np.random.randint(max_deform_nfreq)
    freqs = np.random.rand(nfreqs) * (
            max_deform_freq - min_deform_freq) + min_deform_freq
    phases = np.random.rand(nfreqs) * np.pi * 2
    amps = np.random.rand(nfreqs)
    x = np.arange(0, Nmax)
    deform = np.zeros(Nmax)
    for ii in range(nfreqs):
        deform += amps[ii] * np.sin(freqs[ii] * x + phases[ii])

    ddeform = np.max(deform)
    if ddeform > 0:
        deform = deform / ddeform * amp_max * np.random.rand()

    return deform

def generate_random_2Dlayered(pars, seed=None):
    """
    This method generates a random 2D model, with parameters given in pars.
    Important parameters are:
        Model size:
        -pars.NX : Number of grid cells in X
        -pars.NZ : Number of grid cells in Z
        -pars.dh : Cell size in meters

        Number of layers:
        -pars.num_layers : Minimum number of layers contained in the model
        -pars.layer_dh_min : Minimum thickness of a layer (in grid cell)
        -pars.source_depth: Depth in meters of the source. Velocity above the
                            source is kept constant.

        Layers dip
        -pars.angle_max: Maximum dip of a layer in degrees
        -pars.dangle_max: Maximum dip difference between adjacent layers

        Model velocity
        -pars.vp_max: Maximum Vp velocity
        -pars.vp_min: Minimum Vp velocity
        -pars.dvmax: Maximum velocity difference of two adajcent layers

        Marine survey parameters
        -pars.marine: If True, first layer is water
        -pars.velwater: water velocity
        -pars.d_velwater: variance of water velocity
        -pars.water_depth: Mean water depth
        -pars.dwater_depth: variance of water depth

        Non planar layers
        pars.max_deform_freq: Maximum spatial frequency (1/m) of a layer interface
        pars.min_deform_freq: Minimum spatial frequency (1/m) of a layer interface
        pars.amp_max: Minimum amplitude of the ondulation of the layer interface
        pars.max_deform_nfreq: Maximum number of frequencies of the interface

        Add texture in layers
        pars.texture_zrange
        pars.texture_xrange
        pars.max_texture

    @params:
    pars (str)   : A ModelParameters class containing parameters
                   for model creation.
    seed (str)   : The seed for the random number generator

    @returns:
    vp, vs, rho, vels, layers, angles
    vp (numpy.ndarray)  :  An array containing the vp model
    vs (numpy.ndarray)  :  An array containing the vs model (0 for the moment)
    rho (numpy.ndarray)  :  An array containing the density model
                            (2000 for the moment)
    vels (numpy.ndarray)  : 1D array containing the mean velocity of each layer
    layers (numpy.ndarray)  : 1D array containing the mean thickness of each layer,
                            at the center of the model
    angles (numpy.ndarray)  : 1D array containing slope of each layer
    """

    if seed is not None:
        np.random.seed(seed)

    # Determine the minimum and maximum number of layers
    nmin = pars.layer_dh_min
    nmax = int(pars.NZ / pars.layer_num_min)
    nlmax = int(pars.NZ / nmin)
    nlmin = int(pars.NZ / nmax)
    if pars.num_layers == 0:
        if nlmin < nlmax:
            n_layers = np.random.choice(range(nlmin, nlmax))
        else:
            n_layers = nmin
    else:
        n_layers = int(np.clip(pars.num_layers, nlmin, nlmax))

    # Generate a random number of layers with random thicknesses
    NZ = pars.NZ
    NX = pars.NX
    dh = pars.dh
    top_min = int(pars.source_depth / dh + 2 * pars.layer_dh_min)
    layers = (nmin + np.random.rand(n_layers) * (nmax - nmin)).astype(np.int)
    tops = np.cumsum(layers)
    ntos = np.sum(layers[tops <= top_min])
    if ntos > 0:
        layers = np.concatenate([[ntos], layers[tops > top_min]])

    # Generate random angles for each layer
    n_angles = len(layers)
    angles = np.zeros(layers.shape)
    if not pars. angle_0:
        angles[1] = -pars.angle_max + np.random.rand() * 2 * pars.angle_max
    for ii in range(2, n_angles):
        angles[ii] = angles[ii - 1] + (
                2.0 * np.random.rand() - 1.0) * pars.dangle_max
        if np.abs(angles[ii]) > pars.angle_max:
            angles[ii] = np.sign(angles[ii]) * pars.angle_max

    # Generate a random velocity for each layer. Velocities are somewhat biased
    # to increase in depth
    vels = (np.random.rand(len(layers))-0.5) * pars.dvmax
    ramp = np.abs(pars.vp_trend_max - pars.vp_trend_min) * np.random.rand() + pars.vp_trend_min
    vels = vels + np.linspace(pars.vp_min, pars.vp_min + ramp, vels.shape[0])
    vels[vels > pars.vp_max] = pars.vp_max
    vels[vels < pars.vp_min] = pars.vp_min
    if pars.marine:
        vels[0] = pars.velwater + (np.random.rand() - 0.5) * 2 * pars.d_velwater
        layers[0] = int(pars.water_depth / pars.dh +
                        (
                                np.random.rand() - 0.5) * 2 * pars.dwater_depth / pars.dh)

    # Generate the 2D model, from top layers to bottom
    vel2d = np.zeros([NZ, NX]) + vels[0]
    layers2d = np.zeros([NZ, NX])
    tops = np.cumsum(layers)
    deform = create_deformation(pars.max_deform_freq,
                              pars.min_deform_freq,
                              pars.amp_max,
                              pars.max_deform_nfreq, NX)
    texture = texture_1lay(2 * NZ,
                           NX,
                           lz=pars.texture_zrange,
                           lx=pars.texture_xrange)
    for ii in range(0, len(layers) - 1):
        if np.random.rand() < pars.prob_deform_change:
            deform += create_deformation(pars.max_deform_freq,
                                       pars.min_deform_freq,
                                       pars.amp_max,
                                       pars.max_deform_nfreq, NX)

        texture = texture / np.max(texture) * (
                np.random.rand() + 0.001) * pars.max_texture * vels[ii + 1]
        for jj in range(0, NX):
            # depth of the layer at location x
            dz = int((np.tan(angles[ii + 1] / 360 * 2 * np.pi) * (
                    jj - NX / 2) * dh) / dh)
            # add deformation component
            if pars.amp_max > 0:
                dz = int(dz + deform[jj])
            # Check if the interface is inside the model
            if 0 < tops[ii] + dz < NZ:
                vel2d[tops[ii] + dz:, jj] = vels[ii + 1]
                layers2d[tops[ii] + dz:, jj] = ii
                if not (pars.marine and ii == 0) and pars.max_texture > 0:
                    vel2d[tops[ii] + dz:, jj] += texture[tops[ii]:NZ - dz, jj]
            elif tops[ii] + dz <= 0:
                vel2d[:, jj] = vels[ii + 1]
                layers2d[:, jj] = ii
                if not (pars.marine and ii == 0) and pars.max_texture > 0:
                    vel2d[:, jj] += texture[:vel2d.shape[0], jj]

    # Output the 2D model
    vel2d[vel2d > pars.vp_max] = pars.vp_max
    vel2d[vel2d < pars.vp_min] = pars.vp_min
    vp = vel2d
    vs = vp * 0
    rho = vp * 0 + 2000

    return vp, vs, rho, vels, layers, angles, layers2d


if __name__ == "__main__":

    pars = ModelParameters()
    pars.NX = 1500
    pars.NZ = 250

    pars.layer_dh_min = 8  # minimum thickness of a layer (in grid cells)
    pars.layer_num_min = 8  # minimum number of layers
    pars.num_layers = 0  # Fix the number of layers if not 0

    pars.angle_max = 12  # Maximum dip of a layer
    pars.dangle_max = 3  # Maximum dip difference between two adjacent layers
    pars.max_deform_freq = 0.08  # Max frequency of the layer boundary function
    pars.min_deform_freq = 0.0001  # Min frequency of the layer boundary function
    pars.amp_max = 12  # Maximum amplitude of boundary deformations
    pars.max_deform_nfreq = 40  # Maximum nb of frequencies of boundary
    pars.prob_deform_change = 0.4  # Probability that a boundary shape will

    pars.dvmax = 600
    pars.vp_trend_max = 400
    pars.vp_trend_min = 1000


    pars.texture_xrange = 10
    pars.texture_zrange = 1.95 * pars.NZ
    pars.max_texture = 0.10

    for ii in range(1,20):
        vp, vs, rho, vels, layers, angles, layers2d = generate_random_2Dlayered(pars, seed=ii) #8, 25

        plt.imshow(vp)
        plt.xlabel(str(ii))
        plt.show()
        #plt.imshow(layers2d)
        #plt.show()