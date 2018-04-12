import numpy as np
from essentia.standard import ChordsDetectionBeats
import essentia
import essentia.standard as ess
import essentia.streaming as esstr
import re


def smooth(x, window_len=11, window='hanning'):
    '''Smooth the data using a window with requested size.
    Borrowed from http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x : numpy.array(dtype=float)
        2-dimensional data (smoothing is performaed along axis 0
        for each component along axis 1)
    window_len: int
        the dimension of the smoothing window; should be an odd integer
    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    Returns
    -------
    x: numpy.array
            the smoothed signal
    '''
    y = np.zeros(x.shape)
    for i in range(np.size(x, 1)):
        if np.size(x, 0) < window_len:
            raise (ValueError, "Input vector needs to be bigger than window size.")
        if window_len < 3:
            return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        xx = x[:, i]
        s = np.r_[xx[window_len - 1:0:-1], xx, xx[-1:-window_len:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        start = int(window_len / 2)
        end = start + len(xx)
        y[:, i] = np.convolve(w / w.sum(), s, mode='valid')[start:end]
    return y

SMOOTHING_TIME = 0.2
def smoothChroma(chroma, hopSize = 4096, fs = 44100):
    '''
    Smooth chroma features by convolving each of 12 chroma sequences with hanning window.
    '''
    return smooth(chroma, window_len=int(SMOOTHING_TIME * fs / hopSize), window='hanning').astype(
                         'float32')

def loadChroma(filename, frameSize = 16384, tuningFreq=440, hopSize = 4096):
    '''
    Extract HPCP chroma features with essentia
    Parameters
    ----------
    filename: str
        audio file name
    frameSize : int
        Analysis frame size (samples)
    tuningFreq : float
        tuning frequency (Hz)
    hopSize : int
        Hop size (in samples)
    Returns
    -------
    chroma: numpy.array
    spectra: numpy.array
    '''
    loader = esstr.MonoLoader(filename = filename)
    framecutter = esstr.FrameCutter(hopSize=hopSize, frameSize=frameSize)
    windowing = esstr.Windowing(type="blackmanharris62")
    spectrum = esstr.Spectrum()
    spectralpeaks = esstr.SpectralPeaks(orderBy="magnitude",
                                      magnitudeThreshold=1e-05,
                                      minFrequency=40,
                                      maxFrequency=5000,
                                      maxPeaks=10000)
    hpcp = esstr.HPCP(
        size=12,
        referenceFrequency = tuningFreq,
        harmonics = 8,
        bandPreset = True,
        minFrequency = 40.0,
        maxFrequency = 5000.0,
        bandSplitFrequency = 250.0,
        weightType = "cosine",
        nonLinear = False,
        windowSize = 1.0)
    pool = essentia.Pool()
    # connect algorithms together
    loader.audio >> framecutter.signal
    framecutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> spectralpeaks.spectrum
    spectrum.spectrum >> (pool, 'spectrum.magnitude')
    spectralpeaks.magnitudes >> hpcp.magnitudes
    spectralpeaks.frequencies >> hpcp.frequencies
    hpcp.hpcp >> (pool, 'chroma.hpcp')

    essentia.run(loader)
    chroma = pool['chroma.hpcp']
    #spectra = pool['spectrum.magnitude']
    chroma = smoothChroma(chroma)
    return chroma

def convertChordLabels(syms) :
    return [re.sub('m$', ':min', s) for s in syms]

def chordsByBeats(fileName, beats):
    '''

    :param wav:
    :param beats:
    :return:
    '''
    #chords = ChordsDetectionBeats(hopSize=4096, chromaPick='interbeat_median')
    chords = ChordsDetectionBeats(hopSize=4096, chromaPick='first_beat')
    chroma = loadChroma(fileName, hopSize=4096)
    syms, strengths = chords(chroma, beats)
    return convertChordLabels(syms), strengths
