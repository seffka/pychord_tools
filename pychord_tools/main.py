from essentia.standard import ChordsDetectionBeats
import re
from pychord_tools.lowLevelFeatures import HPCPChromaEstimator
from pychord_tools.lowLevelFeatures import smooth
import numpy as np

def convertChordLabels(syms) :
    # "minor" to Harte syntax, resolve enharmonicity in "jazz" style.
    res = [re.sub('m$', ':min', s) for s in syms]
    res = [re.sub('Gb', 'F#', s) for s in res]
    res = [re.sub('A#', 'Bb', s) for s in res]
    res = [re.sub('C#', 'Db', s) for s in res]
    res = [re.sub('D#', 'Eb', s) for s in res]
    res = [re.sub('G#', 'Ab', s) for s in res]
    return res

def chordsByBeats(fileName, beats, chromaEstimator = HPCPChromaEstimator(), smoothingTime = 0.3, chromaPick='starting_beat'):
    '''

    :param wav:
    :param beats:
    :return:
    '''
    #chords = ChordsDetectionBeats(hopSize=chromaEstimator.hopSize, chromaPick='interbeat_median')
    chords = ChordsDetectionBeats(hopSize=chromaEstimator.hopSize, chromaPick=chromaPick)
    chroma = smooth(
        chromaEstimator.estimateChroma(fileName),
        window_len=int(smoothingTime * chromaEstimator.sampleRate / chromaEstimator.hopSize),
        window='hanning').\
        astype('float32')
    # roll from 'C' based to 'A' based
    chroma = np.roll(chroma, shift=3, axis=1)
    syms, strengths = chords(chroma, beats)
    return convertChordLabels(syms), strengths
