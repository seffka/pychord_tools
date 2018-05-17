from essentia.standard import ChordsDetectionBeats
from pychord_tools.lowLevelFeatures import HPCPChromaEstimator
from pychord_tools.lowLevelFeatures import smooth, SmoothedStartingBeatChromaEstimator
from pychord_tools.commonUtils import convertChordLabels
from pychord_tools.models import loadModel
import os, pychord_tools
import numpy as np

def essentiaChordsByBeats(fileName, beats, chromaEstimator = HPCPChromaEstimator(), smoothingTime = 0.3, chromaPick='starting_beat'):
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


def chordsByBeats(
        fileName, beats, chromaPatternModel, chromaEstimator = HPCPChromaEstimator(),
        segmentChromaEstimator = SmoothedStartingBeatChromaEstimator(smoothingTime=0.6)):
    '''

    :param wav:
    :param beats:
    :return:
    '''
    #chords = ChordsDetectionBeats(hopSize=chromaEstimator.hopSize, chromaPick='interbeat_median')
    beatsChromas = segmentChromaEstimator.getChromaByBeats(beats, chromaEstimator.estimateChroma(fileName))
    syms, strengths = chromaPatternModel.predictExternalLabels(beatsChromas)
    return syms, strengths
