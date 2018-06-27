import essentia.standard
import numpy as np
import json
import re

import os.path as path
import pychord_tools.commonUtils as commonUtils
import pychord_tools.cacher as cacher
from pychord_tools.commonUtils import convertChordLabels, ChordSegment

import essentia
import essentia.streaming as esstr

PITCH_CLASS_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
DEGREES=['I', 'IIb', 'II', 'IIIb', 'III', 'IV', 'Vb', 'V', 'VIb', 'VI', 'VIIb', 'VII']

pitches = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}
alt={'b':-1, '#':1}
shortcuts={'maj':'(3,5)', 'min':'(b3,5)', 'dim':'(b3,b5)', 'aug':'(3,#5)', 'maj7':'(3,5,7)',
'min7':'(b3,5,b7)', '7':'(3,5,b7)', 'dim7':'(b3,b5,bb7)', 'hdim7':'(b3,b5,b7)',
'minmaj7':'(b3,5,7)', 'maj6':'(3,5,6)', 'min6':'(b3,5,6)', '9':'(3,5,b7,9)',
'maj9':'(3,5,7,9)', 'min9':'(b3,5,b7,9)', 'sus4':'(4,5)'}
UNCLASSIFIED = 'unclassified'

######################################################################
# Basic structures
######################################################################

class BeatSegments:
    def __init__(self, startTimes, durations):
        self.startTimes = startTimes
        self.durations = durations


class ChromaSegments(BeatSegments):
    def __init__(self, chromas, startTimes, durations):
        self.chromas = chromas
        BeatSegments.__init__(self, startTimes, durations)


class PitchedPattern:
    def __init__(self, kind, pitchClass = None, pitchClassIndex = 0):
        self.kind = kind
        if pitchClass != None :
            self.pitchClassIndex = PITCH_CLASS_NAMES.index(convertChordLabels(pitchClass))
        else:
            self.pitchClassIndex = pitchClassIndex
    def __repr__(self):
        return PITCH_CLASS_NAMES[self.pitchClassIndex] + ':' + self.kind

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.kind == other.kind and\
                   self.pitchClassIndex == other.pitchClassIndex
        return False


class AnnotatedChromaSegments(ChromaSegments):
    def __init__(self, labels, pitches, kinds, chromas, uids, startTimes, durations):
        self.labels = labels
        self.pitches = pitches
        self.kinds = kinds
        self.uids = uids
        ChromaSegments.__init__(self, chromas, startTimes, durations)

    def pitchedPatterns(self):
        if (len(self.kinds) != len(self.pitches)):
            raise ValueError("kinds and pitches vectors need to be equal size.")
        res = np.empty((len(self.pitches)), dtype='object')
        for i in range(len(self.pitches)):
            res[i] = PitchedPattern(self.kinds[i], pitchClassIndex=self.pitches[i])
        return res


######################################################################
# Interfaces
######################################################################

class BeatSegmentsEstimator:
    def estimateBeats(self, audioFileName):
        pass

class ChromaEstimator:
    def __init__(self, frameSize = 16384, hopSize = 2048, sampleRate = 44100):
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.sampleRate = sampleRate

    def estimateChroma(self, audioFileName):
        pass


class SegmentChromaEstimator:
    def __init__(self, frameSize = 16384, hopSize = 2048, sampleRate = 44100):
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.sampleRate = sampleRate

    def fillSegmentsWithChroma(self, beatSegments, chroma):
        pass

    def getChromaByBeats(self, beats, chroma):
        pass


class LabelTranslator:
    def labelToPitchAndKind(self, label):
        pass


class UidAndAudioPathExtractor:
    def uidAndAudioPathName(self, annotationFileName):
        pass

######################################################################

def degreeIndices(degreeNameList):
    return [DEGREES.index(e) for e in degreeNameList]

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
    for i in range(np.size(x,1)):
      if np.size(x, 0) < window_len:
          raise ValueError("Input vector needs to be bigger than window size.")
      if window_len < 3:
          return x
      if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
          raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
      xx = x[:, i]
      s = np.r_[xx[window_len - 1:0:-1], xx, xx[-1:-window_len:-1]]
      # print(len(s))
      if window == 'flat':  # moving average
          w = np.ones(window_len, 'd')
      else:
          w = eval('np.' + window + '(window_len)')
      start = int(window_len / 2)
      end = start + len(xx)
      y[:,i] = np.convolve(w / w.sum(), s, mode='valid')[start:end]
    return y

"""
def smoothedChromaFromAudio(audiofile, sampleRate=44100, stepSize=2048, smoothingTime=1.25):
    result = rawChromaFromAudio(audiofile, sampleRate, stepSize)
    return smooth(result, window_len=int(smoothingTime * sampleRate / stepSize), window='hanning').astype(
                         'float32')
"""
######################################################################

class DefaultUidAndAudioPathExtractor(UidAndAudioPathExtractor):
    def uidAndAudioPathName(self, annotationFileName):
        with open(annotationFileName) as json_file:
            data = json.load(json_file)
            audioPath = str(path.realpath(
                data['sandbox']['path']))
            return audioPath, audioPath

class HPCPChromaEstimator(ChromaEstimator):
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
    '''
    def __init__(
            self,
            tuningFreq = 440, frameSize = 16384, hopSize = 2048, sampleRate = 44100,
            orderBy="magnitude",
            magnitudeThreshold=1e-05,
            minFrequency=40,
            maxFrequency=5000,
            maxPeaks=10000,
            size=12,
            harmonics=8,
            bandPreset=True,
            bandSplitFrequency=500.0,
            weightType="cosine",
            nonLinear=True,
            windowSize=1.0):
        super().__init__(frameSize, hopSize, sampleRate)
        self.tuningFreq = tuningFreq
        self.orderBy = orderBy
        self.magnitudeThreshold = magnitudeThreshold
        self.minFrequency = minFrequency
        self.maxFrequency = maxFrequency
        self.maxPeaks = maxPeaks
        self.size = size
        self.harmonics = harmonics
        self.bandPreset = bandPreset
        self.bandSplitFrequency = bandSplitFrequency
        self.weightType = weightType
        self.nonLinear = nonLinear
        self.windowSize = windowSize

    def estimateChroma(self, audioFileName):
        loader = esstr.MonoLoader(filename=audioFileName)
        framecutter = esstr.FrameCutter(hopSize=self.hopSize, frameSize=self.frameSize)
        windowing = esstr.Windowing(type="blackmanharris62")
        spectrum = esstr.Spectrum()
        spectralpeaks = esstr.SpectralPeaks(orderBy="magnitude",
                                            magnitudeThreshold=1e-05,
                                            minFrequency=40,
                                            maxFrequency=5000,
                                            maxPeaks=10000)
        hpcp = esstr.HPCP(size=12,
                          referenceFrequency=self.tuningFreq,
                          harmonics=8,
                          bandPreset=True,
                          minFrequency=float(40),
                          maxFrequency=float(5000),
                          bandSplitFrequency=500.0,
                          weightType="cosine",
                          nonLinear=True,
                          windowSize=1.0)
        """
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
        """
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
        # roll from 'A' based to 'C' based
        chroma = pool['chroma.hpcp']
        chroma = np.roll(chroma, shift=-3, axis=1)
        return chroma

######################################################################
# Default implementations
######################################################################

def noteToNumber(note):
    pitch=pitches[note[0]]
    if (len(note) >= 2):
        for i in range(1, len(note)):
            pitch = pitch + alt[note[i]]
    return pitch

class Jazz5LabelTranslator(LabelTranslator):
    def labelToPitchAndKind(self, label):
        partsAndBass = label.split('/')
        parts = partsAndBass[0].split(':')
        note = parts[0]
        if (note[0] == 'N'):
            return 9, UNCLASSIFIED
        pitch = noteToNumber(note)
        if (len(parts) == 1):
            kind = 'maj'
        else:
            kind = parts[1].split('/')[0]
        if (kind in shortcuts):
            kind = shortcuts[kind]
        degrees = set(re.sub("[\(\)]", "", kind).split(','))
        # TODO after the dataset is fixed (bass -> pitch class set).
        if (len(partsAndBass) > 1):
            degrees.add(partsAndBass[1])
        if ('3' in degrees):
            if ('b7' in degrees):
                kind = 'dom'
            else:
                kind = 'maj'
        elif ('b3' in degrees):
            if ('b5' in degrees):
                if ('b7' in degrees):
                    kind = 'hdim7'
                else:
                    kind = 'dim'
            else:
                kind = 'min'
        else:
            kind = UNCLASSIFIED
        return pitch, kind


class MajMinLabelTranslator(LabelTranslator):
    MAJ_DEGREES = {'3', '5'}
    MIN_DEGREES = {'b3', '5'}

    def labelToPitchAndKind(self, label):
        partsAndBass = label.split('/')
        parts = partsAndBass[0].split(':')
        note = parts[0]
        if (note[0] == 'N'):
            return 9, 'unclassified'
        pitch = noteToNumber(note)
        if (len(parts) == 1):
            kind = 'maj'
        else:
            kind = parts[1].split('/')[0]
        if (kind in shortcuts):
            kind = shortcuts[kind]
        degrees = set(re.sub("[\(\)]", "", kind).split(','))
        # TODO after the dataset is fixed (bass -> pitch class set).
        if (len(partsAndBass) > 1):
            degrees.add(partsAndBass[1])
        if (len(partsAndBass) > 1):
            degrees.add(partsAndBass[1])
        if (degrees == self.MAJ_DEGREES):
            kind = 'maj'
        elif (degrees == self.MIN_DEGREES):
            kind = 'min'
        else:
            kind = UNCLASSIFIED
        return pitch, kind

#####################################################################
# Loaders/Estimators
######################################################################

class SmoothedStartingBeatChromaEstimator(SegmentChromaEstimator):
    def __init__(self, frameSize = 16384, hopSize = 2048, sampleRate = 44100, smoothingTime = 1.25):
        super().__init__(frameSize, hopSize, sampleRate)
        self.smoothingTime = smoothingTime

    def fill(self, beats, chroma, smoothedChromas):
        for i in range(len(beats)):
            s = int(float(beats[i]) *
                    self.sampleRate / self.hopSize)
            smoothedChromas[i] = chroma[s]

    def fillSegmentsWithChroma(self, segments, chroma):
        chroma = smooth(
            chroma,
            window_len=int(self.smoothingTime * self.sampleRate / self.hopSize),
            window='hanning').astype('float32')
        segments.chromas = np.zeros((len(segments.startTimes), 12), dtype='float32')
        self.fill(segments.startTimes, chroma, segments.chromas)

    def getChromaByBeats(self, beats, chroma):
        chroma = smooth(
            chroma,
            window_len=int(self.smoothingTime * self.sampleRate / self.hopSize),
            window='hanning').astype('float32')
        res = np.zeros((len(beats), 12), dtype='float32')
        self.fill(beats, chroma, res)
        return res

class AnnotatedBeatChromaEstimator:
    def __init__(self,
                 chromaEstimator = HPCPChromaEstimator(),
                 uidAndAudioPathExtractor = DefaultUidAndAudioPathExtractor(),
                 segmentChromaEstimator = SmoothedStartingBeatChromaEstimator(),
                 labelTranslator = MajMinLabelTranslator(),
                 rollToCRoot=True):
        self.chromaEstimator = chromaEstimator
        self.uidAndAudioPathExtractor = uidAndAudioPathExtractor
        self.beatChromaEstimator = segmentChromaEstimator
        self.labelTranslator = labelTranslator
        self.rollToCRoot = rollToCRoot

    def loadChromasForAnnotationFileList(self, fileList):
        res = AnnotatedChromaSegments(
            labels=np.array([], dtype='object'),
            pitches=np.array([], dtype='int'),
            kinds=np.array([], dtype='object'),
            chromas=np.zeros((0,12), dtype='float32'),
            uids=np.array([], dtype='object'),
            startTimes=np.array([], dtype='float32'),
            durations=np.array([], dtype='float32'))
        for file in fileList:
            chunk = self.loadChromasForAnnotationFile(file)
            res.chromas = np.concatenate((res.chromas, chunk.chromas))
            res.labels = np.concatenate((res.labels, chunk.labels))
            res.pitches = np.concatenate((res.pitches, chunk.pitches))
            res.kinds = np.concatenate((res.kinds, chunk.kinds))
            res.uids = np.concatenate((res.uids, chunk.uids))
            res.startTimes = np.concatenate((res.startTimes, chunk.startTimes))
            res.durations = np.concatenate((res.durations, chunk.durations))
        return res

    # returns AnnotatedChromaSegments for the file list
    def loadChromasForAnnotationFileListFile(self, fileListFile):
        return self.loadChromasForAnnotationFileList(
            commonUtils.loadFileList(fileListFile))

    def loadBeatsAndAnnotations(self, jsonFileName, uid):
        with open(jsonFileName) as json_file:
            print(jsonFileName)
            data = json.load(json_file)
            uid = uid
            duration = float(data['duration'])
            metreNumerator = int(data['metre'].split('/')[0])
            allBeats = []
            allChords = []
            commonUtils.processParts(metreNumerator, data, allBeats, allChords, 'chords')
            segments = commonUtils.toBeatChordSegmentList(allBeats[0], duration, allBeats, allChords)
            #
            chromas = None
            labels = np.empty(len(segments), dtype='object')
            pitches = np.empty(len(segments), dtype='int')
            kinds = np.empty(len(segments), dtype='object')
            uids = np.empty(len(segments), dtype='object')
            startTimes = np.zeros(len(segments), dtype='float32')
            durations = np.zeros(len(segments), dtype='float32')
            for i in range(len(segments)):
                pitch, kind = self.labelTranslator.labelToPitchAndKind(segments[i].symbol)
                s = int(float(segments[i].startTime) *
                        self.chromaEstimator.sampleRate / self.chromaEstimator.hopSize)
                e = int(float(segments[i].endTime) *
                        self.chromaEstimator.sampleRate / self.chromaEstimator.hopSize)
                if (s == e):
                    print("empty segment ", segments[i].startTime, segments[i].endTime)
                    raise
                labels[i] = segments[i].symbol
                pitches[i] = pitch
                kinds[i] = kind
                uids[i] = uid
                startTimes[i] = segments[i].startTime
                durations[i] = float(segments[i].endTime) - float(segments[i].startTime)
            return AnnotatedChromaSegments(labels, pitches, kinds, chromas, uids, startTimes, durations)

    def loadChromasForAnnotationFile(self, annotationFileName):
        uid, audioFileName = self.uidAndAudioPathExtractor.uidAndAudioPathName(
            annotationFileName)
        chroma = self.chromaEstimator.estimateChroma(audioFileName)
        annotatedChromaSegments = self.loadBeatsAndAnnotations(annotationFileName, uid)
        self.beatChromaEstimator.fillSegmentsWithChroma(annotatedChromaSegments, chroma)

        if (self.rollToCRoot):
            for i in range(len(annotatedChromaSegments.chromas)):
                shift = 12 - annotatedChromaSegments.pitches[i]
                annotatedChromaSegments.chromas[i] = np.roll(
                    annotatedChromaSegments.chromas[i], shift=shift)
        return annotatedChromaSegments

class BeatChromaEstimator:
    def __init__(self,
                 beats,
                 pitchedPatterns,
                 duration,
                 chromaEstimator = HPCPChromaEstimator(),
                 beatChromaEstimator = SmoothedStartingBeatChromaEstimator(),
                 uid = ""):
        self.beats = np.concatenate((beats, [duration]))
        self.pitchedPatterns = pitchedPatterns
        self.chromaEstimator = chromaEstimator
        self.duration = duration
        self.uid = uid
        self.beatChromaEstimator = beatChromaEstimator

    def loadChromas(self, audioFileName):
        segments = []
        for i in range(len(self.pitchedPatterns)):
            sym = str(self.pitchedPatterns[i])
            segments.append(ChordSegment(self.beats[i], self.beats[i + 1], sym))

        labels = np.empty(len(segments), dtype='object')
        pitches = np.empty(len(segments), dtype='int')
        kinds = np.empty(len(segments), dtype='object')
        uids = np.empty(len(segments), dtype='object')
        startTimes = np.zeros(len(segments), dtype='float32')
        durations = np.zeros(len(segments), dtype='float32')
        for i in range(len(segments)):
            s = int(float(segments[i].startTime) *
                    self.chromaEstimator.sampleRate / self.chromaEstimator.hopSize)
            e = int(float(segments[i].endTime) *
                    self.chromaEstimator.sampleRate / self.chromaEstimator.hopSize)
            if (s == e):
                print("empty segment ", segments[i].startTime, segments[i].endTime)
                raise
            labels[i] = segments[i].symbol
            pitches[i] = self.pitchedPatterns[i].pitchClassIndex
            kinds[i] = self.pitchedPatterns[i].kind
            uids[i] = self.uid
            startTimes[i] = segments[i].startTime
            durations[i] = float(segments[i].endTime) - float(segments[i].startTime)
        annotatedChromaSegments = AnnotatedChromaSegments(
            labels, pitches, kinds, None, uids, startTimes, durations)
        self.beatChromaEstimator.fillSegmentsWithChroma(
            annotatedChromaSegments, self.chromaEstimator.estimateChroma(audioFileName))

        return annotatedChromaSegments

@cacher.memory.cache(ignore=['sampleRate', 'audioSamples'])
def audioDuration(audioFileName, sampleRate=44100, audioSamples=None):
    if (audioSamples is not None):
        return float(len(audioSamples)) / sampleRate
    else:
        audio = essentia.standard.MonoLoader(filename=audioFileName, sampleRate=sampleRate)()
        return float(len(audio)) / sampleRate
