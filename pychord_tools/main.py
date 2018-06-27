from essentia.standard import ChordsDetectionBeats
import essentia.streaming as esstr
import essentia
from pychord_tools.lowLevelFeatures import HPCPChromaEstimator
from pychord_tools.lowLevelFeatures import smooth, SmoothedStartingBeatChromaEstimator
from pychord_tools.commonUtils import convertChordLabels
from pychord_tools.third_party import NNLSChromaEstimator
import numpy as np
import re
import json
import os

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

def mostLikelySegment(strengths, syms, expected, strngth_threshold = -2.5):
    s = np.argwhere(strengths > strngth_threshold)
    s = s.reshape(s.size)
    if len(s) == 0:
        return None, 0
    start = s[0]
    mismatch=[]
    indices=[]
    for i in range(start, len(syms) - len(expected) + 1):
        mismatch.append(np.sum(expected != syms[i:i+len(expected)]))
        indices.append([i, i+len(expected)])
    mostLikelySegment = indices[np.argmin(mismatch)]
    return mostLikelySegment, np.min(mismatch)


def extractTuningAndDuration(infile):
    chordHopSize = 2048
    frameSize = 8192
    loader = esstr.MonoLoader(filename=infile)
    framecutter = esstr.FrameCutter(hopSize=chordHopSize, frameSize=frameSize)
    windowing = esstr.Windowing(type="blackmanharris62")
    spectrum = esstr.Spectrum()
    spectralpeaks = esstr.SpectralPeaks(orderBy="frequency",
                                  magnitudeThreshold=1e-05,
                                  minFrequency=40,
                                  maxFrequency=5000,
                                  maxPeaks=10000)
    tuning = esstr.TuningFrequency()
    duration = esstr.Duration()
    # use pool to store data
    pool = essentia.Pool()
    # connect algorithms together
    loader.audio >> framecutter.signal
    loader.audio >> duration.signal
    framecutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> spectralpeaks.spectrum
    spectralpeaks.magnitudes >> tuning.magnitudes
    spectralpeaks.frequencies >> tuning.frequencies
    tuning.tuningFrequency >> (pool, 'tonal.tuningFrequency')
    tuning.tuningCents >> (pool, 'tonal.tuningCents')
    duration.duration >> (pool, 'duration.duration')
    # network is ready, run it
    print('Processing audio file...', infile)
    essentia.run(loader)
    return np.average(pool['tonal.tuningFrequency']), pool['duration.duration']

def makeChordStrings(chords):
    """
    Make leadsheet chord string.
    :param chords: array of chords (one per beat)
    :return: string in 4/4 time with given chords and bar lines.
    """
    res = []
    for n in range(0, len(chords) // 4):
        bar = chords[4*n:4*n+4]
        if bar[0] == bar [1] and bar[2] == bar[3]:
            barString = ' '.join(bar[1:3])
        else:
            barString = ' '.join(bar)
        res.append(barString)

    return '|' + '|'.join(res) + '|'

def allDivisors(n, startWith = 1):
    i = startWith
    if n < i:
        return set()
    if n % i == 0:
        return set((i, n / i)) |  allDivisors(n / i, i + 1)
    else:
        return allDivisors(n, i + 1)

def processChords(numerator, blocks, allChords):
    for block in blocks:
        bars = block.split('|')[1:-1]
        for bar in bars:
            chords = [c for c in re.split('\s+', bar) if c != '']
            divisors = allDivisors(numerator)
            if not (len(chords) in divisors):
                raise ValueError("Wrong number of chords in a bar: " + bar)
            multiplier = int(numerator / len(chords))
            newchords = []
            for c in chords:
                newchords.extend([c] * multiplier)
            allChords.extend(newchords)

def processParts(metreNumerator, data, beats, chords, choice='chords'):
    if ('parts' in data.keys()):
        for part in data['parts']:
            processParts(metreNumerator, part, beats, chords, choice)
    else:
        if 'metre' in data :
            metreNumerator = int(data['metre'].split('/')[0])
        beats.extend(data['beats'])
        processChords(metreNumerator, data[choice], chords)

def makeJSONAnnotaton(
        expectedChords,
        audioFile,
        jsonFile,
        beats,
        chromaPatternModel,
        uid = None,
        chromaEstimator=NNLSChromaEstimator(),
        segmentChromaEstimator=SmoothedStartingBeatChromaEstimator(smoothingTime=0.6),
        additionalAttributes = dict()):
    syms, strengths = chordsByBeats(
        audioFile, beats, chromaPatternModel, chromaEstimator=chromaEstimator, segmentChromaEstimator=segmentChromaEstimator)
    segment, mismatch = mostLikelySegment(strengths, syms, expectedChords)
    beats = beats[segment[0]:segment[1]]
    tuning, duration = extractTuningAndDuration(audioFile)
    entry = {}
    if (uid != None):
        entry['uid'] = uid
    entry['title'] = ''
    entry['artist'] = ''
    entry['tuning'] = round(float(tuning), 2)
    entry['metre'] = '4/4'
    entry['duration'] = round(float(duration), 2)
    pathname = os.path.abspath(audioFile)
    entry['sandbox'] = {'path': pathname, 'mismatch': int(mismatch), 'transcriptions': [], 'key': []}
    entry['sandbox'].update(additionalAttributes)
    part = {}
    part['name'] = 'chorus 1'
    part['beats'] = beats.tolist()
    part['chords'] = [makeChordStrings(expectedChords)]
    entry['parts'] = [part]
    with open(jsonFile, 'w') as f:
        json.dump(entry, f, indent=True)
