import json
import os

import essentia
import essentia.streaming as esstr
import numpy as np

from pychord_tools import chordsByBeats
from pychord_tools.lowLevelFeatures import HPCPChromaEstimator

STRENGTH_THRESHOLD = 0.1
def mostLikelyIndices(strengths, syms, expected):
    s = np.argwhere(strengths > STRENGTH_THRESHOLD)
    s = s.reshape(s.size)
    if len(s) == 0:
        return None
    start = s[0]
    mismatch=[]
    indices=[]
    for i in range(start, len(syms) - len(expected) + 1):
        mismatch.append(np.sum(expected != syms[i:i+len(expected)]))
        indices.append([i, i+len(expected)])
    return indices[np.argmin(mismatch)], np.min(mismatch)

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

def makeJSONAnnotaton(expectedChords, audioFile, jsonFile, beats, uid = None, chromaEstimator = HPCPChromaEstimator()):
    syms, strengths = chordsByBeats(
        audioFile, beats, chromaEstimator=chromaEstimator, smoothingTime=0.3, chromaPick='starting_beat')
    segment, mismatch = mostLikelyIndices(strengths, syms, expectedChords)
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
    entry['sandbox'] = {'path': pathname, 'transcriptions': [], 'key': []}
    part = {}
    part['name'] = 'chorus 1'
    part['beats'] = beats.tolist()
    part['chords'] = [makeChordStrings(expectedChords)]
    entry['parts'] = [part]
    with open(jsonFile, 'w') as f:
        json.dump(entry, f, indent=True)
