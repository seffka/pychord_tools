#
# Detects triads with known beats positions.
#
import numpy as np

from pychord_tools import chordsByBeats
from pychord_tools.lowLevelFeatures import HPCPChromaEstimator
from pychord_tools.third_party import NNLSChromaEstimator

majorFile = 'demo/audio/maj35fromE.mp3'
majorBeats = np.array([
        2.68, 3.35, 4.05, 4.74, 5.48, 6.12, 6.81, 7.52, 8.2, 8.84, 9.54, 10.24, 10.9, 11.59, 12.32, 13.03, 13.68, 14.36, 14.98, 15.68, 16.31, 16.96, 17.66, 18.4,
        19.1, 19.79, 20.44, 21.15, 21.8, 22.51, 23.16, 23.87, 24.53, 25.19, 25.9, 26.61, 27.29, 27.98, 28.64, 29.34, 29.98, 30.66, 31.28, 32.01, 32.7, 33.35, 34.05, 34.77,
        35.43, 36.11, 36.74, 37.45, 38.09, 38.81, 39.47, 40.17, 40.84, 41.54, 42.21, 42.91, 43.57, 44.31, 45.0, 45.66, 46.36, 47.08, 47.74, 48.42, 49.06, 49.75, 50.49, 51.11,
        51.79, 52.47, 53.11, 53.82, 54.48, 55.17, 55.9, 56.59, 57.24, 57.91, 58.6, 59.27, 59.97, 60.68, 61.36, 62.04, 62.68, 63.38, 64.01, 64.72, 65.38, 66.06, 66.69
      ], dtype = 'float32')
majorExpected = np.array(['E', 'N', 'A', 'N', 'D', 'N', 'G', 'N', 'C', 'N', 'F', 'N', 'Bb', 'N', 'Eb', 'N', 'Ab', 'N', 'Db', 'N', 'F#', 'N', 'B', 'N',
          'E', 'N', 'A', 'N', 'D', 'N', 'G', 'N', 'C', 'N', 'F', 'N', 'Bb', 'N', 'Eb', 'N', 'Ab', 'N', 'Db', 'N', 'F#', 'N', 'B', 'N',
          'E', 'N', 'A', 'N', 'D', 'N', 'G', 'N', 'C', 'N', 'F', 'N', 'Bb', 'N', 'Eb', 'N', 'Ab', 'N', 'Db', 'N', 'F#', 'N', 'B', 'N',
          'E', 'N', 'A', 'N', 'D', 'N', 'G', 'N', 'C', 'N', 'F', 'N', 'Bb', 'N', 'Eb', 'N', 'Ab', 'N', 'N', 'N', 'N', 'N'])
minorFile = 'demo/audio/min35fromE.mp3'
minorBeats = np.array([2.67, 3.38, 4.05, 4.77, 5.45, 6.17, 6.86, 7.52, 8.13, 8.82, 9.53, 10.23, 10.89, 11.56, 12.22, 12.85, 13.53, 14.25, 14.93, 15.62, 16.29, 16.97, 17.69, 18.37, 19.07, 19.77], dtype = 'float32')
minorExpected = np.array(['E:min', 'N', 'A:min', 'N', 'D:min', 'N', 'G:min', 'N', 'C:min', 'N', 'F:min', 'N', 'Bb:min', 'N', 'Eb:min', 'N', 'Ab:min', 'N', 'Db:min', 'N', 'F#:min', 'N', 'B:min', 'N', 'N'])

def getMatches(file, beats, expected, chromaEstimator = HPCPChromaEstimator(), smoothingTime = 0.3, chromaPick='starting_beat'):
    syms, strengths = chordsByBeats(file,  beats, chromaEstimator, smoothingTime = smoothingTime, chromaPick = chromaPick)
    matches = np.array(syms) == expected

    badIndexes = np.where(matches == False)
    notN = np.where(expected != 'N')
    badIndexes = np.intersect1d(badIndexes, notN)
    print(badIndexes)
    for i in badIndexes:
        print(beats[i], ':', expected[i], '!=', syms[i])
    return syms, strengths, matches

def runTest(chromaEstimator = HPCPChromaEstimator(), smoothingTime = 0.3, chromaPick='starting_beat'):
    majSyms, majStrengths, majMatches = getMatches(
        majorFile,  majorBeats, majorExpected, chromaEstimator, smoothingTime = smoothingTime, chromaPick = chromaPick)
    minSyms, minStrengths, minMatches = getMatches(
        minorFile,  minorBeats, minorExpected, chromaEstimator, smoothingTime = smoothingTime, chromaPick = chromaPick)

    majMatches = majMatches[np.where(majorExpected != 'N')]
    minMatches = minMatches[np.where(minorExpected != 'N')]
    accuracy = 100.0 * (sum(majMatches) + sum(minMatches))/(len(majMatches) + len(minMatches))
    print("Accuracy on non N.C. chords: %.2f %%" % accuracy)
    return accuracy

# Testing different chroma algorithms and smoothing windows. Accuracy:  80.7% -> 100%
print('"Old" essentia approach (HPCP, interbeat median, no preliminary smoothing)')
runTest(smoothingTime = 0.0, chromaPick='interbeat_median')

print('HPCP, interbeat median, smoothing: 0.7')
runTest(smoothingTime = 0.7, chromaPick='interbeat_median')

print('HPCP, starting_beat, smoothing: 0.3')
runTest(smoothingTime = 0.3, chromaPick='starting_beat')

print('NNLS, starting_beat, smoothing: 0.3')
runTest(chromaEstimator = NNLSChromaEstimator(), smoothingTime = 0.3, chromaPick='starting_beat')

print('NNLS, starting_beat, smoothing: 3.0')
runTest(chromaEstimator = NNLSChromaEstimator(), smoothingTime = 3.0, chromaPick='starting_beat')
"""
accuracies = []
smoothing = []
for i in np.arange(0.0, 4.0, 0.1):
    smoothing.append(i)
    accuracies.append(runTest(chromaEstimator = NNLSChromaEstimator(), smoothingTime = i, chromaPick='starting_beat'))
print(smoothing)
print(accuracies)
i = np.argmax(np.array(accuracies))
print(smoothing[i], accuracies[i])
"""
