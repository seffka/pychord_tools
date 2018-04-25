import numpy as np

from pychord_tools import chordsByBeats
from pychord_tools.lowLevelFeatures import HPCPChromaEstimator
from pychord_tools.third_party import NNLSChromaEstimator
from pychord_tools.third_party import rnnBeatSegments

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
    mostLikelySegment = indices[np.argmin(mismatch)]
    accuracy = 100.0 * (len(expected) - np.min(mismatch)) / len(expected)
    return mostLikelySegment, accuracy

def runTest(file, expected, chromaEstimator = HPCPChromaEstimator(), smoothingTime = 0.3, chromaPick='starting_beat'):
    beats = rnnBeatSegments(file)
    syms, strengths = chordsByBeats(file, beats, chromaEstimator = chromaEstimator, smoothingTime=smoothingTime, chromaPick=chromaPick)
    mostLikelySegment, accuracy = mostLikelyIndices(strengths, syms, expected)
    for i in range(mostLikelySegment[0], mostLikelySegment[1]):
        if (syms[i] != expected[i - mostLikelySegment[0]]):
            print(beats[i], ':', expected[i - mostLikelySegment[0]], '!=', syms[i])
    print('Accuracy: ', accuracy)
    return accuracy

berkleeFile = 'demo/audio/berklee_demo.mp3'
berkleeExpected = np.array([
    'G', 'G', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C', 'E', 'E', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C',
    'G', 'G', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C', 'E', 'E', 'A:min', 'A:min', 'Bb',    'Bb',    'F', 'F',
    'G', 'G', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C', 'E', 'E', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C',
    'G', 'G', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C', 'E', 'E', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C',])
sevaFile='demo/audio/seva_demo.mp3'
sevaExpected = np.array(['C', 'C', 'E:min', 'E:min', 'F', 'F', 'A:min', 'A:min', 'G', 'G', 'B', 'B', 'A:min', 'A:min', 'B', 'B',
 'E', 'E', 'E:min', 'E:min', 'B', 'B', 'A:min', 'A:min', 'D', 'D', 'G', 'G', 'D:min', 'D:min', 'C', 'C'])

print('Berklee example (HPCP)')
runTest(berkleeFile, berkleeExpected, smoothingTime = 0.6, chromaPick='starting_beat')
print('Berklee example (NNLS)')
runTest(berkleeFile, berkleeExpected, chromaEstimator = NNLSChromaEstimator(), smoothingTime = 0.6, chromaPick='starting_beat')
print("Seva's example (HPCP)")
runTest(sevaFile, sevaExpected, smoothingTime = 0.6, chromaPick='starting_beat')
print("Seva's example (NNLS)")
runTest(sevaFile, sevaExpected, chromaEstimator = NNLSChromaEstimator(), smoothingTime = 0.6, chromaPick='starting_beat')

"""
accuracies = []
smoothing = []
for i in np.arange(0.0, 4.0, 0.1):
    smoothing.append(i)
    accuracies.append(runTest(sevaFile, sevaExpected, smoothingTime = i, chromaPick='starting_beat'))
print(smoothing)
print(accuracies)
i = np.argmax(np.array(accuracies))
print(smoothing[i], accuracies[i])
"""