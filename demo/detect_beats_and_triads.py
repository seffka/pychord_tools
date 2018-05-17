import numpy as np

from pychord_tools.lowLevelFeatures import SmoothedStartingBeatChromaEstimator
from pychord_tools.third_party import NNLSChromaEstimator
from pychord_tools.third_party import rnnBeatSegments
from pychord_tools import chordsByBeats
from pychord_tools.models import loadModel
from pychord_tools.jsonAnnotation import mostLikelySegment
import os, pychord_tools
import essentia.standard as estd

def runTest(
        file,
        beatsFileName,
        expected,
        chromaPatternModel,
        chromaEstimator=NNLSChromaEstimator(),
        segmentChromaEstimator=SmoothedStartingBeatChromaEstimator(smoothingTime=0.6)):
    beats = rnnBeatSegments(file)
    # write audio with beats.
    audio = estd.MonoLoader(filename=file, sampleRate=44100)()
    marker = estd.AudioOnsetsMarker(onsets=beats, type='beep')
    marked_audio = marker(audio)
    estd.MonoWriter(filename=beatsFileName)(marked_audio)

    syms, strengths = chordsByBeats(
        fileName=file,
        beats=beats,
        chromaPatternModel=chromaPatternModel,
        chromaEstimator=chromaEstimator,
        segmentChromaEstimator=segmentChromaEstimator)
    segment, mismatch = mostLikelySegment(strengths, syms, expected)
    for i in range(segment[0], segment[1]):
        if (syms[i] != expected[i - segment[0]]):
            print(beats[i], ':', expected[i - segment[0]], '!=', syms[i])
    print('Mismatch: ', mismatch)
    return mismatch

berkleeFile = 'audio/berklee_demo.mp3'
berkleeExpected = np.array([
    'G', 'G', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C', 'E', 'E', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C',
    'G', 'G', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C', 'E', 'E', 'A:min', 'A:min', 'Bb',    'Bb',    'F', 'F',
    'G', 'G', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C', 'E', 'E', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C',
    'G', 'G', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C', 'E', 'E', 'A:min', 'A:min', 'D:min', 'D:min', 'C', 'C',])
sevaFile='audio/seva_demo.mp3'
sevaExpected = np.array(['C', 'C', 'E:min', 'E:min', 'F', 'F', 'A:min', 'A:min', 'G', 'G', 'B', 'B', 'A:min', 'A:min', 'B', 'B',
 'E', 'E', 'E:min', 'E:min', 'B', 'B', 'A:min', 'A:min', 'D', 'D', 'G', 'G', 'D:min', 'D:min', 'C', 'C'])

print("Seva's example (NNLS)")
model = loadModel(os.path.join(pychord_tools.__path__[0], 'clnm.pkl'))
runTest(sevaFile, 'beats.wav', sevaExpected, model)

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