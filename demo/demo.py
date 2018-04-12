from pychord_detection import chordsByBeats
#from pychord_detection import smooth
#from pychord_detection import convertChordLabels
import numpy as np
#import essentia
#import vamp

"""
def loadChroma(fileName, hopSize=2048):
    mywindow = np.array([0.001769, 0.015848, 0.043608, 0.084265, 0.136670, 0.199341, 0.270509, 0.348162, 0.430105, 0.514023,
                0.597545, 0.678311, 0.754038, 0.822586, 0.882019, 0.930656, 0.967124, 0.990393, 0.999803, 0.999803,
                0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803,
                0.999803, 0.999803, 0.999803,  0.999803, 0.999803,
                0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999650, 0.996856, 0.991283,
                      0.982963, 0.971942, 0.958281, 0.942058, 0.923362, 0.902299, 0.878986, 0.853553, 0.826144,
                      0.796910, 0.766016, 0.733634, 0.699946, 0.665140, 0.629410, 0.592956, 0.555982, 0.518696,
                      0.481304, 0.444018, 0.407044, 0.370590, 0.334860, 0.300054, 0.266366, 0.233984, 0.203090,
                      0.173856, 0.146447, 0.121014, 0.097701, 0.076638, 0.057942, 0.041719, 0.028058, 0.017037,
                      0.008717, 0.003144, 0.000350])
    audio = essentia.standard.MonoLoader(filename = fileName)()
    stepsize, semitones = vamp.collect(
        audio, 44100, "nnls-chroma:nnls-chroma", output = "semitonespectrum", step_size=hopSize)["matrix"]
    chroma = np.zeros((semitones.shape[0], 12))
    for i in range(semitones.shape[0]):
        tones = semitones[i] * mywindow
        cc = chroma[i]
        for j in range(tones.size):
            cc[j % 12] = cc[j % 12] + tones[j]
    return smooth(chroma, window_len= int(2.0 * 44100 / 2048), window='hanning').astype('float32')

def chordsByBeats(fileName, beats):
    '''

    :param wav:
    :param beats:
    :return:
    '''
    #chords = ChordsDetectionBeats(hopSize=2048, chromaPick='interbeat_median')
    chords = essentia.standard.ChordsDetectionBeats(hopSize=2048, chromaPick='first_beat')
    chroma = loadChroma(fileName, hopSize=2048)
    syms, strengths = chords(chroma, beats)
    return convertChordLabels(syms), strengths
"""

syms, strengths = chordsByBeats('maj35fromE.mp3',  [
        2.68, 3.35, 4.05, 4.74, 5.48, 6.12, 6.81, 7.52, 8.2, 8.84, 9.54, 10.24, 10.9, 11.59, 12.32, 13.03, 13.68, 14.36, 14.98, 15.68, 16.31, 16.96, 17.66, 18.4,
        19.1, 19.79, 20.44, 21.15, 21.8, 22.51, 23.16, 23.87, 24.53, 25.19, 25.9, 26.61, 27.29, 27.98, 28.64, 29.34, 29.98, 30.66, 31.28, 32.01, 32.7, 33.35, 34.05, 34.77,
        35.43, 36.11, 36.74, 37.45, 38.09, 38.81, 39.47, 40.17, 40.84, 41.54, 42.21, 42.91, 43.57, 44.31, 45.0, 45.66, 46.36, 47.08, 47.74, 48.42, 49.06, 49.75, 50.49, 51.11,
        51.79, 52.47, 53.11, 53.82, 54.48, 55.17, 55.9, 56.59, 57.24, 57.91, 58.6, 59.27, 59.97, 60.68, 61.36, 62.04, 62.68, 63.38, 64.01, 64.72, 65.38, 66.06, 66.69
      ])
expected = np.array(['E', 'N', 'A', 'N', 'D', 'N', 'G', 'N', 'C', 'N', 'F', 'N', 'A#', 'N', 'D#', 'N', 'G#', 'N', 'C#', 'N', 'F#', 'N', 'B', 'N',
          'E', 'N', 'A', 'N', 'D', 'N', 'G', 'N', 'C', 'N', 'F', 'N', 'A#', 'N', 'D#', 'N', 'G#', 'N', 'C#', 'N', 'F#', 'N', 'B', 'N',
          'E', 'N', 'A', 'N', 'D', 'N', 'G', 'N', 'C', 'N', 'F', 'N', 'A#', 'N', 'D#', 'N', 'G#', 'N', 'C#', 'N', 'F#', 'N', 'B', 'N',
          'E', 'N', 'A', 'N', 'D', 'N', 'G', 'N', 'C', 'N', 'F', 'N', 'A#', 'N', 'D#', 'N', 'G#', 'N', 'N', 'N', 'N', 'N'])
matches = np.array(syms) == expected
matches = matches[np.where(expected != 'N')]

print("Expected:")
print(expected)
print("Obtained:")
print(syms)
print("Accuracy on non N.C. chords: %.2f %%" % (100.0 * sum(matches)/len(matches)))

syms, strengths = chordsByBeats('min35fromE.mp3',  [2.67, 3.38, 4.05, 4.77, 5.45, 6.17, 6.86, 7.52, 8.13, 8.82, 9.53, 10.23, 10.89, 11.56, 12.22, 12.85, 13.53, 14.25, 14.93, 15.62, 16.29, 16.97, 17.69, 18.37, 19.07, 19.77])
expected = np.array(['E:min', 'N', 'A:min', 'N', 'D:min', 'N', 'G:min', 'N', 'C:min', 'N', 'F:min', 'N', 'A#:min', 'N', 'D#:min', 'N', 'G#:min', 'N', 'C#:min', 'N', 'F#:min', 'N', 'B:min', 'N', 'N'])
matches = np.array(syms) == expected
matches = matches[np.where(expected != 'N')]

print("Expected:")
print(expected)
print("Obtained:")
print(syms)
print("Accuracy on non N.C. chords: %.2f %%" % (100.0 * sum(matches)/len(matches)))

# TODO: check on The Beatles dataset.