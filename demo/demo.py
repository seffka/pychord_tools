from pychord_detection import chordsByBeats
import numpy as np

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