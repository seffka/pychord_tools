import numpy as np
import joblib
import scipy.stats
from scipy.stats import beta
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

from pychord_tools.compositionalData import alr
from pychord_tools.compositionalData import amalgamate
from pychord_tools.compositionalData import subcomposition
from pychord_tools.compositionalData import substituteZeros
from pychord_tools.lowLevelFeatures import AnnotatedBeatChromaEstimator
from pychord_tools.lowLevelFeatures import SmoothedStartingBeatChromaEstimator
from pychord_tools.lowLevelFeatures import DEGREES, degreeIndices, PitchedPattern
from pychord_tools.lowLevelFeatures import PITCH_CLASS_NAMES
from pychord_tools.third_party import NNLSChromaEstimator
from scipy.misc import logsumexp
from pychord_tools.commonUtils import convertChordLabels
from pychord_tools.models import CosineSimilarityBinaryPatternModel, CorrectnessLogNormBalanceModel, testAccuracy, DirichletModel, CorrectnessDirichletBalanceModel

from essentia.standard import MonoLoader
import essentia

chromaEstimator = AnnotatedBeatChromaEstimator(
    chromaEstimator = NNLSChromaEstimator(),
    segmentChromaEstimator = SmoothedStartingBeatChromaEstimator(smoothingTime = 0.6))
segments = chromaEstimator.loadChromasForAnnotationFileListFile('correct.txt')

untransposedEstimator = AnnotatedBeatChromaEstimator(
    chromaEstimator = NNLSChromaEstimator(),
    segmentChromaEstimator = SmoothedStartingBeatChromaEstimator(smoothingTime = 0.6),
    rollToCRoot=False)
realSegments = untransposedEstimator.loadChromasForAnnotationFileListFile('correct.txt')
inds = np.where(realSegments.kinds != 'unclassified')

import matplotlib.pyplot as plt

# Comparison of statistics approach and cosine distance for chord detection.

cosm = CosineSimilarityBinaryPatternModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
cosm.fit(segments)
chords, coslu = cosm.predictExternalLabels(realSegments.chromas)
print("Cosine distance/binary patterns:", testAccuracy(chords, realSegments))

m = CorrectnessLogNormBalanceModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
m.fit(segments)
chords, lu = m.predictExternalLabels(realSegments.chromas)
print("CorrectnessBalanceModel3:", testAccuracy(chords, realSegments))

plt.plot(np.exp(coslu)[inds], label = "normalized utility based on Correctness + Balance")
plt.plot(np.exp(lu)[inds], label = "normalized cosine distance")
plt.legend()
plt.xlabel("sample N")
#plt.plot(np.exp((b+c))[inds])
plt.show()

#############################################################

# TODO: it must be UNIT tests.
# 1) Correctness Balance Model

m = CorrectnessLogNormBalanceModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
m.fit(segments)
p, lu = m.predict(realSegments.chromas)
c = m.correctness(realSegments.chromas, p)
b = m.balance(realSegments.chromas, p)

# All these should be very close:
a = c+b
lu_raw = np.max(m.logUtilities(realSegments.chromas, normalize=False), axis=1)
given = m.logUtilitiesGivenSequence(realSegments.chromas, p)

distRaw = sum((a - lu_raw) * (a - lu_raw))
print(distRaw)
distGiven = sum((given - lu_raw) * (given - lu_raw))
print(distGiven)

plt.plot(given)
plt.plot(a)
plt.plot(lu_raw)
plt.show()

# 2) Cosine Model
cosm = CosineSimilarityBinaryPatternModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
cosm.fit(segments)
p, lu = cosm.predict(realSegments.chromas)
coslu_raw = np.max(cosm.logUtilities(realSegments.chromas, normalize=False), axis=1)
given = cosm.logUtilitiesGivenSequence(realSegments.chromas, p)
#All these should be very close:
distGiven = sum((given - coslu_raw) * (given - coslu_raw))
print(distGiven)

plt.plot(given)
plt.plot(coslu_raw)
plt.show()

#############################################################
# Correlation between Cosine distance and Corretness.
#
#i, keys[i], inds[0][keys[i]],
#print(groundTruthSegments.uids[inds[0][keys[i]]],
#      groundTruthSegments.startTimes[inds[0][keys[i]]],
#      groundTruthSegments.labels[inds[0][keys[i]]])

def writeExtractedBeats(indices, groundTruthSegments, filename):
    res = []
    for i in indices:
        audio = MonoLoader(filename=groundTruthSegments.uids[i], sampleRate=44100)()
        start = int(float(groundTruthSegments.startTimes[i]) * 44100)
        end = start + int(float(groundTruthSegments.durations[i]) * 44100)
        res.append(audio[start:end])
        if i != indices[-1]:
            res.append(np.zeros(20000, dtype = 'float32'))
    essentia.standard.MonoWriter(filename=filename)(essentia.array(np.concatenate(res)))

def top5Errors(data, inds, groundTruthSegments):
    keys = sorted(range(len(data)), key =  lambda k: data[k])
    for i in range(5):
        print(inds[0][keys[i]],
              groundTruthSegments.uids[inds[0][keys[i]]],
              groundTruthSegments.startTimes[inds[0][keys[i]]],
              groundTruthSegments.durations[inds[0][keys[i]]],
              groundTruthSegments.labels[inds[0][keys[i]]])
    return inds[0][keys[0:5]]

cosm = CosineSimilarityBinaryPatternModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
cosm.fit(segments)
m = CorrectnessLogNormBalanceModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
m.fit(segments)
p = realSegments.pitchedPatterns()[inds]

c = m.correctness(realSegments.chromas[inds], p)
b = m.balance(realSegments.chromas[inds], p)
lu = m.logUtilitiesGivenSequence(realSegments.chromas[inds], p)
coslu = cosm.logUtilitiesGivenSequence(realSegments.chromas[inds], p)


# High correlation with other metrics => make sense
print("LogCorrectness/cosine distance correlation", scipy.stats.pearsonr(c, coslu))
print("LogBalance/cosine distance correlation", scipy.stats.pearsonr(b, coslu))
print("LogUtility/cosine distance correlation", scipy.stats.pearsonr(lu, coslu))

# Low inter-correlation => independent
print("LogCorrectness/LogBalance distance correlation", scipy.stats.pearsonr(c, b))

plt.plot(preprocessing.scale(c), label = 'standardized log correctness')
plt.plot(preprocessing.scale(b), label = 'standardized log balance')
plt.plot(preprocessing.scale(np.exp(coslu)), label = 'standardized cosine (log is the same, because it''s ~1)')
plt.xlabel("sample N")
plt.legend()
plt.show()

plt.plot(c, label = 'log correctness')
plt.plot(b, label = 'log balance')
plt.plot(coslu, label = 'log cosine (log is the same, because it''s ~1)')
plt.xlabel("sample N")
plt.legend()
plt.show()

cosineDetected = top5Errors(coslu, inds, realSegments)
writeExtractedBeats(cosineDetected, realSegments, "cos.wav")

correctnessDetected = top5Errors(c, inds, realSegments)
writeExtractedBeats(correctnessDetected, realSegments, "cor.wav")

balanceDetected = top5Errors(b, inds, realSegments)
writeExtractedBeats(balanceDetected, realSegments, "bal.wav")

#############################################################

# Correctness, log-normal balance, dirichlet balance, residuals (various?) plot

# Good

# 3/9
m = CorrectnessLogNormBalanceModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
m.fit(segments)
chords, lu = m.predictExternalLabels(realSegments.chromas)
print("CorrectnessBalanceModel3:", testAccuracy(chords, realSegments))
p, lu = m.predict(realSegments.chromas)
c = m.correctness(realSegments.chromas, p)
b = m.balance(realSegments.chromas, p)

inds = np.where(realSegments.kinds != 'unclassified')
plt.plot(lu[inds])
plt.plot((b+c)[inds])
plt.plot(b[inds])
plt.plot(c[inds])
plt.show()

# 5/7
m = CorrectnessLogNormBalanceModel({'maj':['I', 'III', 'V', 'II', 'VII'], 'min':['I', 'IIIb', 'V', 'II', 'VIIb']}, {'maj': '', 'min': ':min'})
m.fit(segments)
chords, lu = m.predictExternalLabels(realSegments.chromas)
print("CorrectnessBalanceModel5:", testAccuracy(chords, realSegments))
inds = np.where(realSegments.kinds != 'unclassified')
plt.plot(np.exp(lu[inds]))
plt.show()

# DirichletBalance
m = CorrectnessDirichletBalanceModel({'maj':['I', 'III', 'V', 'II', 'VII'], 'min':['I', 'IIIb', 'V', 'II', 'VIIb']}, {'maj':'', 'min':':min'})
m.fit(segments)
chords, lu = m.predictExternalLabels(realSegments.chromas)
print("DirichletBalance:", testAccuracy(chords, realSegments))
inds = np.where(realSegments.kinds != 'unclassified')
plt.plot(np.exp(lu[inds]))
plt.show()

# Bad (?)

# Cosine distances
m = CosineSimilarityBinaryPatternModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
m.fit(segments)
chords, lu = m.predictExternalLabels(realSegments.chromas)
print("Cosine distance/binary patterns:", testAccuracy(chords, realSegments))
inds = np.where(realSegments.kinds != 'unclassified')
plt.plot(np.exp(lu[inds]))
plt.show()

# Dirichlet
m = DirichletModel(['maj', 'min'], {'maj':'', 'min':':min'})
m.fit(segments)
chords, lu = m.predictExternalLabels(realSegments.chromas)
print("Dirichlet:", testAccuracy(chords, realSegments))
inds = np.where(realSegments.kinds != 'unclassified')
plt.plot(lu[inds])
plt.show()

########################################################################################################################

