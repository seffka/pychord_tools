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

# pip install git+https://github.com/ericsuh/dirichlet.git
import dirichlet

def trainAlrGaussian(vectors):
    gmm = GaussianMixture(
        n_components=1,
        covariance_type='full',
        max_iter=200)
    gmm.fit(np.apply_along_axis(alr, 1, vectors))
    return gmm

class ChromaPatternModel:
    """
    Base class for recognizing "patterns" (e.g., chords, scales) in 12D chroma space.
    """
    def __init__(self, kinds, kindToExternalNames = None):
        self.kinds = kinds
        self.NKinds = len(self.kinds)
        if kindToExternalNames != None:
            self.kindToExternalNames = kindToExternalNames
        else:
            self.kindToExternalNames = {k : k for k in self.kinds}

        self.externalNames = np.empty(
            len(PITCH_CLASS_NAMES) * self.NKinds, dtype='object')
        for p in range(len(PITCH_CLASS_NAMES)):
            for c in range(self.NKinds):
                self.externalNames[p * self.NKinds + c] =\
                    PITCH_CLASS_NAMES[p] + kindToExternalNames[self.kinds[c]]

    def logUtilities(self, chromas, normalize = True):
        return np.zeros((len(chromas), len(self.externalNames)))

    def logUtilitiesGivenSequence(self, chromas, pitchedPatterns):
        return np.zeros(len(chromas))

    def predictExternalLabels(self, chromas):
        lu = self.logUtilities(chromas)
        indices = np.argmax(lu, axis = 1)
        return np.array([self.externalNames[x] for x in indices]), np.array([lu[i, indices[i]] for i in range(len(indices))])

    def predict(self, chromas):
        lu = self.logUtilities(chromas)
        indices = np.argmax(lu, axis = 1)
        return np.array([PitchedPattern(kind=self.kinds[x % self.NKinds], pitchClassIndex= x // self.NKinds) for x in indices]),\
               np.array([lu[i, indices[i]] for i in range(len(indices))])

    def fit(self, segments):
        return

    def saveModel(self, fileName):
        joblib.dump(self, fileName)

def degrees2BinaryPattern(degreeList):
    result = np.zeros(12, dtype = 'int')
    result[degreeIndices(degreeList)] = 1
    return result

class CosineSimilarityBinaryPatternModel(ChromaPatternModel):
    """
    Cosine distance to binary patterns
    """
    def __init__(self, kindToDegreesDict, kindToExternalNames = None):
        """
        :param kindToDegreesDict: Dictionary of chord/scale kinds
        to degree name list, e.g., {'maj':['I', 'III', 'V']}
        """
        ChromaPatternModel.__init__(self, list(kindToDegreesDict.keys()), kindToExternalNames)
        self.kindToPatternsDict = {key: degrees2BinaryPattern(value) for key, value in kindToDegreesDict.items()}

    def preprocess(self, chroma):
        return preprocessing.normalize(chroma, norm='l1')

    def logUtilities(self, chromas, normalize = True):
        lps = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio preprocessing should be applied to shifted
            # chroma, so we always do it inside loop.
            preChromas = self.preprocess(chromas)
            chromasNorm = substituteZeros(np.sqrt((preChromas * preChromas).sum(axis=1)), copy=False)
            ki = 0
            for k in self.kinds:
                pattern = np.tile(self.kindToPatternsDict[k], [preChromas.shape[0], 1])
                patternNorm = np.sqrt((pattern * pattern).sum(axis = 1))
                similarities = (pattern * preChromas).sum(axis = 1) / (patternNorm * chromasNorm)
                lps[:, pos + ki] = np.log(substituteZeros(similarities, copy=False))
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize :
            normSum = logsumexp(lps, axis=1)
            return lps - normSum[:, np.newaxis]
        else:
            return lps

    def logUtilitiesGivenSequence(self, chromas, pitchedPatterns):
        res = np.zeros(len(chromas))
        if (len(chromas) != len(pitchedPatterns)):
            raise ValueError("Input vectors need to be equal size.")
        for i in range(len(pitchedPatterns)):
            c = self.preprocess(np.roll(chromas[i].reshape(1,-1), -pitchedPatterns[i].pitchClassIndex))
            pattern = self.kindToPatternsDict[pitchedPatterns[i].kind]
            patternNorm = np.sqrt((pattern * pattern).sum())
            chromasNorm = substituteZeros(np.sqrt((c * c).sum()))
            res[i] = (pattern * c).sum(axis=1) / (patternNorm * chromasNorm)
        return np.log(substituteZeros(res))

class CorrectnessBalanceResidualsModel(ChromaPatternModel):

    def __init__(self, kinds, kindToExternalNames = None):
        ChromaPatternModel.__init__(self, kinds, kindToExternalNames)

    def correctness(self, chroma, pitchedPatterns):
        return

    def balance(self, chroma, pitchedPatterns):
        return

    def residuals(self, chroma, pitchedPatterns):
        return

class CorrectnessLogNormBalanceModel(CorrectnessBalanceResidualsModel):
    """
    Correctness/Balance/Residuals chord model.
    Balance is modeled as Log normal.
    """
    def __init__(self, kindToDegreesDict, kindToExternalNames = None):
        """
        :param kindToDegreesDict: Dictionary of chord/scale kinds
        to degree name list, e.g., {'maj':['I', 'III', 'V']}
        """
        CorrectnessBalanceResidualsModel.__init__(self, list(kindToDegreesDict.keys()), kindToExternalNames)
        self.inDegreeDict = dict()
        self.outDegreeDict = dict()
        self.outDegreeDict = dict()

        for k, v in kindToDegreesDict.items():
            self.inDegreeDict[k] = degreeIndices(v)
            self.outDegreeDict[k] = list(set(range(12)).difference(set(self.inDegreeDict[k])))

        self.balanceGMMs = dict()
        self.residualDirichletAlphas = dict()
        self.betaParams = None

    def preprocess(self, chroma):
        return preprocessing.normalize(substituteZeros(chroma), norm='l1')

    def fit(self, segments):
        """
        Fits the model to given chroma segments.
        :param segments: AnnotatedChromaSegment list
        """
        inChromaSums = dict()

        for k in self.kinds:
            chroma = self.preprocess(segments.chromas[segments.kinds == k])
            partition = [self.inDegreeDict[k], self.outDegreeDict[k]]
            inChromaSums[k] = amalgamate(partition, chroma).transpose()[0]
            inChromaComposition = subcomposition([[e] for e in self.inDegreeDict[k]], chroma)
            self.balanceGMMs[k] = trainAlrGaussian(inChromaComposition)
            outChromaComposition = subcomposition([[e] for e in self.outDegreeDict[k]], chroma).astype('float64')
            self.residualDirichletAlphas[k] = dirichlet.mle(outChromaComposition)

        allChords = np.concatenate(list(inChromaSums.values()))
        self.betaParams = beta.fit(allChords, floc=0, fscale=1)

    def logUtilities(self, chromas, normalize = True):
        lps = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio preprocessing should be applied to shifted
            # chroma, so we always do it inside loop.
            preChromas = self.preprocess(chromas)
            ki = 0
            distBeta = beta(*self.betaParams)
            for k in self.kinds:
                # TODO:
                partition = [self.inDegreeDict[k], self.outDegreeDict[k]]
                inChromaSums = amalgamate(partition, preChromas).transpose()[0]
                inChromaComposition = subcomposition([[e] for e in self.inDegreeDict[k]], preChromas)
                outChromaComposition = subcomposition([[e] for e in self.outDegreeDict[k]], preChromas)
                correctness = distBeta.logcdf(inChromaSums)
                balance = self.balanceGMMs[k].score_samples(np.apply_along_axis(alr, 1, inChromaComposition))
                #residual = scipy.stats.dirichlet.logpdf(
                #    outChromaComposition.transpose()[0:-1, :],
                #    self.residualDirichletAlphas[k])

                #lps[:, pos + ki] = (correctness + balance + residual)
                lps[:, pos + ki] = (correctness + balance)
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize :
            normSum = logsumexp(lps, axis=1)
            return lps - normSum[:, np.newaxis]
        else:
            return lps

    def logUtilitiesGivenSequence(self, chromas, pitchedPatterns):
        return self.correctness(chromas, pitchedPatterns) + self.balance(chromas, pitchedPatterns)

    def correctness(self, chromas, pitchedPatterns):
        res = np.zeros(len(chromas))
        distBeta = beta(*self.betaParams)
        if (len(chromas) != len(pitchedPatterns)):
            raise ValueError("Input vectors need to be equal size.")
        for i in range(len(pitchedPatterns)):
            c = self.preprocess(np.roll(chromas[i].reshape(1,-1), -pitchedPatterns[i].pitchClassIndex))
            k = pitchedPatterns[i].kind
            partition = [self.inDegreeDict[k], self.outDegreeDict[k]]
            inChromaSums = amalgamate(partition, c)[:,0]
            res[i] = distBeta.logcdf(inChromaSums)
        return res

    def balance(self, chromas, pitchedPatterns):
        res = np.zeros(len(chromas))
        distBeta = beta(*self.betaParams)
        if (len(chromas) != len(pitchedPatterns)):
            raise ValueError("Input vectors need to be equal size.")
        for i in range(len(pitchedPatterns)):
            c = self.preprocess(np.roll(chromas[i].reshape(1,-1), -pitchedPatterns[i].pitchClassIndex))
            k = pitchedPatterns[i].kind
            inChromaComposition = subcomposition([[e] for e in self.inDegreeDict[k]], c)
            res[i] = self.balanceGMMs[k].score_samples(np.apply_along_axis(alr, 1, inChromaComposition))
        return res

    def residuals(self, chroma, pitchedPatterns):
        return


class DirichletModel(ChromaPatternModel):
    """
    Simple Dirichlet chord model
    """
    def __init__(self, kinds, kindToExternalNames = None):
        """
        :param kindToDegreesDict: Dictionary of chord/scale kinds
        to degree name list, e.g., {'maj':['I', 'III', 'V']}
        """
        ChromaPatternModel.__init__(self, kinds, kindToExternalNames)
        self.alphas = dict()

    def preprocess(self, chroma):
        return preprocessing.normalize(substituteZeros(chroma), norm='l1')

    def fit(self, segments):
        """
        Fits the model to given chroma segments.
        :param segments: AnnotatedChromaSegment list
        """
        for k in self.kinds:
            chroma = self.preprocess(segments.chromas[segments.kinds == k])
            self.alphas[k] = dirichlet.mle(chroma)

    def logUtilities(self, chromas, normalize):
        lps = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio preprocessing should be applied to shifted
            # chroma, so we always do it inside loop.
            preChromas = self.preprocess(chromas)
            ki = 0
            for k in self.kinds:
                lps[:, pos + ki] = scipy.stats.dirichlet.logpdf(
                    preChromas.transpose()[0:-1, :],
                    self.alphas[k])
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize :
            normSum = logsumexp(lps, axis=1)
            return lps - normSum[:, np.newaxis]
        else:
            return lps

class CorrectnessDirichletBalanceModel(CorrectnessBalanceResidualsModel):
    """
    Correctness/Balance/Residuals chord model
    Balance is modeled as Dirichlet.
    """
    def __init__(self, kindToDegreesDict, kindToExternalNames = None):
        """
        :param kindToDegreesDict: Dictionary of chord/scale kinds
        to degree name list, e.g., {'maj':['I', 'III', 'V']}
        """
        CorrectnessBalanceResidualsModel.__init__(self, list(kindToDegreesDict.keys()), kindToExternalNames)
        self.inDegreeDict = dict()
        self.outDegreeDict = dict()
        self.outDegreeDict = dict()

        for k, v in kindToDegreesDict.items():
            self.inDegreeDict[k] = degreeIndices(v)
            self.outDegreeDict[k] = list(set(range(12)).difference(set(self.inDegreeDict[k])))

        self.dirichlets = dict()
        self.residualDirichletAlphas = dict()
        self.betaParams = None

    def preprocess(self, chroma):
        return preprocessing.normalize(substituteZeros(chroma), norm='l1')

    def fit(self, segments):
        """
        Fits the model to given chroma segments.
        :param segments: AnnotatedChromaSegment list
        """
        inChromaSums = dict()

        for k in self.kinds:
            chroma = self.preprocess(segments.chromas[segments.kinds == k])
            partition = [self.inDegreeDict[k], self.outDegreeDict[k]]
            inChromaSums[k] = amalgamate(partition, chroma).transpose()[0]
            inChromaComposition = subcomposition([[e] for e in self.inDegreeDict[k]], chroma).astype('float64')
            self.dirichlets[k] = dirichlet.mle(inChromaComposition)
            outChromaComposition = subcomposition([[e] for e in self.outDegreeDict[k]], chroma).astype('float64')
            self.residualDirichletAlphas[k] = dirichlet.mle(outChromaComposition)

        allChords = np.concatenate(list(inChromaSums.values()))
        self.betaParams = beta.fit(allChords, floc=0, fscale=1)

    def logUtilities(self, chromas, normalize = True):
        lps = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio preprocessing should be applied to shifted
            # chroma, so we always do it inside loop.
            preChromas = self.preprocess(chromas)
            ki = 0
            distBeta = beta(*self.betaParams)
            for k in self.kinds:
                # TODO:
                partition = [self.inDegreeDict[k], self.outDegreeDict[k]]
                inChromaSums = amalgamate(partition, preChromas).transpose()[0]
                inChromaComposition = subcomposition([[e] for e in self.inDegreeDict[k]], preChromas)
                correctness = distBeta.logcdf(inChromaSums)
                balance = scipy.stats.dirichlet.logpdf(
                    inChromaComposition.transpose()[0:-1, :],
                    self.dirichlets[k])
                lps[:, pos + ki] = (correctness + balance)
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize :
            normSum = logsumexp(lps, axis=1)
            return lps - normSum[:, np.newaxis]
        else:
            return lps

def testAccuracy(predictedChromas, groundTruthSegments):
    inds = np.where(groundTruthSegments.kinds != 'unclassified')
    wrongIndices = [inds[0][i] for i in np.where(predictedChromas[inds] != convertChordLabels(groundTruthSegments.labels[inds]))[0]]
    for i in wrongIndices:
        print(groundTruthSegments.uids[i],
              groundTruthSegments.startTimes[i],
              groundTruthSegments.labels[i], ' != ',
              predictedChromas[i])
    return (float(len(inds[0]) - len(wrongIndices)) / len(inds[0]))

def loadModel(filename):
    return joblib.load(filename)