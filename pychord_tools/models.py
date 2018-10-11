import numpy as np
import joblib
import scipy.stats
from scipy.stats import beta
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from scipy.misc import logsumexp
from scipy.stats import norm

from .compositional_data import alr
from .compositional_data import amalgamate
from .compositional_data import subcomposition
from .compositional_data import substitute_zeros
from .low_level_features import AnnotatedBeatChromaEstimator
from .low_level_features import SmoothedStartingBeatChromaEstimator
from .labels import PITCH_CLASS_NAMES, DEGREES, PitchedPattern, degree_indices, convert_chord_labels
from .third_party import NNLSChromaEstimator

# pip install git+https://github.com/ericsuh/dirichlet.git
import dirichlet


class ChromaPatternModel:
    """
    Base class for recognizing "patterns" (e.g., chords, scales) in 12D chroma space.
    """
    def __init__(self, kinds, kind_to_external_names=None):
        self.kinds = kinds
        self.NKinds = len(self.kinds)
        if kind_to_external_names is not None:
            self.kindToExternalNames = kind_to_external_names
        else:
            self.kindToExternalNames = {k : k for k in self.kinds}

        self.externalNames = np.empty(
            len(PITCH_CLASS_NAMES) * self.NKinds, dtype='object')
        for p in range(len(PITCH_CLASS_NAMES)):
            for c in range(self.NKinds):
                self.externalNames[p * self.NKinds + c] = \
                    PITCH_CLASS_NAMES[p] + kind_to_external_names[self.kinds[c]]

    def log_utilities(self, chromas, normalize=True):
        return np.zeros((len(chromas), len(self.externalNames)))

    def log_utilities_given_sequence(self, chromas, pitched_patterns, normalize=False):
        return np.zeros(len(chromas))

    def predict_external_labels(self, chromas):
        lu = self.log_utilities(chromas)
        indices = np.argmax(lu, axis=1)
        return np.array([self.externalNames[x] for x in indices]),\
               np.array([lu[i, indices[i]] for i in range(len(indices))])

    def predict(self, chromas):
        lu = self.log_utilities(chromas)
        indices = np.argmax(lu, axis=1)
        return np.array(
            [PitchedPattern(kind=self.kinds[x % self.NKinds], pitch_class_index=x // self.NKinds) for x in indices]),\
               np.array([lu[i, indices[i]] for i in range(len(indices))])

    def fit(self, segments):
        return

    def save_model(self, file_name):
        joblib.dump(self, file_name)

    def index(self, pitched_pattern):
        return pitched_pattern.pitch_class_index * self.NKinds + self.kinds.index(pitched_pattern.kind)

def degrees_to_binary_pattern(degree_list):
    result = np.zeros(12, dtype='int')
    result[degree_indices(degree_list)] = 1
    return result


class CosineSimilarityBinaryPatternModel(ChromaPatternModel):
    """
    Cosine distance to binary patterns
    """
    def __init__(self, kind_to_degrees_dict, kind_to_external_names=None):
        """
        :param kind_to_degrees_dict: Dictionary of chord/scale kinds
        to degree name list, e.g., {'maj':['I', 'III', 'V']}
        """
        ChromaPatternModel.__init__(self, list(kind_to_degrees_dict.keys()), kind_to_external_names)
        self.kindToPatternsDict = {key: degrees_to_binary_pattern(value) for key, value in kind_to_degrees_dict.items()}

    def preprocess(self, chroma):
        return preprocessing.normalize(chroma, norm='l1')

    def log_utilities(self, chromas, normalize=True):
        lps = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio pre-processing should be applied to shifted
            # chroma, so we always do it inside loop.
            pre_chromas = self.preprocess(chromas)
            chromas_norm = substitute_zeros(np.sqrt((pre_chromas * pre_chromas).sum(axis=1)), copy=False)
            ki = 0
            for k in self.kinds:
                pattern = np.tile(self.kindToPatternsDict[k], [pre_chromas.shape[0], 1])
                pattern_norm = np.sqrt((pattern * pattern).sum(axis=1))
                similarities = (pattern * pre_chromas).sum(axis=1) / (pattern_norm * chromas_norm)
                lps[:, pos + ki] = np.log(substitute_zeros(similarities, copy=False))
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize:
            norm_sum = logsumexp(lps, axis=1)
            return lps - norm_sum[:, np.newaxis]
        else:
            return lps

    def log_utilities_given_sequence(self, chromas, pitched_patterns, normalize=False):
        res = np.zeros(len(chromas))
        if len(chromas) != len(pitched_patterns):
            raise ValueError("Input vectors need to be equal size.")
        for i in range(len(pitched_patterns)):
            c = self.preprocess(np.roll(chromas[i].reshape(1, -1), -pitched_patterns[i].pitch_class_index))
            pattern = self.kindToPatternsDict[pitched_patterns[i].kind]
            pattern_norm = np.sqrt((pattern * pattern).sum())
            chromas_norm = substitute_zeros(np.sqrt((c * c).sum()))
            res[i] = (pattern * c).sum(axis=1) / (pattern_norm * chromas_norm)
        return np.log(substitute_zeros(res))


class CorrectnessBalanceResidualsModel(ChromaPatternModel):
    def __init__(self, kinds, kind_to_external_names=None):
        ChromaPatternModel.__init__(self, kinds, kind_to_external_names)

    def correctness_given_sequence(self, chroma, pitched_patterns, normalize=False):
        return

    def balance_given_sequence(self, chroma, pitched_patterns, normalize=False):
        return

    def correctness(self, chroma, normalize=False):
        return

    def balance(self, chroma, normalize=False):
        return

    def max_balance(self):
        return 1.0


class CorrectnessLogNormBalanceModel(CorrectnessBalanceResidualsModel):
    """
    Correctness/Balance/Residuals chord model.
    Balance is modeled as Log normal.
    """
    def __init__(self, kind_to_degrees_dict, kind_to_external_names=None):
        """
        :param kind_to_degrees_dict: Dictionary of chord/scale kinds
        to degree name list, e.g., {'maj':['I', 'III', 'V']}
        """
        CorrectnessBalanceResidualsModel.__init__(self, list(kind_to_degrees_dict.keys()), kind_to_external_names)
        self.in_degree_dict = dict()
        self.out_degree_dict = dict()

        for k, v in kind_to_degrees_dict.items():
            self.in_degree_dict[k] = degree_indices(v)
            self.out_degree_dict[k] = list(set(range(12)).difference(set(self.in_degree_dict[k])))

        self.balanceGMMs = dict()
        self.residualDirichletAlphas = dict()
        self.betaParams = None

    def preprocess(self, chroma):
        return preprocessing.normalize(substitute_zeros(chroma), norm='l1')

    def train_alr_gaussian(self, vectors):
        gmm = GaussianMixture(
            n_components=1,
            covariance_type='full',
            max_iter=200)
        gmm.fit(np.apply_along_axis(alr, 1, vectors))
        return gmm

    def fit(self, segments):
        """
        Fits the model to given chroma segments.
        :param segments: AnnotatedChromaSegment list
        """
        in_chroma_sums = dict()

        for k in self.kinds:
            chroma = self.preprocess(segments.chromas[segments.kinds == k])
            partition = [self.in_degree_dict[k], self.out_degree_dict[k]]
            in_chroma_sums[k] = amalgamate(partition, chroma).transpose()[0]
            in_chroma_composition = subcomposition([[e] for e in self.in_degree_dict[k]], chroma)
            self.balanceGMMs[k] = self.train_alr_gaussian(in_chroma_composition)
            out_chroma_composition = subcomposition([[e] for e in self.out_degree_dict[k]], chroma).astype('float64')
            self.residualDirichletAlphas[k] = dirichlet.mle(out_chroma_composition)

        all_chords = np.concatenate(list(in_chroma_sums.values()))
        self.betaParams = beta.fit(all_chords, floc=0, fscale=1)

    def score_balance(self, kind, pre_chromas):
        in_chroma_composition = subcomposition([[e] for e in self.in_degree_dict[kind]], pre_chromas)
        return self.balanceGMMs[kind].score_samples(np.apply_along_axis(alr, 1, in_chroma_composition))

    def log_utilities(self, chromas, normalize=True):
        lps = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        dist_beta = beta(*self.betaParams)
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio preprocessing should be applied to shifted
            # chroma, so we always do it inside loop.
            pre_chromas = self.preprocess(chromas)
            ki = 0
            for k in self.kinds:
                partition = [self.in_degree_dict[k], self.out_degree_dict[k]]
                in_chroma_sums = amalgamate(partition, pre_chromas).transpose()[0]
                in_chroma_composition = subcomposition([[e] for e in self.in_degree_dict[k]], pre_chromas)
                correctness = dist_beta.logcdf(in_chroma_sums)
                balance = self.balanceGMMs[k].score_samples(np.apply_along_axis(alr, 1, in_chroma_composition))
                lps[:, pos + ki] = (correctness + balance)
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize:
            norm_sum = logsumexp(lps, axis=1)
            return lps - norm_sum[:, np.newaxis]
        else:
            return lps

    def correctness(self, chromas, normalize=False):
        res = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        dist_beta = beta(*self.betaParams)
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio preprocessing should be applied to shifted
            # chroma, so we always do it inside loop.
            pre_chromas = self.preprocess(chromas)
            ki = 0
            for k in self.kinds:
                partition = [self.in_degree_dict[k], self.out_degree_dict[k]]
                in_chroma_sums = amalgamate(partition, pre_chromas).transpose()[0]
                res[:, pos + ki] += dist_beta.logcdf(in_chroma_sums)
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize:
            norm_sum = logsumexp(res, axis=1)
            return res - norm_sum[:, np.newaxis]
        else:
            return res

    def balance(self, chromas, normalize=False):
        res = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio preprocessing should be applied to shifted
            # chroma, so we always do it inside loop.
            pre_chromas = self.preprocess(chromas)
            ki = 0
            for k in self.kinds:
                res[:, pos + ki] += self.score_balance(k, pre_chromas)
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize:
            norm_sum = logsumexp(res, axis=1)
            return res - norm_sum[:, np.newaxis]
        else:
            return res

    def log_utilities_given_sequence(self, chromas, pitched_patterns, normalize=False):
        if normalize:
            lu = self.log_utilities(chromas)
            indices = [p.pitch_class_index * self.NKinds + self.kinds.index(p.kind) for p in pitched_patterns]
            return np.array([lu[i, indices[i]] for i in range(len(indices))])
        else:
            return self.correctness_given_sequence(chromas, pitched_patterns) +\
                   self.balance_given_sequence(chromas, pitched_patterns)

    def correctness_given_sequence(self, chromas, pitched_patterns, normalize=False):
        res = np.zeros(len(chromas))
        dist_beta = beta(*self.betaParams)
        if len(chromas) != len(pitched_patterns):
            raise ValueError("Input vectors need to be equal size.")
        if normalize:
            c = self.correctness(chromas, True)
            indices = [p.pitch_class_index * self.NKinds + self.kinds.index(p.kind) for p in pitched_patterns]
            return np.array([c[i, indices[i]] for i in range(len(indices))])
        else:
            for i in range(len(pitched_patterns)):
                c = self.preprocess(np.roll(chromas[i].reshape(1, -1), -pitched_patterns[i].pitch_class_index))
                k = pitched_patterns[i].kind
                partition = [self.in_degree_dict[k], self.out_degree_dict[k]]
                in_chroma_sums = amalgamate(partition, c)[:, 0]
                res[i] = dist_beta.logcdf(in_chroma_sums)
        return res

    def balance_given_sequence(self, chromas, pitched_patterns, normalize=False):
        res = np.zeros(len(chromas))
        if len(chromas) != len(pitched_patterns):
            raise ValueError("Input vectors need to be equal size.")
        if normalize:
            b = self.balance(chromas, True)
            indices = [p.pitch_class_index * self.NKinds + self.kinds.index(p.kind) for p in pitched_patterns]
            return np.array([b[i, indices[i]] for i in range(len(indices))])
        else:
            for i in range(len(pitched_patterns)):
                c = self.preprocess(np.roll(chromas[i].reshape(1, -1), -pitched_patterns[i].pitch_class_index))
                k = pitched_patterns[i].kind
                res[i] = self.score_balance(k, c)
        return res

    def max_balance(self):
        candidates = []
        for k in self.kinds:
            candidates.append(self.balanceGMMs[k].score_samples(
                self.balanceGMMs[k].means_))
        return max(candidates)


class IndependentPDFModel(CorrectnessLogNormBalanceModel):
    def train_alr_gaussian(self, vectors):
        gmm = GaussianMixture(
            n_components=1,
            covariance_type='diag',
            max_iter=200)
        gmm.fit(np.apply_along_axis(alr, 1, vectors))
        return gmm


class IndependentIntegralModel(IndependentPDFModel):
    def __init__(self, kind_to_degrees_dict, kind_to_external_names=None):
        IndependentPDFModel.__init__(self, kind_to_degrees_dict, kind_to_external_names)
        self.ps = None

    def get_ps(self):
        if self.ps is None:
            self.ps = dict()
            for k in self.balanceGMMs.keys():
                self.ps[k] = sorted(self.balanceGMMs[k].score_samples(
                    self.balanceGMMs[k].sample(200000)[0]))
        return self.ps

    def score_balance(self, kind, pre_chromas):
        in_chroma_composition = subcomposition([[e] for e in self.in_degree_dict[kind]], pre_chromas)
        pdfs = self.balanceGMMs[kind].score_samples(np.apply_along_axis(alr, 1, in_chroma_composition))
        inds = np.searchsorted(self.get_ps()[kind], pdfs).astype('float32')
        return np.log(substitute_zeros(inds / len(self.get_ps()[kind]), copy=False))


class DirichletModel(ChromaPatternModel):
    """
    Simple Dirichlet chord model
    """
    def __init__(self, kinds, kind_to_external_names=None):
        """
        :param kind_to_external_names: Dictionary of chord/scale kinds
        to degree name list, e.g., {'maj':['I', 'III', 'V']}
        """
        ChromaPatternModel.__init__(self, kinds, kind_to_external_names)
        self.alphas = dict()

    def preprocess(self, chroma):
        return preprocessing.normalize(substitute_zeros(chroma), norm='l1')

    def fit(self, segments):
        """
        Fits the model to given chroma segments.
        :param segments: AnnotatedChromaSegment list
        """
        for k in self.kinds:
            chroma = self.preprocess(segments.chromas[segments.kinds == k])
            self.alphas[k] = dirichlet.mle(chroma)

    def log_utilities(self, chromas, normalize=True):
        lps = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio preprocessing should be applied to shifted
            # chroma, so we always do it inside loop.
            pre_chromas = self.preprocess(chromas)
            ki = 0
            for k in self.kinds:
                lps[:, pos + ki] = scipy.stats.dirichlet.logpdf(
                    pre_chromas.transpose()[0:-1, :],
                    self.alphas[k])
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize:
            norm_sum = logsumexp(lps, axis=1)
            return lps - norm_sum[:, np.newaxis]
        else:
            return lps


class CorrectnessDirichletBalanceModel(CorrectnessBalanceResidualsModel):
    """
    Correctness/Balance/Residuals chord model
    Balance is modeled as Dirichlet.
    """
    def __init__(self, kind_to_degrees_dict, kind_to_external_names=None):
        """
        :param kind_to_degrees_dict: Dictionary of chord/scale kinds
        to degree name list, e.g., {'maj':['I', 'III', 'V']}
        """
        CorrectnessBalanceResidualsModel.__init__(self, list(kind_to_degrees_dict.keys()), kind_to_external_names)
        self.inDegreeDict = dict()
        self.outDegreeDict = dict()
        self.outDegreeDict = dict()

        for k, v in kind_to_degrees_dict.items():
            self.inDegreeDict[k] = degree_indices(v)
            self.outDegreeDict[k] = list(set(range(12)).difference(set(self.inDegreeDict[k])))

        self.dirichlets = dict()
        self.residualDirichletAlphas = dict()
        self.betaParams = None

    def preprocess(self, chroma):
        return preprocessing.normalize(substitute_zeros(chroma), norm='l1')

    def fit(self, segments):
        """
        Fits the model to given chroma segments.
        :param segments: AnnotatedChromaSegment list
        """
        in_chroma_sums = dict()

        for k in self.kinds:
            chroma = self.preprocess(segments.chromas[segments.kinds == k])
            partition = [self.inDegreeDict[k], self.outDegreeDict[k]]
            in_chroma_sums[k] = amalgamate(partition, chroma).transpose()[0]
            in_chroma_composition = subcomposition([[e] for e in self.inDegreeDict[k]], chroma).astype('float64')
            self.dirichlets[k] = dirichlet.mle(in_chroma_composition)
            out_chroma_composition = subcomposition([[e] for e in self.outDegreeDict[k]], chroma).astype('float64')
            self.residualDirichletAlphas[k] = dirichlet.mle(out_chroma_composition)

        all_chords = np.concatenate(list(in_chroma_sums.values()))
        self.betaParams = beta.fit(all_chords, floc=0, fscale=1)

    def log_utilities(self, chromas, normalize=True):
        lps = np.zeros((len(chromas), len(self.externalNames)))
        chromas = chromas.astype('float64')
        for basePitch in range(len(PITCH_CLASS_NAMES)):
            pos = basePitch * self.NKinds
            # NOTE: log-ratio preprocessing should be applied to shifted
            # chroma, so we always do it inside loop.
            pre_chromas = self.preprocess(chromas)
            ki = 0
            dist_beta = beta(*self.betaParams)
            for k in self.kinds:
                # TODO:
                partition = [self.inDegreeDict[k], self.outDegreeDict[k]]
                in_chroma_sums = amalgamate(partition, pre_chromas).transpose()[0]
                in_chroma_composition = subcomposition([[e] for e in self.inDegreeDict[k]], pre_chromas)
                correctness = dist_beta.logcdf(in_chroma_sums)
                balance = scipy.stats.dirichlet.logpdf(
                    in_chroma_composition.transpose()[0:-1, :],
                    self.dirichlets[k])
                lps[:, pos + ki] = (correctness + balance)
                ki += 1
            chromas = np.roll(chromas, -1, axis=1)
        if normalize:
            norm_sum = logsumexp(lps, axis=1)
            return lps - norm_sum[:, np.newaxis]
        else:
            return lps


def test_accuracy(predicted_chromas, ground_truth_segments):
    inds = np.where(ground_truth_segments.kinds != 'unclassified')
    wrong_indices = [inds[0][i] for i in np.where(predicted_chromas[inds] !=
                                                  convert_chord_labels(ground_truth_segments.labels[inds]))[0]]
    for i in wrong_indices:
        print(ground_truth_segments.uids[i],
              ground_truth_segments.startTimes[i],
              ground_truth_segments.labels[i],
              ' != ',
              predicted_chromas[i])
    return float(len(inds[0]) - len(wrong_indices)) / len(inds[0])


def load_model(filename):
    return joblib.load(filename)
