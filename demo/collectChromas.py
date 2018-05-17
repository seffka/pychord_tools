import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import beta
from scipy.stats import dirichlet
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

import pychord_tools.plots as plots
from pychord_tools.compositionalData import alr
from pychord_tools.compositionalData import alrinv
from pychord_tools.compositionalData import amalgamate
from pychord_tools.compositionalData import subcomposition
from pychord_tools.compositionalData import substituteZeros
from pychord_tools.lowLevelFeatures import AnnotatedBeatChromaEstimator
from pychord_tools.lowLevelFeatures import SmoothedStartingBeatChromaEstimator
from pychord_tools.lowLevelFeatures import DEGREES, degreeIndices
from pychord_tools.third_party import NNLSChromaEstimator

chromaEstimator = AnnotatedBeatChromaEstimator(
    chromaEstimator = NNLSChromaEstimator(),
    segmentChromaEstimator = SmoothedStartingBeatChromaEstimator(smoothingTime = 0.6))
segments = chromaEstimator.loadChromasForAnnotationFileListFile('correct.txt')
#segments = chromaEstimator.loadChromasForAnnotationFileList(['annotations/berklee_demo.json', 'annotations/seva_demo.json'])
#segments = chromaEstimator.loadChromasForAnnotationFileList(['annotations/seva_demo.json'])
dmaj = preprocessing.normalize(substituteZeros(segments.chromas[segments.kinds == 'maj']), norm='l1')
dmin = preprocessing.normalize(substituteZeros(segments.chromas[segments.kinds == 'min']), norm='l1')

dMaj = pd.DataFrame(data=dmaj, columns=DEGREES)
sns.violinplot(data=dMaj, inner="point")
plt.show()

# TODO: degree captions,
sns.violinplot(data=dmin, inner="point")
plt.show()


# Beta
fig, ax = plt.subplots(1,2)
majPartition = [
    degreeIndices(['I', 'III', 'V']),
    degreeIndices(['II', 'VII', 'IIb','IIIb']),
    degreeIndices(['IV', 'Vb', 'VIb', 'VI', 'VIIb'])]
pmaj = amalgamate(majPartition, dmaj).transpose()[0]
params = beta.fit(pmaj, floc=0, fscale=1)
ax[0].hist(pmaj, 12, normed=True)
rv = beta(*params)
x = np.linspace(0,1)
ax[0].plot(x, rv.pdf(x), lw=2)
ax[0].plot(x, rv.cdf(x), lw=2)

ax[1].hist(np.random.beta(params[0], params[1], 100), 12, normed=True)
ax[1].plot(x, rv.pdf(x), lw=2)
ax[1].plot(x, rv.cdf(x), lw=2)

plt.show()

# log normal
chordChroma = subcomposition([[DEGREES.index('I')], [DEGREES.index('V')], [DEGREES.index('III')]],
    dmaj)
vectors = np.apply_along_axis(alr, 1, chordChroma)
gmm = GaussianMixture(
               n_components=1,
               covariance_type='full',
               max_iter=200)
gmm.fit(vectors)
gen = gmm.sample(100)
genChroma =  np.apply_along_axis(alrinv, 1, gen[0])


fig, ax = plt.subplots(1,2)
plots.ternaryPlot(ax[0], chordChroma, 12)
plots.ternaryPlot(ax[1], genChroma, 12)
plt.show()

# Dirichlet
outChroma = subcomposition([[DEGREES.index('IIb')], [DEGREES.index('IIIb')], [DEGREES.index('IV')], [DEGREES.index('Vb')], [DEGREES.index('VIb')], [DEGREES.index('VI')], [DEGREES.index('VIIb')]],
                             dmaj).astype('float64')
alphas = np.arange(0.1, 2.0, 0.1)
ss = []
for alpha in alphas:
    s = 0
    for x in outChroma:
        s += dirichlet.logpdf(x / np.sum(x), alpha * np.ones(7))
    print(alpha, s)
    ss.append(s)
ss = np.array(ss)
i = np.argmax(ss)
print('max: ', alphas[i], ss[i])

fig, ax = plt.subplots(1,2)
plots.plotHexagram(ax[0], dmaj, degrees = ['Vb', 'VIb', 'IV', 'VI', 'IIb', 'VIIb', 'IIIb'], step = 5)
gen = dirichlet.rvs(1.9 * np.ones(7), 20000)[:, 0:3]
plots.ternaryPlot(ax[1], preprocessing.normalize(gen, norm='l1'), 40)
plt.show()


##############################################


pmaj = amalgamate([[DEGREES.index('I'), DEGREES.index('III'), DEGREES.index('V')]], dmaj)[0]
sns.distplot(pmaj)
plt.show()

params = beta.fit(pmaj, floc=0, fscale=1)
myHist = plt.hist(pmaj, 12, normed=True)
rv = beta(*params)
x = np.linspace(0,1)
h = plt.plot(x, rv.pdf(x), lw=2)
h = plt.plot(x, rv.cdf(x), lw=2)
plt.show()

minPartition = degreeIndices(['I', 'IIIb', 'V'])
pmin = amalgamate([minPartition], dmin)[0]
sns.distplot(pmin)
plt.show()

params = beta.fit(pmin, floc=0, fscale=1)
myHist = plt.hist(pmin, 7, normed=True)
rv = beta(*params)
x = np.linspace(0,1)
h = plt.plot(x, rv.pdf(x), lw=2)
h = plt.plot(x, rv.cdf(x), lw=2)
plt.show()

# Beta. estimate.
a = np.concatenate((pmaj, pmin))
params = beta.fit(a, floc=0, fscale=1)
myHist = plt.hist(a, 7, normed=True)
rv = beta(*params)
x = np.linspace(0,1)
h = plt.plot(x, rv.pdf(x), lw=2)
h = plt.plot(x, rv.cdf(x), lw=2)
plt.show()

# TODO: emulate

# TODO: Dirichlet + estimate parameters + emulation

# TODO: log normal + estimate parameters + emulation
fig, ax = plt.subplots()
plots.degreesTernaryPlot(ax, dmaj, 'I', 'V', 'III', 30)
plots.plotLabels(ax, ['I', 'V', 'III'])
plt.show()

fig, ax = plt.subplots()
plots.degreesTernaryPlot(ax, dmin, 'I', 'V', 'IIIb', 30)
plots.plotLabels(ax, ['I', 'V', 'IIIb'])
plt.show()

# TODO: probabilistic model. paint "confidence".

#chromas = logNormalize(chromas)
#dMaj = pd.DataFrame(data=chromas[kinds =='maj'], columns=degrees)
#dMaj < 0
#sns.violinplot(data=dMaj, inner="point")
#plt.show()


# partitions with overtones

d = plots.sortedDegrees(dmaj, flip=True)
print(d)
majPartition = [
    [DEGREES.index('I'), DEGREES.index('III'), DEGREES.index('V')],
    [DEGREES.index('II'), DEGREES.index('VII')],
    [DEGREES.index('IIb'), DEGREES.index('IIIb'), DEGREES.index('IV'), DEGREES.index('Vb'), DEGREES.index('VIb'), DEGREES.index('VI'), DEGREES.index('VIIb')]]
pmaj = amalgamate(majPartition, dmaj)
fig, ax = plt.subplots()
plots.ternaryPlot(ax, pmaj, 30)
plt.show()

minPartition = [
    [DEGREES.index('I'), DEGREES.index('IIIb'), DEGREES.index('V')],
    [DEGREES.index('II'), DEGREES.index('VIIb')],
    [DEGREES.index('IIb'), DEGREES.index('III'), DEGREES.index('IV'), DEGREES.index('Vb'), DEGREES.index('VIb'), DEGREES.index('VI'), DEGREES.index('VII')]]
pmin = amalgamate(minPartition, dmin)
fig, ax = plt.subplots()
plots.ternaryPlot(ax, pmin, 30)
plt.show()

fig, ax = plt.subplots()
plots.plotHexagram(ax, dmaj, degrees = ['Vb', 'VIb', 'IV', 'VI', 'IIb', 'VIIb', 'IIIb'])
plt.show()

fig, ax = plt.subplots()
plots.plotMajHexagram(ax, dmaj)
plt.show()

rmaj = subcomposition(
    [[DEGREES.index('IIb')], [DEGREES.index('VIIb')], [DEGREES.index('IIIb')]],
    dmaj)
fig, ax = plt.subplots()
plots.ternaryPlot(ax, rmaj, 5)
plt.show()


overtones = subcomposition(
    [[DEGREES.index('II')], [DEGREES.index('VII')]],
    dmaj).transpose()[0]
sns.distplot(overtones)
plt.show()


params = beta.fit(overtones, floc=0, fscale=1)
myHist = plt.hist(overtones, 7, normed=True)
rv = beta(*params)
x = np.linspace(0,1)
h = plt.plot(x, rv.pdf(x), lw=2)
h = plt.plot(x, rv.cdf(x), lw=2)
plt.show()


params = beta.fit(pmaj, floc=0, fscale=1)
myHist = plt.hist(pmaj, 9, normed=True)
rv = beta(*params)
x = np.linspace(0,1)
h = plt.plot(x, rv.pdf(x), lw=2)
h = plt.plot(x, rv.cdf(x), lw=2)
plt.show()

minPartition = [DEGREES.index('I'), DEGREES.index('V'), DEGREES.index('IIIb'), DEGREES.index('II'), DEGREES.index('VIIb')]
pmin = plots.estimatePartition(minPartition, dmin)
sns.distplot(pmin)
plt.show()

params = beta.fit(pmin, floc=0, fscale=1)
myHist = plt.hist(pmin, 7, normed=True)
rv = beta(*params)
x = np.linspace(0,1)
h = plt.plot(x, rv.pdf(x), lw=2)
h = plt.plot(x, rv.cdf(x), lw=2)
plt.show()
