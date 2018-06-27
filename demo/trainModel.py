from pychord_tools.lowLevelFeatures import AnnotatedBeatChromaEstimator
from pychord_tools.lowLevelFeatures import SmoothedStartingBeatChromaEstimator
from pychord_tools.third_party import NNLSChromaEstimator
from pychord_tools.models import CosineSimilarityBinaryPatternModel, CorrectnessLogNormBalanceModel, testAccuracy, IndependentPDFModel, IndependentIntegralModel

chromaEstimator = AnnotatedBeatChromaEstimator(
    chromaEstimator = NNLSChromaEstimator(),
    segmentChromaEstimator = SmoothedStartingBeatChromaEstimator(smoothingTime = 0.6))
segments = chromaEstimator.loadChromasForAnnotationFileListFile('correct.txt')

m = CorrectnessLogNormBalanceModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
m.fit(segments)
m.saveModel('../pychord_tools/clnm.pkl')

m = IndependentPDFModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
m.fit(segments)
m.saveModel('../pychord_tools/ind_pdf.pkl')

mi = IndependentIntegralModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
mi.fit(segments)
mi.saveModel('../pychord_tools/ind_integral.pkl')
