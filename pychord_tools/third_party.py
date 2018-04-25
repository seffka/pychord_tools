import madmom.features as mf
import numpy as np
import vamp
import pychord_tools.cacher as cacher
import essentia
from pychord_tools.lowLevelFeatures import audioDuration
from pychord_tools.lowLevelFeatures import ChromaEstimator

@cacher.memory.cache
def nnlsChromaFromAudio(audiofile, sampleRate=44100, stepSize=2048):
    mywindow = np.array(
        [0.001769, 0.015848, 0.043608, 0.084265, 0.136670, 0.199341, 0.270509, 0.348162, 0.430105, 0.514023,
         0.597545, 0.678311, 0.754038, 0.822586, 0.882019, 0.930656, 0.967124, 0.990393, 0.999803, 0.999803,
         0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803,
         0.999803, 0.999803, 0.999803, 0.999803, 0.999803,
         0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999650, 0.996856, 0.991283,
         0.982963, 0.971942, 0.958281, 0.942058, 0.923362, 0.902299, 0.878986, 0.853553, 0.826144,
         0.796910, 0.766016, 0.733634, 0.699946, 0.665140, 0.629410, 0.592956, 0.555982, 0.518696,
         0.481304, 0.444018, 0.407044, 0.370590, 0.334860, 0.300054, 0.266366, 0.233984, 0.203090,
         0.173856, 0.146447, 0.121014, 0.097701, 0.076638, 0.057942, 0.041719, 0.028058, 0.017037,
         0.008717, 0.003144, 0.000350])
    audio = essentia.standard.MonoLoader(filename=audiofile, sampleRate=sampleRate)()
    # estimate audio duration just for caching purposes:
    audioDuration(audiofile, sampleRate=sampleRate, audioSamples=audio)

    stepsize, semitones = vamp.collect(
        audio, sampleRate, "nnls-chroma:nnls-chroma", output="semitonespectrum", step_size=stepSize)["matrix"]
    chroma = np.zeros((semitones.shape[0], 12))
    for i in range(semitones.shape[0]):
        tones = semitones[i] * mywindow
        cc = chroma[i]
        for j in range(tones.size):
            cc[j % 12] = cc[j % 12] + tones[j]
    # roll from 'A' based to 'C' based
    chroma = np.roll(chroma, shift=-3, axis=1)
    return chroma

class NNLSChromaEstimator(ChromaEstimator):
    def __init__(self, hopSize = 2048, sampleRate = 44100):
        super().__init__(16384, hopSize, sampleRate)

    def estimateChroma(self, audioFileName):
        return nnlsChromaFromAudio(audioFileName, self.sampleRate, self.hopSize)


@cacher.memory.cache
def rnnBeatSegments(audioFileName):
    proc = mf.BeatTrackingProcessor(
        fps = 100,
        method='comb', min_bpm=40,
        max_bpm=240, act_smooth=0.09,
        hist_smooth=7, alpha=0.79)
    act = mf.RNNBeatProcessor()(str(audioFileName))
    stamps = proc(act).astype('float32')
    # the last beat is lost, but who cares...
    # TODO: fix this approach
    return np.array(stamps[0:-1])
