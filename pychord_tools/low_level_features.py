import json

import essentia
import essentia.standard
import essentia.streaming as esstr
import numpy as np

from .labels import MajMinLabelTranslator, PitchedPattern
from . import cacher
from . import common_utils
from .common_utils import ChordSegment
from .path_db import get_audio_path


######################################################################
# Basic structures
######################################################################


class BeatSegments:
    def __init__(self, start_times, durations):
        self.start_times = start_times
        self.durations = durations


class ChromaSegments(BeatSegments):
    def __init__(self, chromas, start_times, durations):
        self.chromas = chromas
        BeatSegments.__init__(self, start_times, durations)


class AnnotatedChromaSegments(ChromaSegments):
    def __init__(self, labels, pitches, kinds, chromas, uids, start_times, durations):
        self.labels = labels
        self.pitches = pitches
        self.kinds = kinds
        self.uids = uids
        ChromaSegments.__init__(self, chromas, start_times, durations)

    def pitched_patterns(self):
        if len(self.kinds) != len(self.pitches):
            raise ValueError("kinds and pitches vectors need to be equal size.")
        res = np.empty((len(self.pitches)), dtype='object')
        for i in range(len(self.pitches)):
            res[i] = PitchedPattern(self.kinds[i], pitch_class_index=self.pitches[i])
        return res


######################################################################
# Interfaces
######################################################################

class BeatSegmentsEstimator:
    def estimate_beats(self, uid):
        pass


class ChromaEstimator:
    def __init__(self, frame_size=16384, hop_size=2048, sample_rate=44100):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

    def estimate_chroma(self, uid):
        pass


class SegmentChromaEstimator:
    def __init__(self, frame_size=16384, hop_size=2048, sample_rate=44100):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

    def fill_segments_with_chroma(self, beat_segments, chroma):
        pass

    def get_chroma_by_beats(self, beats, chroma):
        pass


class UidAndAudioPathExtractor:
    def uid_and_audio_path_name(self, annotation_file_name):
        pass

######################################################################


def smooth(x, window_len=11, window='hanning'):
    """Smooth the data using a window with requested size.
    Borrowed from http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x : numpy.array(dtype=float)
        2-dimensional data (smoothing is performaed along axis 0
        for each component along axis 1)
    window_len: int
        the dimension of the smoothing window; should be an odd integer
    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    Returns
    -------
    x: numpy.array
            the smoothed signal
    """
    y = np.zeros(x.shape)
    for i in range(np.size(x, 1)):
        if np.size(x, 0) < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
        if window_len < 3:
            return x
        if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        xx = x[:, i]
        s = np.r_[xx[window_len - 1:0:-1], xx, xx[-1:-window_len:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        start = int(window_len / 2)
        end = start + len(xx)
        y[:, i] = np.convolve(w / w.sum(), s, mode='valid')[start:end]
    return y

######################################################################


class DefaultUidAndAudioPathExtractor(UidAndAudioPathExtractor):
    def uid_and_audio_path_name(self, annotation_file_name):
        with open(annotation_file_name) as json_file:
            data = json.load(json_file)
            uid = data['mbid']
            audio_path = get_audio_path(uid)
            return uid, audio_path


class HPCPChromaEstimator(ChromaEstimator):
    """
    Extract HPCP chroma features with essentia
    Parameters
    ----------
    frame_size : int
        Analysis frame size (samples)
    tuning_freq : float
        tuning frequency (Hz)
    hop_size : int
        Hop size (in samples)
    """
    def __init__(
            self,
            tuning_freq=440,
            frame_size=16384,
            hop_size=2048,
            sample_rate=44100,
            order_by="magnitude",
            magnitude_threshold=1e-05,
            min_frequency=40,
            max_frequency=5000,
            max_peaks=10000,
            size=12,
            harmonics=8,
            band_preset=True,
            band_split_frequency=500.0,
            weight_type="cosine",
            non_linear=True,
            window_size=1.0):
        super().__init__(frame_size, hop_size, sample_rate)
        self.tuning_freq = tuning_freq
        self.order_by = order_by
        self.magnitude_threshold = magnitude_threshold
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.max_peaks = max_peaks
        self.size = size
        self.harmonics = harmonics
        self.band_preset = band_preset
        self.band_split_frequency = band_split_frequency
        self.weight_type = weight_type
        self.non_linear = non_linear
        self.window_size = window_size

    def estimate_chroma(self, uid):
        loader = esstr.MonoLoader(filename=get_audio_path(uid))
        framecutter = esstr.FrameCutter(hopSize=self.hop_size, frameSize=self.frame_size)
        windowing = esstr.Windowing(type="blackmanharris62")
        spectrum = esstr.Spectrum()
        spectralpeaks = esstr.SpectralPeaks(orderBy="magnitude",
                                            magnitudeThreshold=1e-05,
                                            minFrequency=40,
                                            maxFrequency=5000,
                                            maxPeaks=10000)
        hpcp = esstr.HPCP(size=12,
                          referenceFrequency=self.tuning_freq,
                          harmonics=8,
                          bandPreset=True,
                          minFrequency=float(40),
                          maxFrequency=float(5000),
                          bandSplitFrequency=500.0,
                          weightType="cosine",
                          nonLinear=True,
                          windowSize=1.0)
        """
        hpcp = esstr.HPCP(
            size=12,
            referenceFrequency = tuningFreq,
            harmonics = 8,
            bandPreset = True,
            minFrequency = 40.0,
            maxFrequency = 5000.0,
            bandSplitFrequency = 250.0,
            weightType = "cosine",
            nonLinear = False,
            windowSize = 1.0)
        """
        pool = essentia.Pool()
        # connect algorithms together
        loader.audio >> framecutter.signal
        framecutter.frame >> windowing.frame >> spectrum.frame
        spectrum.spectrum >> spectralpeaks.spectrum
        spectrum.spectrum >> (pool, 'spectrum.magnitude')
        spectralpeaks.magnitudes >> hpcp.magnitudes
        spectralpeaks.frequencies >> hpcp.frequencies
        hpcp.hpcp >> (pool, 'chroma.hpcp')

        essentia.run(loader)
        # roll from 'A' based to 'C' based
        chroma = pool['chroma.hpcp']
        chroma = np.roll(chroma, shift=-3, axis=1)
        return chroma


class SmoothedStartingBeatChromaEstimator(SegmentChromaEstimator):
    def __init__(self, frame_size=16384, hop_size=2048, sample_rate=44100, smoothing_time=1.25):
        super().__init__(frame_size, hop_size, sample_rate)
        self.smoothingTime = smoothing_time

    def fill(self, beats, chroma, smoothed_chromas):
        for i in range(len(beats)):
            s = int(float(beats[i]) *
                    self.sample_rate / self.hop_size)
            smoothed_chromas[i] = chroma[s]

    def fill_segments_with_chroma(self, segments, chroma):
        chroma = smooth(
            chroma,
            window_len=int(self.smoothingTime * self.sample_rate / self.hop_size),
            window='hanning').astype('float32')
        segments.chromas = np.zeros((len(segments.start_times), 12), dtype='float32')
        self.fill(segments.start_times, chroma, segments.chromas)

    def get_chroma_by_beats(self, beats, chroma):
        chroma = smooth(
            chroma,
            window_len=int(self.smoothingTime * self.sample_rate / self.hop_size),
            window='hanning').astype('float32')
        res = np.zeros((len(beats), 12), dtype='float32')
        self.fill(beats, chroma, res)
        return res


class AnnotatedBeatChromaEstimator:
    def __init__(self,
                 chroma_estimator=HPCPChromaEstimator(),
                 uid_and_audio_path_extractor=DefaultUidAndAudioPathExtractor(),
                 segment_chroma_estimator=SmoothedStartingBeatChromaEstimator(),
                 label_translator=MajMinLabelTranslator(),
                 roll_to_c_root=True):
        self.chroma_estimator = chroma_estimator
        self.uid_and_audio_path_extractor = uid_and_audio_path_extractor
        self.beat_chroma_estimator = segment_chroma_estimator
        self.label_translator = label_translator
        self.roll_to_c_root = roll_to_c_root

    def load_chromas_for_annotation_file_list(self, file_list):
        res = AnnotatedChromaSegments(
            labels=np.array([], dtype='object'),
            pitches=np.array([], dtype='int'),
            kinds=np.array([], dtype='object'),
            chromas=np.zeros((0, 12), dtype='float32'),
            uids=np.array([], dtype='object'),
            start_times=np.array([], dtype='float32'),
            durations=np.array([], dtype='float32'))
        for file in file_list:
            chunk = self.load_chromas_for_annotation_file(file)
            res.chromas = np.concatenate((res.chromas, chunk.chromas))
            res.labels = np.concatenate((res.labels, chunk.labels))
            res.pitches = np.concatenate((res.pitches, chunk.pitches))
            res.kinds = np.concatenate((res.kinds, chunk.kinds))
            res.uids = np.concatenate((res.uids, chunk.uids))
            res.start_times = np.concatenate((res.start_times, chunk.start_times))
            res.durations = np.concatenate((res.durations, chunk.durations))
        return res

    # returns AnnotatedChromaSegments for the file list
    def load_chromas_for_annotation_file_list_file(self, file_list_file):
        return self.load_chromas_for_annotation_file_list(
            common_utils.load_file_list(file_list_file))

    def load_beats_and_annotations(self, json_file_name, uid):
        with open(json_file_name) as json_file:
            print(json_file_name)
            data = json.load(json_file)
            uid = uid
            duration = float(data['duration'])
            metre_numerator = int(data['metre'].split('/')[0])
            all_beats = []
            all_chords = []
            common_utils.process_parts(metre_numerator, data, all_beats, all_chords, 'chords')
            segments = common_utils.to_beat_chord_segment_list(all_beats[0], duration, all_beats, all_chords)
            #
            chromas = None
            labels = np.empty(len(segments), dtype='object')
            pitches = np.empty(len(segments), dtype='int')
            kinds = np.empty(len(segments), dtype='object')
            uids = np.empty(len(segments), dtype='object')
            start_times = np.zeros(len(segments), dtype='float32')
            durations = np.zeros(len(segments), dtype='float32')
            for i in range(len(segments)):
                pitch, kind = self.label_translator.label_to_pitch_and_kind(segments[i].symbol)
                s = int(float(segments[i].start_time) *
                        self.chroma_estimator.sample_rate / self.chroma_estimator.hop_size)
                e = int(float(segments[i].end_time) *
                        self.chroma_estimator.sample_rate / self.chroma_estimator.hop_size)
                if s == e:
                    print("empty segment ", segments[i].start_time, segments[i].end_time)
                    raise Exception("empty segment")
                labels[i] = segments[i].symbol
                pitches[i] = pitch
                kinds[i] = kind
                uids[i] = uid
                start_times[i] = segments[i].start_time
                durations[i] = float(segments[i].end_time) - float(segments[i].start_time)
            return AnnotatedChromaSegments(labels, pitches, kinds, chromas, uids, start_times, durations)

    def load_chromas_for_annotation_file(self, annotation_file_name):
        uid, audio_file_name = self.uid_and_audio_path_extractor.uid_and_audio_path_name(
            annotation_file_name)
        chroma = self.chroma_estimator.estimate_chroma(uid)
        annotated_chroma_segments = self.load_beats_and_annotations(annotation_file_name, uid)
        self.beat_chroma_estimator.fill_segments_with_chroma(annotated_chroma_segments, chroma)

        if self.roll_to_c_root:
            for i in range(len(annotated_chroma_segments.chromas)):
                shift = 12 - annotated_chroma_segments.pitches[i]
                annotated_chroma_segments.chromas[i] = np.roll(
                    annotated_chroma_segments.chromas[i], shift=shift)
        return annotated_chroma_segments


class BeatChromaEstimator:
    def __init__(self,
                 beats,
                 pitched_patterns,
                 duration,
                 chroma_estimator=HPCPChromaEstimator(),
                 beat_chroma_estimator=SmoothedStartingBeatChromaEstimator(),
                 uid=""):
        self.beats = np.concatenate((beats, [duration]))
        self.pitched_patterns = pitched_patterns
        self.chroma_estimator = chroma_estimator
        self.duration = duration
        self.uid = uid
        self.beat_chroma_estimator = beat_chroma_estimator

    def load_chromas(self, uid):
        segments = []
        for i in range(len(self.pitched_patterns)):
            sym = str(self.pitched_patterns[i])
            segments.append(ChordSegment(self.beats[i], self.beats[i + 1], sym))

        labels = np.empty(len(segments), dtype='object')
        pitches = np.empty(len(segments), dtype='int')
        kinds = np.empty(len(segments), dtype='object')
        uids = np.empty(len(segments), dtype='object')
        start_times = np.zeros(len(segments), dtype='float32')
        durations = np.zeros(len(segments), dtype='float32')
        for i in range(len(segments)):
            s = int(float(segments[i].start_time) *
                    self.chroma_estimator.sample_rate / self.chroma_estimator.hop_size)
            e = int(float(segments[i].end_time) *
                    self.chroma_estimator.sample_rate / self.chroma_estimator.hop_size)
            if s == e:
                print("empty segment ", segments[i].start_time, segments[i].end_time)
                raise Exception("empty segment")
            labels[i] = segments[i].symbol
            pitches[i] = self.pitched_patterns[i].pitch_class_index
            kinds[i] = self.pitched_patterns[i].kind
            uids[i] = self.uid
            start_times[i] = segments[i].start_time
            durations[i] = float(segments[i].end_time) - float(segments[i].start_time)
        annotated_chroma_segments = AnnotatedChromaSegments(
            labels, pitches, kinds, None, uids, start_times, durations)
        self.beat_chroma_estimator.fill_segments_with_chroma(
            annotated_chroma_segments, self.chroma_estimator.estimate_chroma(get_audio_path(uid)))

        return annotated_chroma_segments


@cacher.memory.cache(ignore=['sample_rate', 'audio_samples'])
def audio_duration(uid, sample_rate=44100, audio_samples=None):
    if audio_samples is not None:
        return float(len(audio_samples)) / sample_rate
    else:
        audio = essentia.standard.MonoLoader(filename=get_audio_path(uid), sampleRate=sample_rate)()
        return float(len(audio)) / sample_rate
