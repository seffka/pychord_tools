from collections import Counter
from collections import OrderedDict
import os
import json
import functools
import numpy as np

from . import labels
from . import cacher
from . import common_utils


class SymbolicAnalysisSegment:
    def __init__(self, labels, kind, root, duration, n_beats):
        self.labels = labels
        self.kind = kind
        self.root = root
        self.duration = duration
        self.n_beats = n_beats

    def __repr__(self):
        return str(self.labels) + '\t' + self.kind + '\t' + str(self.duration) + '\t' + self.n_beats


def to_symbolic_analysis_segments(chord_segments, label_translator):
    result = np.empty([len(chord_segments)], dtype='object')
    i = 0
    for segment in chord_segments:
        pitch, kind = label_translator.label_to_pitch_and_kind(segment.symbol)
        duration = float(segment.end_time) - float(segment.start_time)
        result[i] = SymbolicAnalysisSegment(set([segment.symbol]), kind, pitch, duration, 1)
        i += 1
    return result


def merge(symbolic_analysis_segments):
    if len(symbolic_analysis_segments) < 2:
        return symbolic_analysis_segments
    res = []
    current_segment = symbolic_analysis_segments[0]
    for segment in symbolic_analysis_segments[1:]:
        if segment.kind == current_segment.kind and segment.root == current_segment.root:
            current_segment.duration += segment.duration
            current_segment.n_beats += segment.n_beats
        else:
            res.append(current_segment)
            current_segment = segment
    res.append(current_segment)
    return res

INTERVALS = ['P1', 'm2', 'M2', 'm3', 'M3', 'P4', 'd5', 'P5', 'm6', 'M6', 'm7', 'M7']

def to_interval(pitch1, pitch2):
    return INTERVALS[(pitch2 - pitch1) % 12]


def to_sequence(intervals):
    return functools.reduce(lambda x, y: x + "-" + y.split('-', 1)[1], intervals)


def update_n_grams(segments, two_grams, n_grams, limit):
    # ignore unclassified.
    kinds = list(map(lambda x: x.kind, filter(lambda x: x.kind != labels.UNCLASSIFIED, segments)))
    roots = list(map(lambda x: x.root, filter(lambda x: x.kind != labels.UNCLASSIFIED, segments)))
    sequence = np.empty([len(kinds) - 1], dtype='object')
    for i in range(len(sequence)):
        sequence[i] = kinds[i] + '-' + to_interval(roots[i], roots[i + 1]) + '-' + kinds[i + 1]
    for s in sequence:
        two_grams[s] += 1
    for i in range(len(sequence)):
        if i == 0:
            n_grams[sequence[i]] += 1
        for j in range(i, max(0, i - limit), -1):
            n_grams[to_sequence(sequence[j:i + 1])] += 1


def make_transition_c_root_part(two_grams, harmonic_rhythm, label_translator):
    result = np.zeros((label_translator.chords_number(), label_translator.chord_kinds_number()))
    for i in range(label_translator.chord_kinds_number()):
        column = result[:, i]
        sym = tuple(label_translator.chord_kinds())
        related_keys = list(filter(lambda x: x.endswith(sym), two_grams.keys()))
        denominator = float(sum([two_grams[x] for x in related_keys])) * harmonic_rhythm
        # probability for unobserved (i.e. near impossible) events
        if denominator == 0:
            # chord is not used.
            denominator = label_translator.chords_number() * harmonic_rhythm
        p_unobserved = 1.0 / denominator
        for key in related_keys:
            chord, interval, dummy = key.split('-')
            # inverse interval, since we trace it backwrds
            n_interval = -INTERVALS.index(interval) % 12
            n_chord = label_translator.chord_kinds().index(chord)
            pos = label_translator.chord_kinds_number() * n_interval + n_chord
            column[pos] = float(two_grams[key]) / denominator
        # imply harmonic rhythm (chord inertia).
        column[i] = 1.0 - 1.0 / harmonic_rhythm
        column[column == 0] = p_unobserved
        column /= sum(column)
    return result


def harmonic_rhythm_for_file(anno_file_name, label_translator):
    with open(anno_file_name) as json_file:
        data = json.load(json_file)
        duration = float(data['duration'])
        metre_numerator = int(data['metre'].split('/')[0])
        all_beats = []
        all_chords = []
        common_utils.process_parts(metre_numerator, data, all_beats, all_chords, 'chords')
        segments = merge(to_symbolic_analysis_segments(
            common_utils.to_beat_chord_segment_list(0, duration, all_beats, all_chords),
            label_translator))
        # remove unclassified.
        n_beats = list(map(lambda x: x.n_beats, filter(lambda x: x.kind != labels.UNCLASSIFIED, segments)))
        return float(sum(n_beats)) / len(n_beats)

def beats_for_file(anno_file_name):
    with open(anno_file_name) as json_file:
        data = json.load(json_file)
        duration = float(data['duration'])
        metre_numerator = int(data['metre'].split('/')[0])
        all_beats = []
        all_chords = []
        common_utils.process_parts(metre_numerator, data, all_beats, all_chords, 'chords')
        return all_beats

def harmonic_rhythm_for_each_file_in_list(file_list, label_translator):
    result_by_file = {}
    with open(file_list) as list_file:
        for line in list_file:
            infile = line.rstrip()
            basename = os.path.basename(infile)
            name, json_ext = os.path.splitext(basename)
            result_by_file[name] = harmonic_rhythm_for_file(infile, label_translator)
    return result_by_file


class ChordUsageSummary:
    def __init__(self, beats_number, beats_percent, duration_seconds, duration_percent):
        self.beats_number = beats_number
        self.beats_percent = beats_percent
        self.duration_seconds = duration_seconds
        self.duration_percent = duration_percent


class SymbolicStatistics:
    def __init__(self, total_durations, total_beats, label_kinds, no_bass_kinds, chord_summary_dict,
                 unclassified_labels_counter, mean_harmonic_rhythm, mean_bpm,
                 number_of_segments, number_of_distinct_n_grams, max_n_gram_length_within_top):
        self.total_durations = total_durations
        self.total_beats = total_beats
        self.label_kinds = label_kinds
        self.no_bass_kinds = no_bass_kinds
        self.chord_summary_dict = chord_summary_dict
        self.unclassified_labels_counter = unclassified_labels_counter
        self.mean_harmonic_rhythm = mean_harmonic_rhythm
        self.mean_bpm = mean_bpm
        self.number_of_segments = number_of_segments
        self.number_of_distinct_n_grams = number_of_distinct_n_grams
        self.max_n_gram_length_within_top = max_n_gram_length_within_top


@cacher.memory.cache
def estimate_statistics(file_list, label_translator, top=300, max_n_gram=2000):
    all_labels = np.array([], dtype='object')
    all_kinds = np.array([], dtype='object')
    all_roots = np.array([], dtype='object')
    all_durations = np.array([], dtype='float32')
    all_n_beats = np.array([], dtype='int')
    two_grams = Counter()
    n_grams = Counter()
    n_files = 0
    for infile in file_list:
        n_files += 1
        print(infile)
        with open(infile) as json_file:
            data = json.load(json_file)
            duration = float(data['duration'])
            metre_numerator = int(data['metre'].split('/')[0])
            all_beats = []
            all_chords = []
            common_utils.process_parts(metre_numerator, data, all_beats, all_chords, 'chords')
            segments = merge(to_symbolic_analysis_segments(
                common_utils.to_beat_chord_segment_list(0, duration, all_beats, all_chords),
                label_translator))
            update_n_grams(segments, two_grams, n_grams, limit=max_n_gram)
            all_labels = np.append(all_labels, list(map(lambda x: x.labels, segments)))
            all_kinds = np.append(all_kinds, list(map(lambda x: x.kind, segments)))
            all_roots = np.append(all_roots, list(map(lambda x: x.root, segments)))
            all_durations = np.append(all_durations, list(map(lambda x: x.duration, segments)))
            all_n_beats = np.append(all_n_beats, list(map(lambda x: x.n_beats, segments)))

    # output: kinds
    total_duration = sum(all_durations)
    total_beats = sum(all_n_beats)
    label_kinds = Counter()
    no_basslabel_kinds = Counter()

    for labelSet in all_labels:
        for l in labelSet:
            parts = l.split(':')
            if len(parts) == 1:
                label_kinds['maj'] += 1
                no_basslabel_kinds['maj'] += 1
            else:
                label_kinds[parts[1]] += 1
                no_basslabel_kinds[parts[1].split('/')[0]] += 1

    maj_duration = sum(all_durations[all_kinds == 'maj'])
    min_duration = sum(all_durations[all_kinds == 'min'])
    dom_duration = sum(all_durations[all_kinds == 'dom'])
    hdim_duration = sum(all_durations[all_kinds == 'hdim7'])
    dim_duration = sum(all_durations[all_kinds == 'dim'])
    nc_duration = sum(all_durations[all_labels == set('N')])
    unclassified_duration = sum(all_durations[all_kinds == labels.UNCLASSIFIED]) - nc_duration
    maj_beats = sum(all_n_beats[all_kinds == 'maj'])
    min_beats = sum(all_n_beats[all_kinds == 'min'])
    dom_beats = sum(all_n_beats[all_kinds == 'dom'])
    hdim_beats = sum(all_n_beats[all_kinds == 'hdim7'])
    dim_beats = sum(all_n_beats[all_kinds == 'dim'])
    nc_beats = sum(all_n_beats[all_labels == set('N')])
    unclassified_beats = sum(all_n_beats[all_kinds == labels.UNCLASSIFIED]) - nc_beats

    chord_summary_dict = OrderedDict()
    chord_summary_dict['maj'] = ChordUsageSummary(
        maj_beats,  maj_beats * 100.0 / total_beats, maj_duration,  maj_duration * 100.0 / total_duration)
    chord_summary_dict['min'] = ChordUsageSummary(
        min_beats,  min_beats * 100.0 / total_beats, min_duration,  min_duration * 100.0 / total_duration)
    chord_summary_dict['dom'] = ChordUsageSummary(
        dom_beats,  dom_beats * 100.0 / total_beats, dom_duration,  dom_duration * 100.0 / total_duration)
    chord_summary_dict['hdim7'] = ChordUsageSummary(
        hdim_beats,  hdim_beats * 100.0 / total_beats, hdim_duration,  hdim_duration * 100.0 / total_duration)
    chord_summary_dict['dim'] = ChordUsageSummary(
        dim_beats,  dim_beats * 100.0 / total_beats, dim_duration,  dim_duration * 100.0 / total_duration)
    chord_summary_dict['N'] = ChordUsageSummary(
        nc_beats,  nc_beats * 100.0 / total_beats, nc_duration,  nc_duration * 100.0 / total_duration)
    chord_summary_dict['unclassified'] = ChordUsageSummary(
        unclassified_beats,  unclassified_beats * 100.0 / total_beats, unclassified_duration,
        unclassified_duration * 100.0 / total_duration)
    # what's unclassified
    unclassified = np.array([], dtype='object')
    for u in all_labels[all_kinds == labels.UNCLASSIFIED]:
        unclassified = np.append(unclassified, list(u))

    # remove unclassified.
    kinds = all_kinds[all_kinds != labels.UNCLASSIFIED]
    n_beats = all_n_beats[all_kinds != labels.UNCLASSIFIED]

    harmonic_rhythm = float(sum(n_beats)) / len(n_beats)
    transition_c_root_part = make_transition_c_root_part(two_grams, harmonic_rhythm, label_translator)
    transition_matrix = transition_c_root_part
    for i in range(1, labels.N_PITCH_CLASSES):
        block = np.roll(transition_c_root_part, i * label_translator.chord_kinds_number(), 0)
        transition_matrix = np.hstack((transition_matrix, block))

    top_two_grams = two_grams.most_common(top)
    top_n_grams = n_grams.most_common(top)
    max_top_len = max([x[0].count('-')/2 for x in top_n_grams])
    sym_stat = SymbolicStatistics(
        total_duration,
        total_beats,
        label_kinds,
        no_basslabel_kinds,
        chord_summary_dict,
        Counter(unclassified),
        harmonic_rhythm,
        60.0 / (total_duration / total_beats),
        len(kinds),
        len(n_grams),
        max_top_len)

    return sym_stat, top_two_grams, top_n_grams, transition_matrix


def save_transition_matrix(outfile, matrix):
    np.savez(
        outfile,
        matrix=matrix)


def load_transition_matrix(infile):
    data = np.load(infile)
    return data['matrix']
