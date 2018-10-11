import re
import numpy as np

PITCH_CLASS_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
N_PITCH_CLASSES = len(PITCH_CLASS_NAMES)
DEGREES = ['I', 'IIb', 'II', 'IIIb', 'III', 'IV', 'Vb', 'V', 'VIb', 'VI', 'VIIb', 'VII']
PITCHES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
ALT = {'b': -1, '#': 1}
SHORTCUTS = {'maj': '(3,5)', 'min': '(b3,5)', 'dim': '(b3,b5)', 'aug': '(3,#5)', 'maj7': '(3,5,7)',
             'min7': '(b3,5,b7)', '7': '(3,5,b7)', 'dim7': '(b3,b5,bb7)', 'hdim7': '(b3,b5,b7)',
             'minmaj7': '(b3,5,7)', 'maj6': '(3,5,6)', 'min6': '(b3,5,6)', '9': '(3,5,b7,9)',
             'maj9': '(3,5,7,9)', 'min9': '(b3,5,b7,9)', 'sus4': '(4,5)'}
UNCLASSIFIED = 'unclassified'
NO_CHORD = 'N'


class LabelTranslator:
    def chord_kinds(self):
        pass

    def chord_mirex_kinds(self):
        pass

    def chord_names(self):
        pass

    def chords_number(self):
        pass

    def chord_kinds_number(self):
        pass

    def label_to_pitch_and_kind(self, label):
        pass


def note_to_number(note):
    pitch = PITCHES[note[0]]
    if len(note) >= 2:
        for i in range(1, len(note)):
            pitch += ALT[note[i]]
    return pitch


JAZZ5_KINDS = ["maj", "min", "dom", "hdim7", "dim"]
JAZZ5_MIREX_KINDS = ["", ":min", ":7", ":hdim7", ":dim"]
JAZZ5_NAMES = np.empty(60, dtype='object')

for p in range(len(PITCH_CLASS_NAMES)):
    for c in range(len(JAZZ5_MIREX_KINDS)):
        JAZZ5_NAMES[p * 5 + c] = PITCH_CLASS_NAMES[p] + JAZZ5_MIREX_KINDS[c]


class Jazz5LabelTranslator(LabelTranslator):
    def chords_number(self):
        return len(JAZZ5_NAMES)

    def chord_names(self):
        return JAZZ5_NAMES

    def chord_mirex_kinds(self):
        return JAZZ5_MIREX_KINDS

    def chord_kinds_number(self):
        return len(JAZZ5_KINDS)

    def chord_kinds(self):
        return JAZZ5_KINDS

    def label_to_pitch_and_kind(self, label):
        parts_and_bass = label.split('/')
        parts = parts_and_bass[0].split(':')
        note = parts[0]
        if note[0] == 'N':
            return 9, UNCLASSIFIED
        pitch = note_to_number(note)
        if len(parts) == 1:
            kind = 'maj'
        else:
            kind = parts[1].split('/')[0]
        if kind in SHORTCUTS:
            kind = SHORTCUTS[kind]
        degrees = set(re.sub("[\(\)]", "", kind).split(','))
        # TODO after the dataset is fixed (bass -> pitch class set).
        if len(parts_and_bass) > 1:
            degrees.add(parts_and_bass[1])
        if '3' in degrees:
            if 'b7' in degrees:
                kind = 'dom'
            else:
                kind = 'maj'
        elif 'b3' in degrees:
            if 'b5' in degrees:
                if 'b7' in degrees:
                    kind = 'hdim7'
                else:
                    kind = 'dim'
            else:
                kind = 'min'
        else:
            kind = UNCLASSIFIED
        return pitch, kind


MAJ_MIN_KINDS = ['maj', 'min']
MAJ_MIN_MIREX_KINDS = ['', ':min']
MAJ_MIN_NAMES = np.empty(24, dtype='object')
for p in range(len(PITCH_CLASS_NAMES)):
    for c in range(len(MAJ_MIN_MIREX_KINDS)):
        MAJ_MIN_NAMES[p * 2 + c] = PITCH_CLASS_NAMES[p] + MAJ_MIN_MIREX_KINDS[c]


class MajMinLabelTranslator(LabelTranslator):
    MAJ_DEGREES = set(('3', '5'))
    MIN_DEGREES = set(('b3', '5'))

    def chord_kinds(self):
        return MAJ_MIN_KINDS

    def chord_mirex_kinds(self):
        return MAJ_MIN_MIREX_KINDS

    def chord_names(self):
        return MAJ_MIN_NAMES

    def chord_kinds_number(self):
        return len(MAJ_MIN_KINDS)

    def chords_number(self):
        return len(MAJ_MIN_NAMES)

    def label_to_pitch_and_kind(self, label):
        parts_and_bass = label.split('/')
        parts = parts_and_bass[0].split(':')
        note = parts[0]
        if note[0] == 'N':
            return 9, 'unclassified'
        pitch = note_to_number(note)
        if len(parts) == 1:
            kind = 'maj'
        else:
            kind = parts[1].split('/')[0]
        if kind in SHORTCUTS:
            kind = SHORTCUTS[kind]
        degrees = set(re.sub("[\(\)]", "", kind).split(','))
        # TODO after the dataset is fixed (bass -> pitch class set).
        if len(parts_and_bass) > 1:
            degrees.add(parts_and_bass[1])
        if len(parts_and_bass) > 1:
            degrees.add(parts_and_bass[1])
        if degrees == self.MAJ_DEGREES:
            kind = 'maj'
        elif degrees == self.MIN_DEGREES:
            kind = 'min'
        else:
            kind = UNCLASSIFIED
        return pitch, kind


class PitchedPattern:
    def __init__(self, kind, pitch_class=None, pitch_class_index=0):
        self.kind = kind
        if pitch_class is not None:
            self.pitch_class_index = PITCH_CLASS_NAMES.index(convert_chord_labels(pitch_class))
        else:
            self.pitch_class_index = pitch_class_index

    def __repr__(self):
        return PITCH_CLASS_NAMES[self.pitch_class_index] + ':' + self.kind

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.kind == other.kind and \
                   self.pitch_class_index == other.pitch_class_index
        return False


def degree_indices(degree_name_list):
    return [DEGREES.index(e) for e in degree_name_list]


def convert_chord_labels(syms):
    # "minor" to Harte syntax, resolve enharmonicity in "jazz" style.
    res = [re.sub('m$', ':min', s) for s in syms]
    res = [re.sub('Gb', 'F#', s) for s in res]
    res = [re.sub('A#', 'Bb', s) for s in res]
    res = [re.sub('C#', 'Db', s) for s in res]
    res = [re.sub('D#', 'Eb', s) for s in res]
    res = [re.sub('G#', 'Ab', s) for s in res]
    return res