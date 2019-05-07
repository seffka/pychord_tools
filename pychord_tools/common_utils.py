import json
import re
import numpy as np

def load_file_list(list_file_name):
    result = []
    with open(list_file_name) as list_file:
        for line in list_file:
            result.append(line.rstrip())
    return result


def load_file_list_plus_prefix(list_file_name, path_prefix):
    result = []
    with open(list_file_name) as list_file:
        for line in list_file:
            result.append(line.rstrip())
    return [path_prefix + x for x in result]


def all_divisors(n, start_with=1):
    i = start_with
    if n < i:
        return set()
    if n % i == 0:
        return set((i, n / i)) | all_divisors(n / i, i + 1)
    else:
        return all_divisors(n, i + 1)


def merge_segments(chord_segments):
    if len(chord_segments) < 2:
        return chord_segments
    res = []
    current_segment = chord_segments[0]
    for segment in chord_segments[1:]:
        if segment.symbol == current_segment.symbol:
            current_segment.end_time = segment.end_time
        else:
            res.append(current_segment)
            current_segment = segment
    res.append(current_segment)
    return res


def is_in_chord_mode(chords):
    for c in chords:
        if len(c.split(':')) > 2:
            return False
    return True


def process_chords(numerator, blocks, all_bars, all_chords, all_events, all_beats):
    n = 0

    for block in blocks:
        bars = block.split('|')[1:-1]
        for bar in bars:
            chords = [c for c in re.split('\s+', bar) if c != '']
            beats = all_beats[n*numerator:(n+1) * numerator]
            if all_bars is not None:
                if len(all_bars) > 0:
                    all_bars[-1][1] = beats[0]
                #print('all_bars', all_bars)
                all_bars.append([beats[0], beats[0]])
                if len(beats) > 1:
                    all_bars[-1][1] = beats[-1] + (beats[-1] - beats[-2])

            extended_beats = np.array(beats)
            if (n+1) * numerator < len(all_beats):
                extended_beats = np.append(extended_beats, all_beats[(n+1) * numerator])
            else:
                extended_beats = np.append(extended_beats, extended_beats[-1] + (extended_beats[1:]-extended_beats[:-1]).mean())
            if is_in_chord_mode(chords):
                divisors = all_divisors(numerator)
                if not (len(chords) in divisors):
                    raise ValueError("Wrong number of chords in a bar: " + bar)
                multiplier = numerator // len(chords)
                newchords = []
                for c in chords:
                    newchords.extend([c] * multiplier)
                all_chords.extend(newchords)
                all_events.extend(beats)
            else:
                # events mode
                bar_events = []
                pure_chords = []
                pos = 0
                for c in chords:
                    components = c.split(':')
                    pure_chords.append(':'.join(components[0:2]))
                    bar_events.append(pos)
                    # TODO: supports non-4 denominators
                    if len(components) > 2:
                        pos += 4.0 / int(components[2])
                    else:
                        pos += 1
                if abs(pos - numerator) > 0.001:
                    raise ValueError("Inapropriate bar length: " + pos)
                bar_events = np.array(bar_events, dtype=float)
                conditions = []
                funcs = []
                for i in range(numerator):
                    conditions.append((bar_events >= i) & (bar_events < (i+1)))
                    funcs.append(lambda x, i=i: extended_beats[i] + (extended_beats[i + 1] - extended_beats[i]) * (x - float(i)))
                all_chords.extend(pure_chords)
                all_events.extend(np.piecewise(bar_events, conditions, funcs))
            n += 1


def process_parts(metre_numerator, data, events, chords, choice, beatz = None, bars = None):
    if 'parts' in data.keys():
        for part in data['parts']:
            process_parts(metre_numerator, part, events, chords, choice, beatz, bars)
    else:
        if 'metre' in data:
            metre_numerator = int(data['metre'].split('/')[0])
        if beatz is not None:
            beatz.extend(data['beats'])
        process_chords(metre_numerator, data[choice], bars, chords, events, data['beats'])


class ChordSegment:
    start_time = 0.0
    end_time = 0.0
    symbol = ''

    def __init__(self, start_time, end_time, symbol):
        self.start_time = start_time
        self.end_time = end_time
        self.symbol = symbol

    def __repr__(self):
        return str(self.start_time) + '\t' + str(self.end_time) + '\t' + self.symbol


def to_mirex_lab(start_time, end_time, beat_segments, symbols, strengths):
    if len(beat_segments.start_times) < len(symbols) or len(symbols) != len(strengths):
        raise ValueError("inappropriate lists lengths")
    res = []
    if start_time < beat_segments.start_times[0]:
        res.append(ChordSegment(start_time, beat_segments.start_times[0], 'N'))
    for i in range(len(symbols)):
        sym = symbols[i] if strengths[i] > 0 else 'N'
        res.append(ChordSegment(
            beat_segments.start_times[i],
            beat_segments.start_times[i] + beat_segments.durations[i],
            sym))
    if res[-1].end_time < end_time:
        res.append(ChordSegment(res[-1].end_time, end_time, 'N'))
    return merge_segments(res)


def to_beat_chord_segment_list(start_time, end_time, beats, symbols):
    if len(beats) < len(symbols):
        raise ValueError("inappropriate lists lengths")
    if len(beats) == len(symbols):
        beats += [end_time]
    res = []
    if start_time < beats[0]:
        res.append(ChordSegment(start_time, beats[0], 'N'))
    for i in range(len(symbols)):
        sym = symbols[i]
        if beats[i] < beats[i+1]:
            res.append(ChordSegment(beats[i], beats[i + 1], sym))
        else:
            print("wrong beats order: " + str(beats[i]) + ", " + str(beats[i + 1]))
    if res[-1].end_time < end_time:
        res.append(ChordSegment(res[-1].end_time, end_time, 'N'))
    return res


def json_to_lab(choice, infile, outfile):
    with open(infile, 'r') as data_file:
        data = json.load(data_file)
        duration = float(data['duration'])
        metre_numerator = int(data['metre'].split('/')[0])
        all_beats = []
        all_chords = []
        process_parts(metre_numerator, data, all_beats, all_chords, choice)
        segments = merge_segments(to_beat_chord_segment_list(0, duration, all_beats, all_chords))
        with open(outfile, 'w') as content_file:
            for s in segments:
                content_file.write(str(s) + '\n')
