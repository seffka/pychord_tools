import json
import re
import numpy as np
import math

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


def process_chords(metre, blocks, all_bars, all_chords, all_events, all_beats):
    numerator, denominator = [int(x) for x in metre.split('/')]
    beats_count = 0
    for block in blocks:
        bars = block.split('|')
        incompleteBars = np.repeat(False, len(bars))
        if len(bars) > 0 and not bars[0].strip():
            bars = bars[1:]
            incompleteBars = incompleteBars[1:]
        else:
            incompleteBars[0] = True

        if len(bars) > 0 and not bars[-1].strip():
            bars = bars[:-1]
            incompleteBars = incompleteBars[:-1]
        else:
            incompleteBars[-1] = True
        for (bar, isIncomplete) in zip(bars, incompleteBars):
            chords = [c for c in re.split('\s+', bar) if c != '']
            if isIncomplete:
                beats_in_bar = 0
                # TODO: remove code duplication
                for c in chords:
                    components = c.split(':')
                    if len(components) > 2:
                        beats_in_bar += denominator / int(components[2])
                    else:
                        # default duration is one denominator-th.
                        beats_in_bar += 1.0
                if beats_in_bar < 0.01:
                    raise ValueError("Empty incomplete bar")
                elif beats_in_bar >= numerator:
                    raise ValueError("Too many beats in incomplete bar: %d" % (beats_in_bar))
                next_beats_count = beats_count + math.floor(beats_in_bar + 0.01)
            else:
                next_beats_count = beats_count + numerator
            beats = all_beats[beats_count:next_beats_count]
            if all_bars is not None:
                if len(all_bars) > 0:
                    all_bars[-1][1] = beats[0]
                all_bars.append([beats[0], beats[0]])
                if len(beats) > 1:
                    all_bars[-1][1] = beats[-1] + (beats[-1] - beats[-2])

            extended_beats = np.array(beats)
            if next_beats_count < len(all_beats):
                extended_beats = np.append(extended_beats, all_beats[next_beats_count])
            elif next_beats_count == len(all_beats):
                # extrapolate last beat's duration
                extended_beats = np.append(extended_beats, extended_beats[-1] + (extended_beats[1:]-extended_beats[:-1]).mean())
            else:
                raise ValueError("beats array is too short: %d. At least %d is expected" %(len(all_beats),  next_beats_count))
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
                # TODO: process X-chords
                # newchords = np.array(newchords)
                # ignore X-chords (eXtensions)
                # originals = newchords != 'X:'
                # all_chords.extend(newchords[originals])
                # all_events.extend(np.array(beats)[originals])
            else:
                # events mode
                bar_events = []
                pure_chords = []
                # position. in denominator-th durations.
                pos = 0
                for c in chords:
                    components = c.split(':')
                    pure_chords.append(':'.join(components[0:2]))
                    bar_events.append(pos)
                    if len(components) > 2:
                        pos += denominator / int(components[2])
                    else:
                        # default duration is one denominator-th.
                        pos += 1.0
                if not isIncomplete and abs(pos - numerator) > 0.01:
                    raise ValueError("Inapropriate bar |" + bar + "| length: " + str(pos) +
                                     ", expected: " +  str(numerator))
                bar_events = np.array(bar_events, dtype=float)
                conditions = []
                funcs = []
                for i in range(round(pos)):
                    conditions.append((bar_events >= i) & (bar_events < (i+1)))
                    funcs.append(lambda x, i=i: extended_beats[i] + (extended_beats[i + 1] - extended_beats[i]) * (x - float(i)))
                pure_chords = np.array(pure_chords)
                # ignore X-chords (eXtensions)
                originals = pure_chords != 'X:'
                all_chords.extend(pure_chords[originals])
                all_events.extend(np.piecewise(bar_events, conditions, funcs)[originals])
            beats_count = next_beats_count


def process_parts(metre, data, events, chords, choice, beatz = None, bars = None):
    """
    Process single stream content.

    :param metre: metre string (e.g., "4/4")
    :param data: root data element (deserialized from JSON file)
    :param events: list of event times. To be filled by the function.
    :param chords: list of chord names associated with the events. To be filled by the function.
    :param choice: name of the entity with chroma events  (e.g., "chords", "modes").
    :param beatz: list of the beats. To be filled by the function.
    :param bars: list of the bars ([start, end] arrays). To be filled by the function.
    :return:
    """
    if 'parts' in data.keys():
        for part in data['parts']:
            process_parts(metre, part, events, chords, choice, beatz, bars)
    else:
        if 'metre' in data:
            metre = data['metre']
        if beatz is not None:
            beatz.extend(data['beats'])
        process_chords(metre, data[choice], bars, chords, events, data['beats'])


def process_single_part_stream(stream_name, metre, data, events, chords, choice, beatz, bars = None):
    if stream_name not in chords:
        chords[stream_name] = []
    if stream_name not in events:
        events[stream_name] = []
    process_chords(metre, data[choice], bars, chords[stream_name], events[stream_name], beatz)

SINGLETON_STREAM_NAME = "singleton"

def process_parts_multistream(metre, data, events, chords, choice, beatz = None, bars = None):
    """
    Process multistream (polyphonic) content.
    :param metre: metre string (e.g., "4/4")
    :param data: root data element (deserialized from JSON file)
    :param events: dict <stream name> -> <list of timestamps> of events. To be filled by the function.
    :param chords: dict <stream name> -> <list of labels of the <choice>>
           of chord kinds associated with the events. To be filled by the function.
    :param choice: name of the entity with chroma events  (e.g., "chords", "modes").
    :param beatz: list of the beats. To be filled by the function.
    :param bars: list of the bars ([start, end] arrays). To be filled by the function.
    :return:
    """
    if 'parts' in data.keys():
        for part in data['parts']:
            process_parts_multistream(metre, part, events, chords, choice, beatz, bars)
    else:
        if 'metre' in data:
            metre = data['metre']
        if beatz is not None:
            beatz.extend(data['beats'])
        bars_to_pass = bars
        if "streams" in data:
            for stream in data["streams"].keys():
                process_single_part_stream(stream, metre, data["streams"][stream], events, chords, choice, data['beats'], bars_to_pass)
                bars_to_pass = None
        else:
            process_single_part_stream(metre, data, events, chords, choice, data['beats'], bars)


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
        all_beats = []
        all_chords = []
        process_parts(data['metre'], data, all_beats, all_chords, choice)
        segments = merge_segments(to_beat_chord_segment_list(0, duration, all_beats, all_chords))
        with open(outfile, 'w') as content_file:
            for s in segments:
                content_file.write(str(s) + '\n')
