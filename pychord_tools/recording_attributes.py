import musicbrainzngs as mb
from . import cacher


class RecordingAttributes:
    def __init__(self, title, artist, year, started, finished, lineup, composers):
        self.title = title
        self.artist = artist
        self.year = year
        self.started = started
        self.finished = finished
        self.lineup = lineup
        self.composers = composers

    def __str__(self):
        return ' '.join([
            self.title,
            self.artist,
            str(self.year),
            str(self.started),
            str(self.finished),
            str(self.lineup),
            str(self.composers)]).encode('utf-8')
    __repr__ = __str__


def extract_composers_otherwise_writers(relation_list):
    """
    Returns list of work's composers names (or writers, if there's no one composer).
    :param relation_list: musicBrainz work relation list
    :return: array of strings (composers names)
    """
    composers = []
    writers = []
    for rel in relation_list:
        if rel['type'] == 'composer':
            composers.append(rel['artist']['name'])
        elif rel['type'] == 'writer':
            writers.append(rel['artist']['name'])
    if len(composers) > 0:
        return composers
    else:
        return writers


def get_composers(work_relatios):
    for workRel in work_relatios:
        if workRel['type'] == 'performance':
            work_id = workRel['work']['id']
            work_attrs = mb.get_work_by_id(
                id=work_id,
                includes=['artist-rels'])
            if 'artist-relation-list' in work_attrs['work'].keys():
                return extract_composers_otherwise_writers(
                    work_attrs['work']['artist-relation-list'])
    return []


def get_lineup(artist_relations):
    result = {}
    for rel in artist_relations:
        if rel['type'] == 'instrument':
            key = ' and '.join(rel['attribute-list'])
            result.setdefault(key, []).append(rel['artist']['name'])
        elif rel['type'] == 'vocal':
            result.setdefault('vocal', []).append(rel['artist']['name'])
    return result


def estimate_recording_time(artist_relations):
    begin_date = None
    end_date = None
    for rel in artist_relations:
        if 'begin' in rel.keys() and (begin_date is None or begin_date > rel['begin']):
            begin_date = rel['begin']
        if 'end' in rel.keys() and (end_date is None or end_date < rel['end']):
            end_date = rel['end']
    if begin_date is not None:
        year = begin_date[:4]
    else:
        year = None
    return begin_date, end_date, year


@cacher.memory.cache
def load_recording_attributes_from_music_brainz(mbid):
    """
    Loads attributes for the single recording.
    :param mbid: MBID
    :return: RecordingAttributes
    """
    mb.set_useragent("application", "0.01", "http://example.com")
    rec_attrs = mb.get_recording_by_id(
        id=mbid,
        includes=['artist-credits', 'artist-rels', 'work-rels'])
    lineup = {}
    composers = []
    recording_started = None
    recording_finished = None
    recording_year = None
    if 'artist-relation-list' in rec_attrs['recording'].keys():
        artist_relations = rec_attrs['recording']['artist-relation-list']
        recording_started, recording_finished, recording_year = estimate_recording_time(artist_relations)
        lineup = get_lineup(artist_relations)
    if 'work-relation-list' in rec_attrs['recording'].keys():
        composers = get_composers(rec_attrs['recording']['work-relation-list'])
    return RecordingAttributes(
        rec_attrs['recording']['title'],
        rec_attrs['recording']['artist-credit-phrase'],
        recording_year,
        recording_started,
        recording_finished,
        lineup,
        composers)


def load_dict_from_music_brainz(mbids):
    """
    Loads recording attributes from musicbrainz DB
    for multiple mbids.
    :param mbids: array of musicbrainz ids.
    :return: dictionary mbid -> RecordingAttributes
    """
    result = {}
    for mbid in mbids:
        result[mbid] = load_recording_attributes_from_music_brainz(mbid)
    return result
