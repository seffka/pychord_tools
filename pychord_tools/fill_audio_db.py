#
# Scans directory tree for audio files and prepares DB with MusicBrainzID to audio file pathname relationships.
#
import os
import re
import sys
import argparse
import acoustid
import musicbrainzngs as m

from .path_db import set_audio_path, dump_path_db

m.set_useragent("application", "0.01", "http://example.com")

API_KEY = 'cSpUJKpD'


def get_args():
    # Description for documentation
    parser = argparse.ArgumentParser(
        description='Scans directory tree for audio files and prepares DB with MusicBrainzID to audio file pathname'
                    ' relationships')

    parser.add_argument(
        '-p', '--path', type=str, help='Path to Audio files', required=True)
    parser.add_argument(
        '-o', type=argparse.FileType('w'), default='db.json', help='output file (default is: db.json)')
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    inputdir = args.path
    # Return all variable values
    return inputdir, args.o.name


def collect_recording_ids(d, result):
    if type(d) is dict:
        for k in d.keys():
            if k == 'track-list':
                for r in d['track-list']:
                    result.add(r['recording']['id'])
            elif type(d[k]) is list:
                for e in d[k]:
                    collect_recording_ids(e, result)
            else:
                collect_recording_ids(d[k], result)


def aidmatch(filename):
    try:
        results = acoustid.match(API_KEY, filename)
    except acoustid.NoBackendError:
        print("chromaprint library/tool not found")
        sys.exit(1)
    except acoustid.FingerprintGenerationError:
        print("fingerprint could not be calculated")
        sys.exit(1)
    except acoustid.WebServiceError as exc:
        print("web service request failed:", exc.message)
        sys.exit(1)
    return results


######################################################################

inputDir, outfile = get_args()

res = []
for path, dname, fnames in os.walk(inputDir):
    for fname in fnames:
        if re.search('(\.wav$)|(\.mp3$)|(\.flac$)', fname):
            pathname = '/'.join((path, fname))
            print(pathname)
            a = aidmatch(pathname)
            for score, rid, title, artist in a:
                print(rid, title, artist)
                set_audio_path(rid, pathname)
with open(outfile, 'w') as fo:
    dump_path_db(fo)
