#
# Scans directory tree for audio files and prepares DB with MusicBrainzID to audio file pathname relationships.
#
import os
import re
import argparse

from .low_level_features import DefaultUidAndAudioPathExtractor
from .path_db import get_path_db_files, PYCHORD_TOOLS_PATH_DB_FILES

def get_args():
    # Description for documentation
    parser = argparse.ArgumentParser(
        description='Checks, if all annotations in the given directory have associated audio or chroma files.')

    parser.add_argument('path', type=str, help='Path to annotation files')
    args = parser.parse_args()
    return args.path

######################################################################

inputDir = get_args()

db_path = get_path_db_files()
if db_path is None:
    print(PYCHORD_TOOLS_PATH_DB_FILES, "is not set. Exiting.")
    exit(-1)
else:
    print(PYCHORD_TOOLS_PATH_DB_FILES, ": ", db_path)

res = []
extractor = DefaultUidAndAudioPathExtractor()
for fname in os.listdir(inputDir):
    if re.search('\.json$', fname):
        pathname = '/'.join((inputDir, fname))
        uid, audio = extractor.uid_and_audio_path_name(pathname)
        if audio is None:
            print("Audio is missing:", pathname, uid)
            res.append(pathname)
if len(res) == 0:
    print("Audio is assigned to all annotations.")
else:
    print("Audio is missed for" , len(res), "annotation(s).")