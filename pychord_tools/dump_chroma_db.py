#
# Scans directory tree for audio files and prepares DB with MusicBrainzID to audio file pathname relationships.
#
import os
import re
import argparse

from .low_level_features import DefaultUidAndAudioPathExtractor
from .path_db import set_feature_path, dump_path_db
from . import third_party

def get_args():
    # Description for documentation
    parser = argparse.ArgumentParser(
        description='Extract and dump chroma.')

    parser.add_argument('path', type=str, help='Path to annotation files')
    parser.add_argument('-o', type=str, default='features', help='Path to the ouput directory')
    args = parser.parse_args()
    return args.path, args.o

######################################################################

inputDir, outputDir = get_args()

res = []
extractor = DefaultUidAndAudioPathExtractor()
params = {'sample_rate': 44100, 'step_size': 2048}

for fname in os.listdir(inputDir):
    if re.search('\.json$', fname):
        pathname = '/'.join((inputDir, fname))
        uid, audio = extractor.uid_and_audio_path_name(pathname)
        if audio is None:
            print("Audio is missing:", pathname, uid)
            res.append(pathname)
            continue;
        print("Dumping chroma for", fname, uid)
        name, ext = os.path.splitext(fname)
        csv_gz_name = os.path.join(outputDir, name + '.csv.gz')
        third_party.dump_nnls_chroma_to_csv_zip(
            csv_gz_name,
            third_party.nnls_chroma_from_audio(
                uid, sample_rate=params['sample_rate'], step_size=params['step_size']),
            sample_rate = params['sample_rate'], step_size = params['step_size'])
        set_feature_path(uid, 'nnls_chroma', params, csv_gz_name)

with open(os.path.join(outputDir, 'db.json'), 'w') as fout:
    dump_path_db(fout, relative_paths=True, filter='nnls_chroma')
if len(res) == 0:
    print("Chroma is dumped for all annotations.")
else:
    print("Audio is missed for" , len(res), "annotation(s).")