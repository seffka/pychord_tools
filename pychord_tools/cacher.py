import joblib
import os

######################################################################
# Cache management
######################################################################


def get_cahe_dir():
    if 'PYCHORD_TOOLS_CACHE_DIR' in os.environ:
        return os.environ['PYCHORD_TOOLS_CACHE_DIR']
    else:
        return None

memory = joblib.Memory(cachedir=get_cahe_dir(), verbose=0)


def clear_cache():
    memory.clear()
