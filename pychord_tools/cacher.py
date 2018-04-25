import joblib
import os

######################################################################
# Cache management
######################################################################

def getCaheDir():
    if 'PYCHORD_TOOLS_CACHE_DIR' in os.environ:
        return os.environ['PYCHORD_TOOLS_CACHE_DIR']
    else:
        return None

memory = joblib.Memory(cachedir=getCaheDir(), verbose=0)

def clearCache():
    memory.clear()
