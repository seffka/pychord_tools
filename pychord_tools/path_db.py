import json
import os

dbPathMap = None

PYCHORD_TOOLS_PATH_DB_FILES = 'PYCHORD_TOOLS_PATH_DB_FILES'


def get_path_db_files():
    if PYCHORD_TOOLS_PATH_DB_FILES in os.environ:
        return os.environ[PYCHORD_TOOLS_PATH_DB_FILES]
    else:
        return None


def dict_to_tuple(feature, d=None):
    if d is None:
        return feature,
    else:
        return (feature,) + tuple(((k, e) for k, e in d.items()))


def process_element(dir_base, el):
    global dbPathMap
    if type(el) == dict:
        uid = el.pop('uid', None)
        content = {}
        for k in el.keys():
            feature = k
            if type(el[k]) == dict:
                path = el[k].pop('path', None)
                if not os.path.isabs(path):
                    path = os.path.join(dir_base, path)
                feature_key = dict_to_tuple(feature, el[k])
                content[feature_key] = path
        if uid in dbPathMap:
            dbPathMap[uid].update(content)
        else:
            dbPathMap[uid] = content


def add_file_content_to_map(path):
    with open(path) as f:
        data = json.load(f)
        dir_base = os.path.dirname(os.path.realpath(f.name))
        if type(data) == list:
            for el in data:
                process_element(dir_base, el)
        else:
            process_element(dir_base, data)


def init():
    global dbPathMap
    if dbPathMap is None:
        dbPathMap = {}
        files = get_path_db_files()
        if files is not None:
            for f in files.split(os.pathsep):
                add_file_content_to_map(f)


def get_path(uid, key):
    init()
    if uid in dbPathMap:
        d = dbPathMap.get(uid)
        if key in d:
            return d[key]
    return None


def get_audio_path(uid):
    return get_path(uid, dict_to_tuple('audio'))


def set_path(uid, key, path):
    init()
    if uid in dbPathMap:
        d = dbPathMap[uid]
    else:
        d = {}
        dbPathMap[uid] = d
    d[key] = path


def get_features_path(uid, feature, param_dict):
    return get_path(uid, dict_to_tuple(feature, param_dict))


def set_audio_path(uid, path):
    set_path(uid, dict_to_tuple('audio'), path)


def set_feature_path(uid, feature, param_dict, path):
    set_path(uid, dict_to_tuple(feature, param_dict), path)


def dump_path_db(file, relative_paths=False, filter = None):
    res = []
    base_dir = os.path.dirname(os.path.realpath(file.name))
    for (uid, uidMap) in dbPathMap.items():
        uid_data = {'uid': uid}
        for (k, v) in uidMap.items():
            region, *rest = k
            if filter is None or filter == region:
                d = dict(rest)
                if relative_paths:
                    d['path'] = os.path.relpath(v, base_dir)
                else:
                    d['path'] = os.path.abspath(v)
                uid_data[region] = d
        if len(uid_data) > 1:
            res.append(uid_data)
    json.dump(res, file, indent=4)
