import pretty_midi
import os
import errno
import argparse
from pypianoroll import Multitrack
from parameters import CONFIG
import json

def make_sure_path_exists(path):
    """Create intermidate directories if the path does not exist."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def change_prefix(path, src, dst):
    """Return the path with its prefix changed from `src` to `dst`."""
    return os.path.join(dst, os.path.relpath(path, src))

def findall_endswith(postfix, root):
    """Traverse `root` recursively and yield all files ending with `postfix`."""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(postfix):
                yield os.path.join(dirpath, filename)

def get_midi_info(pm):
    """Return useful information from a MIDI object."""
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time
    else:
        first_beat_time = pm.estimate_beat_start()

    tc_times, tempi = pm.get_tempo_changes()

    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = None

    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'constant_time_signature': time_sign,
        'constant_tempo': tempi[0] if len(tc_times) == 1 else None
    }

    return midi_info

def converter(filepath, src, dst):
    """Convert a MIDI file to a multi-track piano-roll and save the
    resulting multi-track piano-roll to the destination directory. Return a
    tuple of `midi_md5` and useful information extracted from the MIDI file.
    """
    try:
        midi_md5 = os.path.splitext(os.path.basename(filepath))[0]
        multitrack = Multitrack(beat_resolution=CONFIG['beat_resolution'],
                                name=midi_md5)

        pm = pretty_midi.PrettyMIDI(filepath)
        multitrack.parse_pretty_midi(pm)
        midi_info = get_midi_info(pm)

        result_dir = change_prefix(os.path.dirname(filepath), src, dst)
        make_sure_path_exists(result_dir)
        multitrack.save(os.path.join(result_dir, midi_md5 + '.npz'))

        return (midi_md5, midi_info)

    except:
        return None

def main(src, dst, midi_info_path):
    """Main function."""
    make_sure_path_exists(dst)
    midi_info = {}
    for midi_path in findall_endswith('.mid', src):
        kv_pair = converter(midi_path, src, dst)
        if kv_pair is not None:
            midi_info[kv_pair[0]] = kv_pair[1]

    if midi_info_path is not None:
        with open(midi_info_path, 'w') as f:
            json.dump(midi_info, f)

    print("{} files have been successfully converted".format(len(midi_info)))

if __name__ == "__main__":
    """ test """
    src = "../data/"
    des = "../data/"
    info_path = "../data/midi_info.txt"
    main(src, des, info_path)