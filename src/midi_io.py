from pypianoroll import Multitrack, Track
import os
from parameters import *
import numpy as np
from collections import defaultdict
import json
import errno
from pretty_midi import PrettyMIDI
import warnings
def findall_endswith(postfix, root):
    """Traverse `root` recursively and yield all files ending with `postfix`."""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(postfix):
                yield os.path.join(dirpath, filename)

def make_sure_path_exists(path):
    """Create intermidate directories if the path does not exist."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

"""
Use a new package (pypianoroll to process raw_data)

Input one file path, output a pianoroll representation of the music.
If merge == True, then return (n_time_stamp, 128) other wish return (n_time_stamp, 128, num_track)
If velocity == True then return the velocity in the value divided by the maximum velocity value. (127)
(other wise return binary (0/1))
"""
def midi2Pianoroll(filepath,
                   merge=True,
                   velocity=False,
                   ):
    """Convert a MIDI file to a multi-track piano-roll and save the
    resulting multi-track piano-roll to the destination directory. Return a
    tuple of `midi_md5` and useful information extracted from the MIDI file.
    """
    midi_md5 = os.path.splitext(os.path.basename(filepath))[0]
    # prettyMIDI = PrettyMIDI(filepath)
    # time_signiture = ""
    # if len(prettyMIDI.time_signature_changes)>0:
    #     time_signiture = "{}/{}".format(prettyMIDI.time_signature_changes[0].numerator,
    #                                     prettyMIDI.time_signature_changes[0].denominator)
    # if time_signiture == "":
    #     raise ValueError(midi_md5 + " no timesigniture")

    # if time_signiture != CONFIG['time_signatures']:
    #     warnings.warn(midi_md5 + "'s timesiniture is " + time_signiture + " != " + CONFIG['time_signatures'])
    #     return None

    multitrack = Multitrack(filepath, beat_resolution=CONFIG['beat_resolution'], name=midi_md5)
    if merge:
        result = multitrack.get_merged_pianoroll(mode="max")
    else:
        result = multitrack.get_stacked_pianoroll()
        right = None
        left = None
        for each_track in multitrack.tracks:
            if each_track.name.strip().lower() == 'piano right':
                right = each_track.pianoroll
            if each_track.name.strip().lower() == 'piano left':
                left = each_track.pianoroll
        if right is None or left is None:
            return None
        result = np.stack([right, left], axis=2)

    if not velocity:
        result = np.where(result > 0, 1, 0)
    else:
        result = result / CONFIG['velocity_high']
    return result

"""
pianoroll_data: List. One element corresponding to one piece of music
Do not support multi-track.
Pass value to `dictionary_dict` would let this function to generate one-hot output
"""
def createSeqNetInputs(pianoroll_data: list, x_seq_length: int, lagging: int,
                       dictionary_dict=None,#
                       ) -> list:
    x = []
    y = []
    for i, piano_roll in enumerate(pianoroll_data):
        size = piano_roll.shape[0]
        if dictionary_dict:
            piano_roll = pianoroll2Embedding(piano_roll, dictionary_dict)
            size = len(piano_roll)

        pos = 0
        x_tmp = []
        y_tmp = []
        while pos + x_seq_length + lagging < size:
            x_tmp.append(piano_roll[pos:pos + x_seq_length])
            y_tmp.append(piano_roll[pos + lagging: pos + x_seq_length + lagging])
            pos += x_seq_length

        x_tmp = np.stack(x_tmp, axis=1) #(T, B, D)
        y_tmp = np.stack(y_tmp, axis=1)
        x.append(x_tmp)
        y.append(y_tmp)

    print(len(x))
    print("x shape (Bar-size, N_Bars, (Dim))", x[0].shape)
    print("y shape", y[0].shape)
    return x, y

"""
Get all the chords for the training set.
"""
def get_dictionary_of_chord(root_path,
                            two_hand=True, # True if the chord is chord is played by two hand.
                            dir = "../output/chord_dictionary/",
                            force = False,
                            ):
    make_sure_path_exists(dir)
    if not force and not two_hand and os.path.exists(os.path.join(dir, right_hand_corpus_file_name)) and os.path.exists(os.path.join(dir, left_hand_corpus_file_name)):
        return
    if not force and two_hand and os.path.exists(os.path.join(dir, two_hand_corpus_file_name)):
        return

    def func(result):
        chord_set = set()
        chord_list = list()
        chord = np.unique(result, axis=0)
        index = np.argwhere(chord > 0)
        _chord_tmp_dict = defaultdict(list)
        for row in index:
            _chord_tmp_dict[row[0]].append(row[1])

        chord_tmp_set = {str(value) for _, value in _chord_tmp_dict.items()}
        for i in chord_tmp_set:
            if i not in chord_set:
                chord_list.append(i)
                chord_set.add(i)

        return chord_list

    if two_hand:
        dic = dict()
        dic['None'] = END_TOKEN
        dic['[]'] = SILENCE_TOEKN
        count = 2
        for midi_path in findall_endswith('.mid', root_path):
            result = midi2Pianoroll(midi_path, merge=True, velocity=False)
            lis = func(result)
            for i in lis:
                if i not in dic.keys():
                    dic[i] = count
                    count += 1
        print(f"In total, there are {count} chords")
        with open(os.path.join(dir, two_hand_corpus_file_name), "w") as f:
            f.write(json.dumps(dic))

    else:
        dic_left = dict()
        dic_right = dict()
        count_left, count_right = 2, 2
        dic_right['None'] = END_TOKEN
        dic_right['[]'] = SILENCE_TOEKN
        dic_left['None'] = END_TOKEN
        dic_left['[]'] = SILENCE_TOEKN
        for midi_path in findall_endswith('.mid', root_path):
            result = midi2Pianoroll(midi_path, merge=False, velocity=False)
            if result is not None:
                left = result[:, :, 1]
                right = result[:, :, 0]
                lis_left = func(left)
                lis_right = func(right)
                for i in lis_left:
                    if i not in dic_left.keys():
                        dic_left[i] = count_left
                        count_left += 1

                for i in lis_right:
                    if i not in dic_right.keys():
                        dic_right[i] = count_right
                        count_right += 1

        print(f"In total, there are {count_left}/{count_right} left/right-hand chords")
        with open(os.path.join(dir, left_hand_corpus_file_name), "w") as f:
            f.write(json.dumps(dic_left))

        with open(os.path.join(dir, right_hand_corpus_file_name), "w") as f:
            f.write(json.dumps(dic_right))

"""
Convert pianoroll to one-hot version based on dictionary
"""
def pianoroll2Embedding(pianoroll: np.array, dictionary_dict: dict) -> list:
    pianoroll_to_str = []
    i = 0
    for row in pianoroll:
        id = np.argwhere(row).flatten()
        value = id.tolist()
        pianoroll_to_str.append(str(value))
        i += 1

    pianoroll_to_str.append("None")
    id_ = [int(dictionary_dict[v]) for v in pianoroll_to_str]
    return id_


"""
Write the pianoroll output to midi file. 
If 
Note: Do not support multi-track yet.
Input Shape: (n_time_stamp, 128)
"""
def pianoroll2Midi(piano_roll: np.array,
                   name="test_midi",
                   dir="../output/",
                   velocity=False,  # True if the input array contains velocity info (means not binary but continuous)
                   dictionary_dict = None  # True if the using the dic mode
                   ):
    if dictionary_dict is None:
        if velocity:
            piano_roll = np.floor(piano_roll * CONFIG['velocity_high'])
            piano_roll = np.clip(piano_roll, a_min=CONFIG['velocity_low'], a_max=CONFIG['velocity_high'])
        else:
            # fill in the default velocity
            piano_roll = np.where(piano_roll == 1, CONFIG['velocity'], piano_roll)
    else:
        # making sure this they are all one-hot vecotr (time_length, 128)
        # if not np.all(np.sum(piano_roll, axis=1) == 1):
        #     raise ValueError("This is not one-hot vector")
        piano_roll = embedding2pianoroll(piano_roll, dictionary_dict)

    make_sure_path_exists(dir)
    track = Track(piano_roll, is_drum=False, name="piano")
    multi_track = Multitrack(tracks=[track],
                             tempo=CONFIG['tempo'],
                             downbeat=None,
                             beat_resolution=CONFIG['beat_resolution'],
                             )
    file_name = os.path.join(dir, name if name.endswith(".mid") else name+".mid")
    multi_track.write(file_name)


def embedding2pianoroll(chord_list, dictionary):
    reverse_dic = {value: key for key, value in dictionary.items()}
    piano_tmp = np.zeros((len(chord_list), 128))
    # id_array = np.argwhere(piano_roll == 1)[:, 1]
    for i in range(len(chord_list)):
        notes_list = eval(reverse_dic[chord_list[i]])
        if notes_list is None:
            break
        elif len(notes_list) == 0:
            continue
        else:
            piano_tmp[i, notes_list] = CONFIG['velocity']

    piano_roll = piano_tmp
    return piano_roll

def load_corpus(path):
    with open(path, "r") as f:
        dic = json.load(f)
    return dic, len(dic)


def save_npy(x, path):
    np.save(path, x)

def load_npy(x, path):
    return np.load(path)
