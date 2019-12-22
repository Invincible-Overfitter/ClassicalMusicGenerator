from midi_io import *
import os, argparse
from parameters import *
import numpy as np

def get_processed_data(path, output_name):
    x = np.load(os.path.join(path, output_name+"_x.npy"))
    y = np.load(os.path.join(path, output_name+"_y.npy"))
    return x, y

def preprocess(args: argparse.ArgumentParser):
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_name = args.name
    save_path_x, save_path_y = os.path.join(output_dir, output_name+"_x"), os.path.join(output_dir, output_name+"_y")
    make_sure_path_exists(output_dir)
    if not os.path.exists(input_dir):
        raise ValueError(f"Path {input_dir} not existed")

    # get the dictionary first
    corpus_path = "../output/chord_dictionary/"
    get_dictionary_of_chord(input_dir, two_hand=False, dir=corpus_path, force=False)
    right_corpus, corpus_size = load_corpus(os.path.join(corpus_path, right_hand_corpus_file_name))

    # load the music files
    right_tracks = []
    left_tracks = []
    for midi_path in findall_endswith(".mid", input_dir):
        pianoroll_data = midi2Pianoroll(midi_path, merge=False, velocity=False)  # (T, 128, 2)
        if pianoroll_data is not None:
            right_track, left_track = pianoroll_data[:, :, 0], pianoroll_data[:, :, 1]
            right_tracks.append(right_track)
            left_tracks.append(left_track)

    # preprocess the right-hand pianoroll
    input_datax, input_datay = createSeqNetInputs(right_tracks, time_len, time_len, dictionary_dict=right_corpus)
    print(f"shape x: {input_datax[0].shape}")
    tot_x = np.hstack(input_datax)  # (T, B)
    tot_y = np.hstack(input_datay)  # (T, B)
    save_npy(tot_x, save_path_x)
    save_npy(tot_y, save_path_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='input directory', default="../raw_data/classical/")
    parser.add_argument('-o', '--output_dir', type=str, help='output directory', default="../processed_data/classical/")
    parser.add_argument('-name', '--name', type=str, help='the name of the generated file', default="total_data")
    args = parser.parse_args()
    # preprocess(args)
    x, y = get_processed_data("../processed_data/classical/", "total_data")
    print(f"shape x = {x.shape}, shape y = {y.shape}")