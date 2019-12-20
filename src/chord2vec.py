from midi_io import *
from gensim.models import Word2Vec
from parameters import *
import numpy as np
model_name = "dict_atten_left_right"
root = "../data/naive"
origin_num_bars = 10
target_num_bars = 20
target_length = STAMPS_PER_BAR * target_num_bars
origin_length = origin_num_bars * STAMPS_PER_BAR

left_tracks, right_tracks = [], []
# get_dictionary_of_chord(root, two_hand=False)
for midi_path in findall_endswith('.mid', root):
    piano_roll_data = midi2Pianoroll(midi_path, merge=False, velocity=False, )
    right_track, left_track = piano_roll_data[:, :, 0], piano_roll_data[:, :, 1]
    left_tracks.append(left_track)
    right_tracks.append(right_track)

data = []
right_dictionary, token_size = load_corpus("../output/chord_dictionary/right-hand.json")
for song in right_tracks:
    dic_data = pianoroll2Embedding(song, right_dictionary)
    dic_data = [str(i) for i in dic_data]
    data.append(dic_data)

model = Word2Vec(data, min_count=1, size=300, max_vocab_size=token_size, workers=8, window = STAMPS_PER_BAR, sg=1, iter=100)
index = [str(i) for i in range(token_size)]
model.save("../models/chord_embedding.model")
vectors = model[index]
with open("../output/chord_dictionary/chord2vec.npy", "wb") as f:
    np.save(f, vectors)
