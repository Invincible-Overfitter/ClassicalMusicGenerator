import numpy as np
import torch
from generate import *
from midi_io_dic_mode import *
from midi_io_musegan import findall_endswith, make_sure_path_exists
from parameters import *
import os
device = torch.device("cpu")
from seq2seq_model import DecoderRNN, EncoderRNN, AttnDecoderRNN
from midi_io_dic_mode import *
from parameters import *
import torch
from torch import optim
from torch import nn
import argparse

device = torch.device("cpu")


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(0)
    encoder_output, encoder_hidden = encoder(input_tensor)  # (time_len, batch_size, D)
    decoder_input = torch.zeros((1, target_tensor.size(1), target_tensor.size(2)), dtype=torch.float, device=device)
    decoder_hidden = encoder_hidden
    ones = torch.ones(input_tensor.size(1), input_tensor.size(2))
    zeros = torch.zeros(input_tensor.size(1), input_tensor.size(2))
    loss = 0
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) # decoder_output：(1, B, D)
        loss += criterion(decoder_output[0], target_tensor[di])
        if torch.rand(1)[0] > threshold:
            decoder_input = target_tensor[di].unsqueeze(0)

        else:
            decoder_input = torch.where(decoder_output[0, :, :] > 0.5, ones, zeros)
            decoder_input = decoder_input.unsqueeze(0).detach()  # detach from history as input

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def trainIters(train_x, train_y, encoder, decoder, learning_rate=1e-3, batch_size=32):
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.BCELoss(reduction="sum")
    for iter in range(1, len(train_x)+1): # iterate each sone
        input_tensor = train_x[iter-1]
        target_tensor = train_y[iter-1]
        input_tensor = torch.tensor(input_tensor, dtype=torch.float)
        target_tensor = torch.tensor(target_tensor, dtype=torch.float)
        loss = 0
        for i in range(0, input_tensor.size(1), batch_size):
            loss += train(input_tensor[:, i: i+batch_size], target_tensor[:, i: i+batch_size],
                          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss

    return print_loss_total


def train_left(x=None, y=None, path=None):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam
    from keras.models import load_model
    import os

    if path is None or not os.path.exists(path):
        path = path if path is not None else "../models/keras_mlp.h5"
        model = Sequential()
        model.add(Dense(256))
        model.add(Dense(128, activation="sigmoid"))
        adam = Adam(learning_rate=1e-3)
        model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['mae'])
        model.fit(x, y, epochs=25, batch_size=32, verbose=2)
        model.save(path)
    else:
        model = load_model(path)

    return model


def get_left(model, x):
    pred = model.predict(x)
    pred = np.where(pred>0.5, 1, 0)
    return pred


def combine_left_and_right(left, right):
    assert len(left) == len(right)
    final_chord = (left + right)
    return final_chord


def predict(root, origin_length, encoder1, decoder1, target_length, model_name, model):
    import time
    dir_name = os.path.join("../output/", model_name + "_" + str(time.time()))
    make_sure_path_exists(dir_name)
    for midi_path in findall_endswith(".mid", root):
        mid_name = midi_path.split("/")[-1]
        pianoroll_data = midiToPianoroll(midi_path, merge=False, velocity=False)
        if pianoroll_data.shape[2] < 2:
            return
        right_track, left_track = pianoroll_data[:, :, 0], pianoroll_data[:, :, 1]
        for i in [500,]:
            if (i + origin_length) > len(right_track):
                break

            input_datax = torch.tensor(right_track[i:i + origin_length], dtype=torch.float).unsqueeze(1)
            output, generate_seq = generate(input_datax, encoder1, decoder1, target_length)
            generate_seq = torch.squeeze(generate_seq).numpy()
            pred_left = get_left(model, generate_seq)
            chord = combine_left_and_right(pred_left, generate_seq)
            pianorollToMidi(chord, name=f"{mid_name}_gen_by_{model_name}-{i}", velocity=False, dir=dir_name)
            generate_seq = torch.squeeze(output).numpy()
            pred_left = get_left(model, generate_seq)
            chord = combine_left_and_right(pred_left, generate_seq)
            pianorollToMidi(chord, name=f"{mid_name}_{model_name}-{i}", velocity=False, dir=dir_name)


def train_mul(args):
    model_name = "left_right_mul-beat8"
    origin_num_bars = 10
    target_num_bars = 20
    target_length = STAMPS_PER_BAR * target_num_bars
    origin_length = origin_num_bars * STAMPS_PER_BAR
    right_tracks = []
    left_tracks = []
    for midi_path in findall_endswith(".mid", "../data/naive"):
        pianoroll_data = midiToPianoroll(midi_path, merge=False, velocity=False)  # (n_time_stamp, 128, num_track)
        try:
            right_track, left_track = pianoroll_data[:, :, 0], pianoroll_data[:, :, 1]
            right_tracks.append(right_track)
            left_tracks.append(left_track)
        except:
            pass

    input_datax, input_datay = createSeqNetInputs(right_tracks, time_len, output_len)
    encoder1 = EncoderRNN(input_dim, hidden_dim).to(device)
    decoder1 = DecoderRNN(input_dim, hidden_dim).to(device)
    # decoder1 = AttnDecoderRNN(input_dim, hidden_dim, dropout_p=0.1, max_length=time_len).to(device)
    if args.load_epoch != 0:
        encoder1.load_state_dict(torch.load(f'../models/mul_encoder_{model_name}_' + str(args.load_epoch)))
        decoder1.load_state_dict(torch.load(f'../models/mul_decoder_{model_name}_' + str(args.load_epoch)))

    for i in range(1, args.epoch_number + 1):

        loss = trainIters(input_datax, input_datay, encoder1, decoder1)
        print(f'{i} loss {loss}')
        if i % 50 == 0:
            torch.save(encoder1.state_dict(), f'../models/mul_encoder_{model_name}_' + str(i + args.load_epoch))
            torch.save(decoder1.state_dict(), f'../models/mul_decoder_{model_name}_' + str(i + args.load_epoch))
    x = np.concatenate(right_tracks)
    y = np.concatenate(left_tracks)
    model = train_left(x, y, path="../models/keras_mul_beat8.h5")
    predict("../data/test", origin_length, encoder1, decoder1, target_length, model_name, model)


def train_one(args):
    midi_path = "../data/chp_op18.mid"
    model_name = "left_right"
    origin_num_bars = 8
    target_num_bars = 20
    learning_rate = args.learning_rate
    target_length = STAMPS_PER_BAR * target_num_bars
    origin_length = origin_num_bars * STAMPS_PER_BAR

    pianoroll_data = midiToPianoroll(midi_path, merge=False, velocity=False) # (n_time_stamp, 128, num_track)
    right_track, left_track = pianoroll_data[:, :, 0], pianoroll_data[:, :, 1]
    input_datax, input_datay = createSeqNetInputs([right_track], time_len , output_len)

    encoder1 = EncoderRNN(input_dim, hidden_dim).to(device)
    decoder1 = DecoderRNN(input_dim, hidden_dim).to(device)
    if args.load_epoch != 0:
        encoder1.load_state_dict(torch.load(f'../models/mul_encoder_{model_name}_' + str(args.load_epoch)))
        decoder1.load_state_dict(torch.load(f'../models/mul_decoder_{model_name}_' + str(args.load_epoch)))

    print("shape of data ", pianoroll_data.shape)
    for i in range(1, args.epoch_number+1):
        loss = trainIters(input_datax, input_datay, encoder1, decoder1, learning_rate=learning_rate)
        print(f'{i} loss {loss}')
        if i % 50 == 0:
            torch.save(encoder1.state_dict(), f'../models/mul_encoder_{model_name}_' + str(i + args.load_epoch))
            torch.save(decoder1.state_dict(), f'../models/mul_decoder_{model_name}_' + str(i + args.load_epoch))

    # generating
    for midi_path in findall_endswith(".mid", "../data/test"):
        predict(midi_path, origin_length, encoder1, decoder1, target_length, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a MIDI_NET')
    parser.add_argument('-e', '--epoch_number', type=int, help='the epoch number you want to train')
    parser.add_argument('-l', '--load_epoch', type=int, help='the model epoch need to be loaded', default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate', default=0.001)
    args = parser.parse_args()
    train_mul(args)







