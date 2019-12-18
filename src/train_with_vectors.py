# from seq2seq_model import DecoderRNN, EncoderRNN
from seq2seq_model import EncoderRNN, AttnDecoderRNN
from midi_io_dic_mode import *
from train_left_right import combine_left_and_right
from parameters import *
import torch
import numpy as np
from torch import optim
from torch import nn
import argparse

device = torch.device("cpu")
token_size = 0

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    target_length = target_tensor.size(0)
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_input = torch.zeros((1, target_tensor.size(1)), dtype=torch.long, device=device)
    decoder_hidden = encoder_hidden
    loss = 0
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        loss_ = criterion(decoder_output[0], target_tensor[di])
        if loss_ >= 0.3:
            loss += loss_

        prediction = torch.argmax(decoder_output[0], dim=1)
        if torch.rand(1)[0] > threshold:
            decoder_input = target_tensor[di].unsqueeze(0)
        else:
            decoder_input = prediction.unsqueeze(0).detach()  # detach from history as input
    if loss == 0:
        return 0

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def trainIters(train_x, train_y, encoder, decoder, learning_rate=1e-3, batch_size=32):
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for iter in range(1, len(train_x)+1): # iterate each sone
        input_tensor = train_x[iter-1]
        target_tensor = train_y[iter-1]
        input_tensor = torch.tensor(input_tensor, dtype=torch.long)
        target_tensor = torch.tensor(target_tensor, dtype=torch.long)
        loss = 0
        c = 1
        for i in range(0, input_tensor.size(1), batch_size):
            loss += train(input_tensor[:, i: i+batch_size], target_tensor[:, i: i+batch_size],
                          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            c += 1

        print_loss_total += loss

    return print_loss_total / c

def train_left(x, y, dictionary_size):
    mat = np.zeros((dictionary_size, dictionary_size))
    for i in range(len(x)):
        mat[x[i], y[i]] += 1

    return mat

def get_left(mat, x):
    y = []
    for i in x:
        a = np.argmax(mat[i])
        y.append(a)
    return y

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
            output, generate_seq = generate(input_datax, encoder1, decoder1, target_length, random=True, random_interval=12)
            generate_seq = torch.squeeze(generate_seq).numpy()
            pred_left = get_left(model, generate_seq)
            chord = combine_left_and_right(pred_left, generate_seq)
            pianorollToMidi(chord, name=f"{mid_name}_gen_by_{model_name}-{i}", velocity=False, dir=dir_name)
            generate_seq = torch.squeeze(output).numpy()
            pred_left = get_left(model, generate_seq)
            chord = combine_left_and_right(pred_left, generate_seq)
            pianorollToMidi(chord, name=f"{mid_name}_{model_name}-{i}", velocity=False, dir=dir_name)


def train_mul(args):
    model_name = "dict_atten_left_right_chord2vec"
    root = "../data/naive"
    origin_num_bars = 10
    target_num_bars = 20
    target_length = STAMPS_PER_BAR * target_num_bars
    origin_length = origin_num_bars * STAMPS_PER_BAR

    left_tracks, right_tracks = [], []
    # get_dictionary_of_chord(root, two_hand=False)
    for midi_path in findall_endswith('.mid', root):
        piano_roll_data = midiToPianoroll(midi_path, merge=False, velocity=False, )
        right_track, left_track = piano_roll_data[:, :, 0], piano_roll_data[:, :, 1]
        left_tracks.append(left_track)
        right_tracks.append(right_track)

    right_dictionary, token_size = load_corpus("../output/chord_dictionary/right-hand.json")
    with open("../output/chord_dictionary/chord2vec.npy", "rb") as f:
        weights_matrix = np.load(f)
        weights_matrix = torch.from_numpy(weights_matrix)
    epoch = args.epoch_number
    encoder1 = EncoderRNN(token_size, 300, hidden_dim).to(device)
    # encoder1.embedding.load_state_dict({'weight': weights_matrix})
    # if non_trainable:
    #     encoder1.embedding.weight.requires_grad = False

    attn_decoder1 = AttnDecoderRNN(token_size, 300, hidden_dim, encoder1.embedding, dropout_p=0.1,
                                   max_length=time_len).to(device)

    if args.load_epoch != 0:
        encoder1.load_state_dict(torch.load(f'../models/E_{model_name}_' + str(args.load_epoch)))
        attn_decoder1.load_state_dict(torch.load(f'../models/D_{model_name}_' + str(args.load_epoch)))

    input_datax, input_datay = createSeqNetInputs(right_tracks, time_len, output_len, right_dictionary)

    for i in range(1, epoch + 1):
        loss = trainIters(input_datax, input_datay, encoder1, attn_decoder1, learning_rate=args.lr)
        print(f'{i + args.load_epoch} loss {loss}')
        if i % 50 == 0:
            torch.save(encoder1.state_dict(), f'../models/E_{model_name}_' + str(i + args.load_epoch))
            torch.save(attn_decoder1.state_dict(), f'../models/D_{model_name}_' + str(i + args.load_epoch))

    x = np.concatenate(right_tracks)
    y = np.concatenate(left_tracks)
    model = train_left(x, y)
    predict("../data/test", origin_length, encoder1, attn_decoder1, target_length, model_name, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a MIDI_NET')
    parser.add_argument('-e', '--epoch_number', type=int, help='the epoch number you want to train')
    parser.add_argument('-l', '--load_epoch', type=int, help='the model epoch need to be loaded', default=0)
    parser.add_argument('-lr', '--lr', type=float, help='learning_rate', default=0.001)
    args = parser.parse_args()
    train_mul(args)

