from generate import *
device = torch.device("cpu")
from seq2seq_model import EncoderRNN, AttnDecoderRNN
from midi_io import *
from parameters import *
import torch
from torch import nn
from torch import optim
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
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output) # decoder_output：(1, B, D)
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


def train_mul(args):
    model_name = "best model"
    right_tracks = []
    left_tracks = []
    for midi_path in findall_endswith(".mid", "../raw_data/classical"):
        pianoroll_data = midi2Pianoroll(midi_path, merge=False, velocity=False)  # (n_time_stamp, 128, num_track)
        if pianoroll_data is not None:
            right_track, left_track = pianoroll_data[:, :, 0], pianoroll_data[:, :, 1]
            right_tracks.append(right_track)
            left_tracks.append(left_track)

    input_datax, input_datay = createSeqNetInputs(right_tracks, time_len, output_len)
    encoder1 = EncoderRNN(input_dim, hidden_dim).to(device)
    decoder1 = AttnDecoderRNN(input_dim, hidden_dim, dropout_p=0.1, max_length=time_len).to(device)
    if args.load_epoch != 0:
        encoder1.load_state_dict(torch.load(f'../models/mul_encoder_{model_name}_' + str(args.load_epoch)))
        decoder1.load_state_dict(torch.load(f'../models/mul_decoder_{model_name}_' + str(args.load_epoch)))

    for i in range(1, args.epoch_number + 1):
        loss = trainIters(input_datax, input_datay, encoder1, decoder1)
        print(f'{i} loss {loss}')
        if i % 50 == 0:
            torch.save(encoder1.state_dict(), f'../models/mul_encoder_{model_name}_' + str(i + args.load_epoch))
            torch.save(decoder1.state_dict(), f'../models/mul_decoder_{model_name}_' + str(i + args.load_epoch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a MIDI_NET')
    parser.add_argument('-e', '--epoch_number', type=int, help='the epoch number you want to train')
    parser.add_argument('-l', '--load_epoch', type=int, help='the model epoch need to be loaded', default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate', default=0.001)
    args = parser.parse_args()
    train_mul(args)







