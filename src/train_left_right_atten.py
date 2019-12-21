from generate import *
device = torch.device("cpu")
from seq2seq_model import EncoderRNN, AttnDecoderRNN
from parameters import *
import torch
from torch import nn
from torch import optim
import argparse

device = torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(descrption='train a MIDI_NET')
    parser.add_argument('-e', '--epoch_number', type=int, help='the epoch number you want to train')
    parser.add_argument('-n', '--model_name', type=str, help='the model name')
    parser.add_argument('-l', '--load_epoch', type=int, help='the model epoch need to be loaded', default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate', default=0.001)
    args = parser.parse_args()

    right_tracks = []
    left_tracks = []

    # TODO: load the dataset
    train_x = []      # [T, B]
    train_y = []      # [T, B]

    encoder = EncoderRNN(input_dim, hidden_dim).to(device)
    decoder = AttnDecoderRNN(input_dim, hidden_dim, dropout_p=0.1, max_length=time_len).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    if args.load_epoch != 0:
        encoder.load_state_dict(torch.load(f'../models/encoder_{model_name}_' + str(args.load_epoch)))
        decoder.load_state_dict(torch.load(f'../models/decoder_{model_name}_' + str(args.load_epoch)))

    for i in range(1, args.epoch_number + 1):
        loss_total = 0  # Reset every print_every
        for idx in range(0, train_x.shape[1], batch_size):  # iterate each sone
            input_tensor = train_x[:, idx * batch_size:(idx + 1) * batch_size]
            target_tensor = train_y[:, idx * batch_size:(idx + 1) * batch_size]
            input_tensor = torch.tensor(input_tensor, dtype=torch.long)
            target_tensor = torch.tensor(target_tensor, dtype=torch.long)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            target_length = target_tensor.size(0)

            encoder_output, encoder_hidden = encoder(input_tensor)  # (time_len, batch_size, D)
            decoder_input = torch.zeros((target_tensor.size(1)), dtype=torch.float,
                                        device=device)
            decoder_hidden = encoder_hidden

            loss = 0
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                if di == 0:
                    decoder_input = torch.zeros((1, target_tensor.size(1)), dtype=torch.float,
                                                device=device)
                    decoder_hidden = encoder_hidden
                    context = torch.zeros(batch_size, hidden_dim).to(device)

                decoder_output, context, decoder_hidden, _ = decoder(decoder_input, context, decoder_hidden,
                                                         encoder_output)  # decoder_output：(1, B, D)
                loss += criterion(decoder_output, target_tensor[di])
                if torch.rand(1)[0] > threshold:
                    decoder_input = target_tensor[di]

                else:
                    decoder_input = decoder_output.argmax(dim=-1).detach()

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                loss_total += loss.item()

        print(f'{i} loss {loss_total}')
        if i % 50 == 0:
            torch.save(encoder.state_dict(), f'../models/mul_encoder_{model_name}_' + str(i + args.load_epoch))
            torch.save(decoder.state_dict(), f'../models/mul_decoder_{model_name}_' + str(i + args.load_epoch))







