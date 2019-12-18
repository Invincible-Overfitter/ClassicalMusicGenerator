import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 10

class EncoderRNN(nn.Module):
    def __init__(self, token_size, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(token_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size)

    def forward(self, input):         # [S, N]
        input = self.embedding(input)  # [S, N, E]
        output, hidden = self.gru(input)  # [S, N, H]
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, token_size, embed_size, hidden_size, embedding):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(embed_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden_size, token_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):   # [1, N, H]
        input = self.embedding(input)  # [1, N, E]
        output, hidden = self.gru(input, hidden)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, token_size, embed_size, hidden_size, embedding, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = embedding
        self.ref_embedding = nn.Embedding(token_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, token_size)

    def forward(self, input, hidden, encoder_outputs):
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        print("embedded shape: ", embedded.shape)

        print(embedded.shape)
        print(hidden.shape)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        '''
        input = self.embedding(input)  # [1, N, E]
        attn_weights = F.softmax(
            self.attn(torch.cat((input[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(0, 1)).transpose(0, 1)

        output = torch.cat((input[0], attn_applied[0]), 1)  #1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden, attn_weights
