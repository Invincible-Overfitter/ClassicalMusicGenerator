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
        # self.ref_embedding = nn.Embedding(token_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRUCell(self.embed_size + self.hidden_size, self.hidden_size)
        self.prob_1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 4)
        self.prob_2 = nn.Linear(self.hidden_size * 4, token_size)

    def forward(self, input, context, hidden, encoder_outputs):
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        print("embedded shape: ", embedded.shape)

        print(embedded.shape)
        print(hidden.shape)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        '''

        input = self.embedding(input)  # [N, H]
        query = self.gru(torch.cat([input, context], dim=1), hidden)

        # Input/output shape of bmm: (N, T, H), (N, H, 1) -> (N, T, 1)
        energy = torch.bmm(encoder_outputs.transpose(0, 1), query.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(energy, dim=1)

        # Input/output shape of bmm: (N, 1, T), (N, T, H) -> (N, 1, H)
        context = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(0, 1)).squeeze(1)

        output = torch.cat((query, context), 1)  #1)
        output = self.prob_1(output)
        output = F.relu(output)
        output = self.prob_2(output)
        return output, context, query, attn_weights
