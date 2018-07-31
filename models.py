import torch.nn as nn
import torch

class MainRNNModel(nn.Module):
    """Container module with an encoder, and a recurrent module for main classification network."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhid_ffn, nlayers, dropconnect=0.5):
        super(MainRNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity)
        self.pre_decoder = nn.Linear(nhid, nhid_ffn)
        self.decoder = nn.Linear(nhid_ffn, 10)

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhid_ffn = nhid_ffn
        self.nlayers = nlayers
        self.dropconnect = dropconnect

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.pre_decoder.weight.data.uniform_(-initrange, initrange)
        self.pre_decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, input, hidden):
        if self.ninp == 1:
            input = input.float()
            emb = input.view(input.size(0), input.size(1), 1)
        else:
            emb = self.encoder(input)
        _, hidden = self.rnn(emb, hidden)
        return hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def out(self, hidden):
        h, _ = hidden
        h = h[-1]  # hidden state of last layer

        pre_decoded = self.pre_decoder(h)

        if self.training == True:
            mask = self.decoder.weight.data.new().resize_(self.decoder.weight.size()).bernoulli_(1- self.dropconnect) / (1 - self.dropconnect)
            weight = mask * self.decoder.weight
        else:
            weight = self.decoder.weight
        decoded = torch.mm(pre_decoded, weight.t()) + self.decoder.bias.data
        return decoded


class AuxiliaryRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder for auxiliary network."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhid_ffn, nlayers, dropconnect=0.5):
        super(AuxiliaryRNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity)
        
        self.pre_decoder = nn.Linear(nhid, nhid_ffn)
        self.decoder = nn.Linear(nhid_ffn, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhid_ffn = nhid_ffn
        self.nlayers = nlayers
        self.dropconnect = dropconnect

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.pre_decoder.weight.data.uniform_(-initrange, initrange)
        self.pre_decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, input, hidden):
        if self.ninp == 1:
            input = input.float()
            emb = input.view(input.size(0), input.size(1), 1)
        else:
            emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        
        pre_decoded = self.pre_decoder(output.view(output.size(0)*output.size(1), output.size(2)))

        if self.training:
            mask = self.decoder.weight.data.new().resize_(self.decoder.weight.size()).bernoulli_(1- self.dropconnect) / (1 - self.dropconnect)
            weight = mask * self.decoder.weight
        else:
            weight = self.decoder.weight
        decoded = torch.mm(pre_decoded, weight.t()) + self.decoder.bias.data
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, hidden=None):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            new_h = weight.new_zeros(self.nlayers, bsz, self.nhid)
            new_c = weight.new_zeros(self.nlayers, bsz, self.nhid)
            if hidden:
                h, c = hidden
                new_h[0], new_c[0] = h[-1], c[-1]
            return (new_h, new_c)
        else:
            new_h = weight.new_zeros(self.nlayers, bsz, self.nhid)
            if hidden:
                new_h[0] = hidden[-1]
            return new_h
