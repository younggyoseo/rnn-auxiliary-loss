import torch.nn as nn
import torch

class MainRNNModel(nn.Module):
    """Container module with an encoder, and a recurrent module for main classification network."""

    def __init__(self, ntoken, ninp, nout, nhid, nhid_ffn, nlayers, dropconnect=0.5):
        super(MainRNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers)

        self.pre_decoder = nn.Linear(nhid, nhid_ffn)
        self.decoder = nn.Linear(nhid_ffn, nout)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
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
        emb = self.encoder(input)
        _, hidden = self.rnn(emb, hidden)
        return hidden

    def init_hidden(self, bsz, hidden=None):
        weight = next(self.parameters())
        new_h = weight.new_zeros(self.nlayers, bsz, self.nhid)
        new_c = weight.new_zeros(self.nlayers, bsz, self.nhid)
        if hidden:
            h, c = hidden
            new_h[0], new_c[0] = h[-1], c[-1]
        return (new_h, new_c)

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


class AuxRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder for auxiliary network."""

    def __init__(self, ntoken, ninp, nout, nhid, nhid_ffn, nlayers, dropconnect=0.5):
        super(AuxRNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers)

        self.pre_decoder = nn.Linear(nhid, nhid_ffn)
        self.decoder = nn.Linear(nhid_ffn, nout)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
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
        new_h = weight.new_zeros(self.nlayers, bsz, self.nhid)
        new_c = weight.new_zeros(self.nlayers, bsz, self.nhid)
        if hidden:
            h, c = hidden
            new_h[0], new_c[0] = h[-1], c[-1]
        return (new_h, new_c)
