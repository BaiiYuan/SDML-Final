import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython import embed

device = "cuda" if torch.cuda.is_available else "cpu"
class RNNbase(nn.Module):
    def __init__(self, input_size=39, classes=5, embedding_size=64, hidden_size=64, window_size=128, drop_p=0.5):
        super(RNNbase, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.dropout = nn.Dropout(drop_p)

        self.embedd = nn.Sequential(
                      nn.Linear(self.input_size, self.embedding_size),
                      nn.ReLU() # Can not add Dropout
                      )

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.classify = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(self.hidden_size * self.window_size, 256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256, self.classes),
                        nn.ReLU(),
                        nn.LogSoftmax(dim=1)
                        )

    def forward(self, input_data):
        # x = input_data / input_data.norm(p=2, dim=0, keepdim=True).norm(p=2, dim=1, keepdim=True).detach()
        # x = (input_data - input_data.mean(dim=0, keepdim=True).detach()) / input_data.std(dim=0, keepdim=True).detach()
        # x = input_data
        x = self.dropout(self.embedd(input_data))
        rnn_output, h_n = self.rnn(x) # h_n: (1, batch, hidden)
        # embed()
        pred = self.classify(rnn_output.reshape(rnn_output.size()[0], -1))

        return pred

class RNNatt(nn.Module):
    def __init__(self, input_size=39, classes=5, embedding_size=32, hidden_size=128, window_size=128, drop_p=0.5):
        super(RNNatt, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.dropout = nn.Dropout(drop_p)

        self.embedd = nn.Linear(self.input_size, self.embedding_size)

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        # self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.window_size)

        self.classify = nn.Sequential(
                        nn.Linear(self.hidden_size, self.classes),
                        nn.LogSoftmax(dim=1)
                        )

    def forward(self, input_data):
        x = self.embedd(input_data+1e-2*torch.rand_like(input_data))
        x = F.relu(x)
        # embed()
        # x = x + 1e-2*torch.rand_like(x) # noise
        x = self.dropout(x)
        rnn_output, h_n = self.rnn(x) # h_n: (1, batch, hidden)
        # rnn_output, h_n = self.lstm(x);h_n, c_n = h_n
        att = self.attn(h_n[-1])
        attn_weights = F.softmax(att, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), rnn_output)
        # attn_applied = attn_applied + 1e-2*torch.rand_like(attn_applied) # noise
        pred = self.classify(attn_applied.squeeze(1))

        return pred

if __name__ == "__main__":
    model = RNNatt()
    seq_in = torch.randn(64, 128, 39)

    y = model(seq_in)
    print("seq_in: {}, y: {}".format(seq_in.size(), y.size()))
