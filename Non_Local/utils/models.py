import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = "cuda" if torch.cuda.is_available else "cpu"
class RNNbase(nn.Module):
    def __init__(self, input_size=9, classes=5, embedding_size=64, hidden_size=128, drop_p=0.2):
        super(RNNbase, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(drop_p)

        self.embed = nn.Linear(9, self.embedding_size)
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.classify = nn.Sequential(
                        nn.Linear(self.hidden_size, self.classes),
                        nn.LogSoftmax(dim=1)
                        )

    def forward(self, input_data):
        x = input_data / input_data.norm(p=2, dim=2, keepdim=True).detach()
        x = self.dropout(self.embed(x))
        rnn_output, h_n = self.rnn(x) # h_n: (1, batch, hidden)
        pred = self.classify(h_n[-1])

        return pred
        
class Nonlocalbase(nn.Module):
    def __init__(self, input_size=9, classes=5, embedding_size=64, hidden_size=128, drop_p=0.25):
        super(Nonlocalbase, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # self.mean = torch.tensor([0.0125, 0.0125, 0.0125, 0.86909, 0.86909, 0.86909, -1.66461, -1.66461, -1.66461], dtype=torch.float).to(device)
        # self.std = torch.tensor([0.96046, 0.96046, 0.96046, 5.9774, 5.9774, 5.9774, 91.66598, 91.66598, 91.66598], dtype=torch.float).to(device)

        self.dropout = nn.Dropout(drop_p)

        self.embed = nn.Linear(input_size, self.embedding_size)

        self.conv1 = nn.Conv1d(embedding_size, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(embedding_size, 32, kernel_size=1)
        self.conv3 = nn.Conv1d(embedding_size, 32, kernel_size=1)
        self.conv4 = nn.Conv1d(32, embedding_size, kernel_size=1)

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.classify = nn.Sequential(
                        nn.Linear(self.hidden_size, self.classes),
                        nn.LogSoftmax(dim=1)
                        )

    def forward(self, x):
        # x = ((x - self.mean) / self.std + torch.randn_like(x) * 0.02).detach()
        x = self.dropout(self.embed(x)) # (batch, 128, embedding_size)
        x = x.transpose(1, 2) #           (batch, embedding_size, 128) ~ (batch, C, THW)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        A = torch.matmul(x1.transpose(1, 2), x2) # (128, 128) ~ (THW, THW)
        A = F.softmax(A, dim=1)

        h = torch.matmul(x3, A)
        h = self.conv4(h).transpose(1, 2) # (batch, 128, embedding_size)

        rnn_output, h_n = self.rnn(h) # h_n: (1, batch, hidden)
        # h = self.mlp(h).squeeze(2)
        pred = self.classify(h_n[-1])

        return pred

class RNNatt(nn.Module):
    def __init__(self, input_size=9, classes=5, embedding_size=64, hidden_size=128, window_size=130, drop_p=0.2):
        super(RNNatt, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.mean = torch.tensor([-0.00475, 0.04073, -0.00207, -0.04262, 0.90025, 1.7851, -0.61606, -2.40396, -3.45526], dtype=torch.float).to(device)
        self.std = torch.tensor([0.90533, 1.24434, 0.73502, 5.96793, 6.2825, 5.54391, 103.20517, 78.90106, 89.22932], dtype=torch.float).to(device)

        self.dropout = nn.Dropout(drop_p)

        self.embedd = nn.Linear(self.input_size, self.embedding_size)

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.window_size)

        self.classify = nn.Sequential(
                        nn.Linear(self.hidden_size, self.classes),
                        nn.LogSoftmax(dim=1)
                        )

    def forward(self, x):
        x = ((x - self.mean) / self.std + torch.randn_like(x) * 0.02).detach()
        x = torch.relu(self.dropout(self.embedd(x)))
        rnn_output, h_n = self.rnn(x) # h_n: (1, batch, hidden)
        att = self.attn(h_n[-1])
        attn_weights = F.softmax(att, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), rnn_output)
        pred = self.classify(attn_applied.squeeze(1))

        return pred



if __name__ == "__main__":
    model = RNNatt()
    x = torch.randn(64, 128, 9)

    y = model(x)
    print("x: {}, y: {}".format(x.size(), y.size()))
