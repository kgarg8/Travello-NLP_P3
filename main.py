import torch
from torch import nn, optim
import sys
from utils import load_dataset
import argparse
from torch.utils.data import DataLoader
import pdb

sys.path.insert(0, './database/features')

from datavec1 import X1
from datavec2 import X2
from labels1 import y1
from labels2 import y2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = 1000 #TO CHANGE
        # n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args,):
        self.args = args
        self.X_train, self.y_train, self.X_val, self.y_val = load_dataset(X1, y1, self.args.num_features)

    def __len__(self):
        return self.args.batch_size

    def __getitem__(self, index):
        return torch.from_numpy(self.X_train[index:index + self.args.sequence_length]), torch.from_numpy(self.y_train[index:index + self.args.sequence_length]), torch.from_numpy(self.X_val[index:index + self.args.sequence_length]), torch.from_numpy(self.y_val[index:index + self.args.sequence_length])

def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)

        for batch, (X_train, y_train, X_val, y_val) in enumerate(dataloader):
            # X_train - [16,4,8], y_train - [16, 4]
            pdb.set_trace()
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(X_train.to(device),
                                               (state_h.to(device), state_c.to(device)))
            loss = criterion(y_pred.transpose(1, 2), y_train.to(device))

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})


parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--sequence-length', type=int, default=4)
parser.add_argument('--num-features', type=int, default=8)
args = parser.parse_args()
print(args)

dataset = Dataset(args)
model = Model(dataset).to(device)
train(dataset, model, args)
