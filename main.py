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
        # self.lstm_size = 128
        self.lstm_size = 8  # Hardcoded, TO CHANGE
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = 2  # TO CHANGE
        # n_vocab = len(dataset.uniq_words)
        # self.embedding = nn.Embedding(
        #     num_embeddings=n_vocab,
        #     embedding_dim=self.embedding_dim,
        # )
        self.lstm = nn.LSTM(
            # input_size=self.lstm_size,
            input_size=8,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        # embed = self.embedding(x) # v2: create embeddings using NN
        embed = x.float()  # v1: only hand-designed features
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args,):
        self.args = args
        self.X_train, self.y_train, self.X_val, self.y_val = load_dataset(
            X1, y1, self.args.num_features)

    def __len__(self):
        return self.X_train.shape[0] - self.args.sequence_length - 1

    def __getitem__(self, index):
        return torch.from_numpy(self.X_train[index:index + self.args.sequence_length]), torch.from_numpy(self.y_train[index:index + self.args.sequence_length]), torch.from_numpy(self.X_val[index:index + self.args.sequence_length]), torch.from_numpy(self.y_val[index:index + self.args.sequence_length])


def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        dataloader_iter = iter(dataloader)
        batch = 0
        while(dataloader_iter):
            try:
                # X_train - [16,4,8], y_train - [16, 4]
                X_train, y_train, X_val, y_val = next(dataloader_iter)
            except RuntimeError:
                continue

            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(X_train.to(device),
                                               (state_h.to(device), state_c.to(device)))
            train_loss = criterion(y_pred.transpose(
                1, 2), y_train.long().to(device))

            state_h = state_h.detach()
            state_c = state_c.detach()

            train_loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print({'epoch': epoch, 'batch': batch,
                       'train_loss': train_loss.item()})

            # TODO: Validation
            # if batch % 500 == 0:
                # y_pred_val, (state_h, state_c) = model(X_val.to(device),
                #                                        (state_h.to(device), state_c.to(device)))
                # val_loss = criterion(y_pred_val.transpose(
                #     1, 2), y_val.long().to(device))
                # print({'epoch': epoch, 'batch': batch,
                #        'val_loss': val_loss.item()})

            batch += 1


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
