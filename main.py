import torch
from torch import nn, optim
import sys
from utils import load_dataset, RunningAverage
import argparse
from torch.utils.data import DataLoader
import utils
import os
import logging
import pdb

sys.path.insert(0, './database/features')

from datavec1 import X1
from datavec2 import X2
from labels1 import y1
from labels2 import y2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self):
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
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.X_train, self.y_train, self.X_val, self.y_val = load_dataset(
            X1, y1, self.args.num_features)
        print(self.X_train.shape, self.y_train.shape,
              self.X_val.shape, self.y_val.shape)

    def __len__(self):
        if self.mode == 'train':
            return self.X_train.shape[0] - self.args.sequence_length - 1
        else:
            return self.X_val.shape[0] - self.args.sequence_length - 1

    def __getitem__(self, index):
        if self.mode == 'train':
            return torch.from_numpy(self.X_train[index:index + self.args.sequence_length]), torch.from_numpy(self.y_train[index:index + self.args.sequence_length])
        elif self.mode == 'val':
            return torch.from_numpy(self.X_val[index:index + self.args.sequence_length]), torch.from_numpy(self.y_val[index:index + self.args.sequence_length])


def accuracy(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / torch.numel(correct_pred)

    return acc.item()


def train(dataset, model, args, mode):
    model.train()
    loader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader_iter = iter(loader)
    state_h, state_c = model.init_state(args.sequence_length)
    batch = 0
    total_loss = 0
    total_acc = 0
    while True:
        try:
            # X - [16,4,8], y - [16, 4]
            X, y = next(dataloader_iter)
        except RuntimeError:
            continue
        except StopIteration:
            break

        optimizer.zero_grad()
        y_pred, (state_h, state_c) = model(X.to(device),
                                           (state_h.to(device), state_c.to(device)))
        loss = criterion(y_pred.transpose(1, 2), y.long().to(device))
        total_loss += loss.item()

        acc = accuracy(y_pred.transpose(1, 2), y.long().to(device))
        total_acc += acc

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            logging.info({'epoch': epoch, 'batch': batch,
                          'train_loss': '{:05.4f}'.format(loss.item())})
        batch += 1

    logging.info({'epoch': epoch, 'train_loss': '{:05.4f}'.format(
        total_loss / batch),  'accuracy': '{:05.3f}'.format(total_acc / batch)})


def val(dataset, model, args, mode):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader_iter = iter(loader)
    state_h, state_c = model.init_state(args.sequence_length)
    total_loss = 0
    total_acc = 0
    batch = 0
    while True:
        try:
            X, y = next(dataloader_iter)
        except RuntimeError:
            continue
        except StopIteration:
            break

        y_pred, (state_h, state_c) = model(X.to(device),
                                           (state_h.to(device), state_c.to(device)))
        loss = criterion(y_pred.transpose(
            1, 2), y.long().to(device))
        total_loss += loss.item()

        acc = accuracy(y_pred.transpose(1, 2), y.long().to(device))
        total_acc += acc
        batch += 1

    logging.info({'epoch': epoch, 'val_loss': '{:05.4f}'.format(
        total_loss / batch),  'accuracy': '{:05.3f}'.format(total_acc / batch)})
    return total_acc / batch


utils.set_logger(os.path.join(args.model_dir, 'train.log'))

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--sequence-length', type=int, default=4)
parser.add_argument('--num-features', type=int, default=8)
parser.add_argument('--model_dir', default='experiments/base_model')
parser.add_argument('--restore_file', default='best',
                    help="Optional, file name from which reload weights before training e.g. 'best' or 'last'")
args = parser.parse_args()
logging.info(args)

train_set = Dataset(args, 'train')
val_set = Dataset(args, 'val')
model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

metrics = {
    'accuracy': accuracy,
    # add more metrics if required for each token type
}

patience = 5
best_val_acc = 0.0

if args.restore_file is not None:
    restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    logging.info('Restoring parameters from {}'.format(restore_path))
    utils.load_checkpoint(restore_path, model, optimizer)

    filepath = args.model_dir + 'val_best_weights.json'
    if os.path.exists(filepath):
        f = open(filepath)
        data = json.load(f)
        best_val_acc = data['accuracy']
        f.close()

for epoch in range(args.max_epochs):
    train(train_set, model, args, 'train')
    val_acc = val(val_set, model, args, 'val')
    val_metrics = {'accuracy': val_acc}
    is_best = val_acc >= best_val_acc

    utils.save_checkpoint({'epoch': epoch + 1,
                           'state_dict': model.state_dict(),
                           'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=args.model_dir)

    if is_best:
        logging.info('- Found new best accuracy')
        counter = 0  # reset counter
        best_val_acc = val_acc

        best_json_path = os.path.join(
            args.model_dir, 'val_best_weights.json')
        utils.save_dict_to_json(val_metrics, best_json_path)
    else:
        counter += 1

    if counter > patience:
        logging.info('- No improvement in a while, stopping training...')
    last_json_path = os.path.join(
        args.model_dir, 'val_last_weights.json')
    utils.save_dict_to_json(val_metrics, last_json_path)
