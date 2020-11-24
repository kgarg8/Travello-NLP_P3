import torch
from torch import nn, optim
import sys
import numpy as np
from utils import load_dataset, RunningAverage
import argparse
from torch.utils.data import DataLoader
import transformers as ppb
import utils
import os
import logging
import pdb
from collections import Counter
from string import punctuation

sys.path.insert(0, './database/features')
from datavec1 import X1_num
from addresses1 import X1_str
from labels1 import y1
from main_bert import AddressDataset, get_dataloaders

from sklearn.metrics import f1_score, recall_score, precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LstmModel(nn.Module):
    def __init__(self, args):
        super(LstmModel, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128  # v3
        self.num_layers = 3

        len_dataset = len(X1_num)
        self.embedding = nn.Embedding(
            num_embeddings=len_dataset,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, 2)

    def forward(self, x, x_int, prev_state):
        # v3: only NN embeddings
        embed = self.embedding(x_int).mean(dim=2)
        output, state = self.lstm(embed.float(), prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


class AddressDataset2(torch.utils.data.Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        for i, address in enumerate(X1_str):
            X1_str[i] = ''.join([c for c in address if c not in punctuation])

        all_text2 = ' '.join(X1_str)
        words = all_text2.split()
        count_words = Counter(words)
        total_words = len(words)
        sorted_words = count_words.most_common(total_words)
        vocab_to_int = {w: i for i, (w, c) in enumerate(sorted_words)}
        X1_int = []
        for address in X1_str:
            encoded = [vocab_to_int[w] for w in address.split()]
            X1_int.append(encoded)

        X1_int = self.pad_features(X1_int, 40)
        val_split = round(len(X1_int) * 0.2)
        self.X_train = torch.tensor(X1_int[:-val_split])
        self.y_train = torch.tensor(y1[:-val_split])
        self.X_val = torch.tensor(X1_int[-val_split:])
        self.y_val = torch.tensor(y1[-val_split:])

    def pad_features(self, reviews_int, seq_length):
        ''' Return features of address_ints, where each address is padded with 0's or truncated to the input seq_length.
        '''
        features = np.zeros((len(reviews_int), seq_length), dtype=int)

        for i, review in enumerate(reviews_int):
            review_len = len(review)

            if review_len <= seq_length:
                zeroes = list(np.zeros(seq_length - review_len))
                new = zeroes + review
            elif review_len > seq_length:
                new = review[0:seq_length]

            features[i, :] = np.array(new)
        return features

    def __len__(self):
        if self.mode == 'train':
            return len(self.X_train) - self.args.sequence_length - 1
        else:
            return len(self.X_val) - self.args.sequence_length - 1

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.X_train[index:index + self.args.sequence_length], self.y_train[index:index + self.args.sequence_length]
        elif self.mode == 'val':
            return self.X_val[index:index + self.args.sequence_length], self.y_val[index:index + self.args.sequence_length]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.X_train, self.y_train, self.X_val, self.y_val = load_dataset(
            X1_num, y1, self.args.num_features)
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


def other_metrics(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    f1 = f1_score(y_test[:, 0].cpu().numpy(), y_pred_tags[:, 0].cpu().numpy())
    recall = recall_score(y_test[:, 0].cpu().numpy(),
                          y_pred_tags[:, 0].cpu().numpy())
    precision = precision_score(
        y_test[:, 0].cpu().numpy(), y_pred_tags[:, 0].cpu().numpy())

    return f1, recall, precision


def train(dataset, dataset2, model, args, mode):
    model.train()
    spt_loader = DataLoader(dataset, batch_size=args.batch_size)
    loader = DataLoader(dataset2, batch_size=args.batch_size)
    dataloader_iter = iter(loader)
    spt_dataloader_iter = iter(spt_loader)
    state_h, state_c = model.init_state(args.sequence_length)
    batch = 0
    total_loss = 0
    total_acc = 0
    while True:
        try:
            X, y = next(spt_dataloader_iter)
            X_str, y2 = next(dataloader_iter)
        except RuntimeError:
            continue
        except StopIteration:
            break

        optimizer.zero_grad()
        y_pred, (state_h, state_c) = model(X.to(device), X_str.to(device),
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


def val(dataset, dataset2, model, args, mode):
    model.eval()
    spt_loader = DataLoader(dataset, batch_size=args.batch_size)
    loader = DataLoader(dataset2, batch_size=args.batch_size)
    dataloader_iter = iter(loader)
    spt_dataloader_iter = iter(spt_loader)
    state_h, state_c = model.init_state(args.sequence_length)
    total_loss = 0
    total_acc = 0
    total_f1 = 0
    total_recall = 0
    total_precision = 0
    batch = 0
    while True:
        try:
            X, y = next(spt_dataloader_iter)
            X_str, y2 = next(dataloader_iter)
        except RuntimeError:
            continue
        except StopIteration:
            break

        y_pred, (state_h, state_c) = model(X.to(device), X_str.to(device),
                                           (state_h.to(device), state_c.to(device)))
        loss = criterion(y_pred.transpose(
            1, 2), y.long().to(device))
        total_loss += loss.item()

        acc = accuracy(y_pred.transpose(1, 2), y.long().to(device))
        f1, recall, precision = other_metrics(
            y_pred.transpose(1, 2), y.long().to(device))

        total_acc += acc
        total_f1 += f1
        total_recall += recall
        total_precision += precision

        batch += 1

    logging.info({'epoch': epoch, 'val_loss': '{:05.4f}'.format(
        total_loss / batch),  'accuracy': '{:05.3f}'.format(total_acc / batch)})
    logging.info({'f1-score': '{:05.3f}'.format(total_f1 / batch), 'recall': '{:05.3f}'.format(
        total_recall / batch), 'precision': '{:05.3f}'.format(total_precision / batch)})

    return total_acc / batch


parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--sequence-length', type=int, default=4)
parser.add_argument('--num-features', type=int, default=8)
parser.add_argument('--model_dir', default='experiments/base_model')
parser.add_argument('--restore_file', default='best',
                    help="Optional, file name from which reload weights before training e.g. 'best' or 'last' or 'None'")
parser.add_argument('--patience', type=int, default=5,
                    help="Epochs for early stopping")
parser.add_argument('--seed', type=int, default=100,
                    help='Seed for randomization')
parser.add_argument('--val_split', type=float, default=0.2,
                    help='Size of validation set')
parser.add_argument('--shuffle', action='store_true',
                    help='Flag for shuffling dataset')
args = parser.parse_args()

utils.set_logger(os.path.join(args.model_dir, 'train.log'))
logging.info(args)

dataset_len = 0
train_set = Dataset(args, 'train')
val_set = Dataset(args, 'val')
train_set2 = AddressDataset2(args, 'train')
val_set2 = AddressDataset2(args, 'val')

model = LstmModel(args).to(device)

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
    train(train_set, train_set2, model, args, 'train')
    val_acc = val(val_set, val_set2, model, args, 'val')
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
