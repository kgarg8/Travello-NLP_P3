from torch import nn, optim
import argparse
import sys
import utils
import os
import numpy as np
import random
import torch
import transformers as ppb
from utils import load_dataset, RunningAverage
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import logging
from sklearn.metrics import f1_score, recall_score, precision_score
import pdb

sys.path.insert(0, './database/features')

from labels1 import y1
from labels2 import y2
from addresses1 import X1_str
from addresses2 import X2_str

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        num_labels = 2
        model_class, tokenizer_class, pretrained_weights = (
            ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights)
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        last_hidden_states = self.bert(
            input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:, 0, :]
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


class AddressDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        tokenizer = ppb.BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized = list(map(lambda x: tokenizer.encode(
            x, add_special_tokens=True), X1_str))

        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        self.padded = np.array([i + [0] * (max_len - len(i))
                                for i in tokenized])

    def __len__(self, args):
        return len(self.padded)

    def __getitem__(self, index):
        attention_mask = np.where(self.padded[index] != 0, 1, 0)
        return torch.tensor(self.padded[index]), attention_mask, torch.tensor(y1[index])


def get_dataloaders(dataset, args):
    dataset_size = len(dataset.padded)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=val_sampler)

    return train_loader, val_loader


def accuracy(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / torch.numel(correct_pred)

    return acc.item()


def other_metrics(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    # pdb.set_trace()
    f1 = f1_score(y_test.cpu().numpy(), y_pred_tags.cpu().numpy())
    recall = recall_score(y_test.cpu().numpy(), y_pred_tags.cpu().numpy())
    precision = precision_score(
        y_test.cpu().numpy(), y_pred_tags.cpu().numpy())

    return f1, recall, precision


def train(loader, model, args, epoch):
    model.train()
    dataloader_iter = iter(loader)
    batch = 0
    total_loss = 0
    total_acc = 0
    while True:
        try:
            X, attention_mask, y = next(dataloader_iter)
        except RuntimeError:
            continue
        except StopIteration:
            break

        optimizer.zero_grad()
        y_pred = model(X.to(device), attention_mask.to(device))
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y.long().to(device))
        total_loss += loss.item()

        acc = accuracy(y_pred, y.long().to(device))
        total_acc += acc

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            logging.info({'epoch': epoch, 'batch': batch,
                          'train_loss': '{:05.4f}'.format(loss.item())})
        batch += 1

    logging.info({'epoch': epoch, 'train_loss': '{:05.4f}'.format(
        total_loss / batch),  'accuracy': '{:05.3f}'.format(total_acc / batch)})


def val(loader, model, args, epoch):
    model.eval()
    dataloader_iter = iter(loader)
    total_loss = 0
    total_acc = 0
    total_f1 = 0
    total_recall = 0
    total_precision = 0
    batch = 0
    while True:
        try:
            X, attention_mask, y = next(dataloader_iter)
        except RuntimeError:
            continue
        except StopIteration:
            break

        criterion = nn.CrossEntropyLoss()
        y_pred = model(X.to(device), attention_mask.to(device))
        loss = criterion(y_pred, y.long().to(device))
        total_loss += loss.item()

        acc = accuracy(y_pred, y.long().to(device))
        f1, recall, precision = other_metrics(y_pred, y.long().to(device))

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--model_dir', default='experiments/base_model')
    parser.add_argument('--restore_file', default='best',
                        help="Optional, file name from which reload weights before training e.g. 'best' or 'last' or None")
    parser.add_argument('--patience', type=int, default=5,
                        help="Epochs for early stopping")
    parser.add_argument('--seed', type=int, default=100,
                        help='Seed for randomization')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Size of validation set')
    parser.add_argument('--shuffle', action='store_true',
                        help='Flag for shuffling dataset')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(os.path.join(args.model_dir, 'train.log')):
        with open(os.path.join(args.model_dir, 'train.log'), 'w') as fp:
            pass

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info(args)

    dataset = AddressDataset(args)
    train_loader, val_loader = get_dataloaders(dataset, args)
    model = BertModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)

    metrics = {
        'accuracy': accuracy,
        # add more metrics if required for each token type
    }

    args.patience = 5
    best_val_acc = 0.0

    if args.restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

        filepath = args.model_dir + 'val_best_weights.json'
        if os.path.exists(filepath):
            f = open(filepath)
            data = json.load(f)
            best_val_acc = data['accuracy']
            f.close()

    for epoch in range(args.max_epochs):
        # train(train_loader, model, args, epoch)
        val_acc = val(val_loader, model, args, epoch)
        val_metrics = {'accuracy': val_acc}
        is_best = val_acc > best_val_acc

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

        if counter > args.patience:
            logging.info('- No improvement in a while, stopping training...')
            break

        last_json_path = os.path.join(
            args.model_dir, 'val_last_weights.json')
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == "__main__":
    main()
