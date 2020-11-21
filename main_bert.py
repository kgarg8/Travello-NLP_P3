import torch
from torch import nn, optim
import sys
from utils import load_dataset, RunningAverage
import argparse
from torch.utils.data import DataLoader
import transformers as ppb
import utils
import os
import numpy as np
import logging
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
        model_class, tokenizer_class, pretrained_weights = (
            ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights)
        self.dropout = nn.Dropout()
        # self.classifier =

    def forward(self, input_ids, attention_mask):
        last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:,0,:]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        tokenizer = ppb.BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized = list(map(lambda x: tokenizer.encode(x, add_special_tokens=True), X1_str[0])) # incorrect: replace by X1_str
        
        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        val_len = len(padded)//6
        self.X_train, self.y_train, self.X_val, self.y_val = torch.tensor(padded[:-val_len]), torch.tensor(y1[:-val_len]), torch.tensor(padded[-val_len:]), torch.tensor(y1[-val_len:])
        print(self.X_train.shape, self.y_train.shape,
              self.X_val.shape, self.y_val.shape)

    def __len__(self):
        if self.mode == 'train':
            return self.X_train.shape[0]
            pdb.set_trace()
        else:
            return self.X_val.shape[0]

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.X_train, self.y_train
        elif self.mode == 'val':
            return self.X_val, self.y_val


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
        y_pred, (state_h, state_c) = model(X.to(device))
        loss = criterion(y_pred.transpose(1, 2), y.long().to(device))
        total_loss += loss.item()

        acc = accuracy(y_pred.transpose(1, 2), y.long().to(device))
        total_acc += acc

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

        y_pred, (state_h, state_c) = model(X.to(device))
        loss = criterion(y_pred.transpose(
            1, 2), y.long().to(device))
        total_loss += loss.item()

        acc = accuracy(y_pred.transpose(1, 2), y.long().to(device))
        total_acc += acc
        batch += 1

    logging.info({'epoch': epoch, 'val_loss': '{:05.4f}'.format(
        total_loss / batch),  'accuracy': '{:05.3f}'.format(total_acc / batch)})
    return total_acc / batch


parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=3)
parser.add_argument('--sequence-length', type=int, default=4)
parser.add_argument('--num-features', type=int, default=8)
parser.add_argument('--model_dir', default='experiments/base_model')
parser.add_argument('--restore_file', default='best',
                    help="Optional, file name from which reload weights before training e.g. 'best' or 'last'")
args = parser.parse_args()

utils.set_logger(os.path.join(args.model_dir, 'train.log'))
logging.info(args)

train_set = Dataset(args, 'train')
val_set = Dataset(args, 'val')
model = BertModel().to(device)

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
        break
    
    last_json_path = os.path.join(
        args.model_dir, 'val_last_weights.json')
    utils.save_dict_to_json(val_metrics, last_json_path)
