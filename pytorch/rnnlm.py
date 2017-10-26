from __future__ import division, print_function
import time
import sys
import random
import argparse
from itertools import count
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('MB_SIZE', type=int, help='minibatch size')
parser.add_argument('EMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
parser.add_argument('--CUDA', type=int, default=-1, help='use CUDA')
args = parser.parse_args()

train_file = 'data/text/train.txt'
test_file = 'data/text/dev.txt'

def read(fname):
    """
    Read a file where each line is of the form "word1 word2 ..."
    Yields lists of the form [word1, word2, ...]
    """
    with open(fname, "r") as fh:
        for line in fh:
            sent = [w2i[x] for x in line.strip().split()]
            sent.append(w2i["<s>"])
            yield torch.LongTensor(sent)

w2i = defaultdict(count(0).next)
mask = w2i['<MASK>']
assert mask == 0
train = list(read(train_file))
vocab_size = len(w2i)
test = list(read(test_file))
S = w2i['<s>']

def get_batch(sequences, volatile=False):
    lengths = torch.LongTensor([len(s) for s in sequences])
    batch = torch.LongTensor(lengths.max(), len(sequences)).fill_(mask)
    for i, s in enumerate(sequences):
        batch[:len(s), i] = s
    if args.CUDA:
        batch = batch.cuda()
    return Variable(batch, volatile=volatile), lengths

class RNNLM(nn.Module):
    def __init__(self):
        super(RNNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, args.EMBED_SIZE)
        self.rnn = nn.RNN(args.EMBED_SIZE, args.HIDDEN_SIZE)
        self.proj = nn.Linear(args.HIDDEN_SIZE, vocab_size)
    def forward(self, sequences):
        rnn_output, _ = self.rnn(self.embeddings(sequences))
        return self.proj(rnn_output.view(-1, args.HIDDEN_SIZE))

# build the model
rnnlm = RNNLM()
optimizer = optim.Adam(rnnlm.parameters(), lr=0.001)
weight = torch.FloatTensor(vocab_size).fill_(1)
weight[mask] = 0
loss_fn = nn.CrossEntropyLoss(weight, size_average=False)
if args.CUDA:
    rnnlm.cuda()
    loss_fn.cuda()

# Sort training sentences in descending order and count minibatches
train.sort(key=lambda x: -len(x))
test.sort(key=lambda x: -len(x))
train_order = range(0, len(train), args.MB_SIZE)  # [x*args.MB_SIZE for x in range(int((len(train)-1)/args.MB_SIZE + 1))]
test_order = range(0, len(test), args.MB_SIZE)  # [x*args.MB_SIZE for x in range(int((len(test)-1)/args.MB_SIZE + 1))]

# Perform training
print("startup time: %r" % (time.time() - start))
start = time.time()
i = total_time = dev_time = total_tagged = current_words = current_loss = 0

for ITER in range(100):
    random.shuffle(train_order)
    for sid in train_order:
        i += 1
        # train
        batch, lengths = get_batch(train[sid:sid + args.MB_SIZE])
        scores = rnnlm(batch[:-1])
        loss = loss_fn(scores, batch[1:].view(-1))
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log loss
        current_words += lengths.sum() - lengths.size(0)  # ignore <s>
        current_loss += loss.data[0]
        if i % int(500 / args.MB_SIZE) == 0:
            print(current_loss / current_words)
            total_tagged += current_words
            current_loss = current_words = 0
            total_time = time.time() - start
        # log perplexity
        if i % int(10000 / args.MB_SIZE) == 0 or total_time > args.TIMEOUT:
            dev_start = time.time()
            dev_loss = dev_words = 0
            for j in test_order:
                batch, lengths = get_batch(test[j:j + args.MB_SIZE], volatile=True)
                scores = rnnlm(batch[:-1])
                dev_loss += loss_fn(scores, batch[1:].view(-1)).data[0]
                dev_words += lengths.sum() - lengths.size(0)  # ignore <s>
            dev_time += time.time() - dev_start
            train_time = time.time() - start - dev_time
            print("nll=%.4f, ppl=%.4f, words=%r, time=%.4f, word_per_sec=%.4f" % (
                dev_loss / dev_words, np.exp(dev_loss / dev_words), dev_words, train_time, total_tagged / train_time))
        if total_time > args.TIMEOUT:
            sys.exit(0)

    print("epoch %r finished" % ITER)
