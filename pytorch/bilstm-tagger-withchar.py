# coding: utf-8
from __future__ import print_function
import time
start = time.time()

from collections import Counter, defaultdict
import random
import sys
import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--CEMBED_SIZE', default=32, type=int)
parser.add_argument('--WEMBED_SIZE', default=128, type=int)
parser.add_argument('--HIDDEN_SIZE', default=256, type=int)
parser.add_argument('--MLP_SIZE', default=256, type=int)
parser.add_argument('--TIMEOUT', default=600, type=int)
parser.add_argument('--CUDA', default=1, type=int)
args = parser.parse_args()


# format of files: each line is "word1|tag2 word2|tag2 ..."
train_file = "/home/guismay/projects/dynet-benchmark/data/tags/train.txt"
dev_file = "/home/guismay/projects/dynet-benchmark/data/tags/dev.txt"


class Vocab:

    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(lambda: len(w2i))
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(lambda: len(w2i))
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self):
        return len(self.w2i.keys())


def read(fname):
    """
    Read a POS-tagged file where each line is of the form "word1|tag2 word2|tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    with open(fname, "r") as fh:
        for line in fh:
            line = line.strip().split()
            sent = [tuple(x.rsplit("|", 1)) for x in line]
            yield sent


train = list(read(train_file))
dev = list(read(dev_file))
words = []
tags = []
chars = set()
wc = Counter()
for sent in train:
    for w, p in sent:
        words.append(w)
        tags.append(p)
        wc[w] += 1
        chars.update(w)
words.append("_UNK_")
chars.add("_UNK_")
chars.add("<*>")

vw = Vocab.from_corpus([words])
vt = Vocab.from_corpus([tags])
vc = Vocab.from_corpus([chars])
UNK = vw.w2i["_UNK_"]
CUNK = vc.w2i["_UNK_"]
pad_char = vc.w2i["<*>"]

nwords = vw.size()
ntags = vt.size()
nchars = vc.size()
print ("nwords=%r, ntags=%r, nchars=%r" % (nwords, ntags, nchars))


def get_var(x, volatile=False):
    x = Variable(x, volatile=volatile)
    return x.cuda() if args.CUDA else x


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.lookup_w = nn.Embedding(nwords, args.WEMBED_SIZE, padding_idx=UNK)
        self.lookup_c = nn.Embedding(nchars, args.CEMBED_SIZE, padding_idx=CUNK)
        self.lstm = nn.LSTM(args.WEMBED_SIZE, args.HIDDEN_SIZE, 1, bidirectional=True)
        self.lstm_c_f = nn.LSTM(args.CEMBED_SIZE, args.WEMBED_SIZE / 2, 1)
        self.lstm_c_r = nn.LSTM(args.CEMBED_SIZE, args.WEMBED_SIZE / 2, 1)
        self.proj1 = nn.Linear(2 * args.HIDDEN_SIZE, args.MLP_SIZE)
        self.proj2 = nn.Linear(args.MLP_SIZE, ntags)

    def forward(self, words, volatile=False):
        word_ids = []
        needs_chars = []
        char_ids = []
        for i, w in enumerate(words):
            if wc[w] > 5:
                word_ids.append(vw.w2i[w])
            else:
                word_ids.append(UNK)
                needs_chars.append(i)
                char_ids.append([pad_char] + [vc.w2i.get(c, CUNK) for c in w] + [pad_char])
        embeddings = self.lookup_w(get_var(torch.LongTensor(word_ids), volatile=volatile))
        if needs_chars:
            max_len = max(len(x) for x in char_ids)
            fwd_char_ids = [ids + [pad_char for _ in range(max_len - len(ids))] for ids in char_ids]
            rev_char_ids = [ids[::-1] + [pad_char for _ in range(max_len - len(ids))] for ids in char_ids]
            char_embeddings = torch.cat([
                    self.lstm_c_f(self.lookup_c(get_var(torch.LongTensor(fwd_char_ids).t())))[0],
                    self.lstm_c_r(self.lookup_c(get_var(torch.LongTensor(rev_char_ids).t())))[0]
                ], 2)
            unk_embeddings = torch.cat([char_embeddings[len(words[j]) + 1, i].unsqueeze(0) for i, j in enumerate(needs_chars)], 0)
            embeddings = embeddings.index_add(0, get_var(torch.LongTensor(needs_chars)), unk_embeddings)
        return self.proj2(self.proj1(self.lstm(embeddings.unsqueeze(1))[0].squeeze(1)))


model = Model(args)
if args.CUDA:
    model.cuda()
optimizer = optim.Adam(model.parameters())


print("startup time: %r" % (time.time() - start))
start = time.time()
i = all_time = dev_time = all_tagged = this_tagged = this_loss = 0

for ITER in range(100):
    random.shuffle(train)
    for s in train:
        i += 1
        if i % 500 == 0:
            print(this_loss / this_tagged, file=sys.stderr)
            all_tagged += this_tagged
            this_loss = this_tagged = 0
            all_time = time.time() - start
        if i % 10000 == 0 or all_time > args.TIMEOUT:  # eval on dev
            dev_start = time.time()
            good_sent = bad_sent = good = bad = 0.0
            for sent in dev:
                words, golds = zip(*sent)
                tags = [vt.i2w[i] for i in model(words, volatile=True).max(1)[1].cpu().data.view(-1)]
                if tags == list(golds): good_sent += 1
                else: bad_sent += 1
                for go, gu in zip(golds, tags):
                    if go == gu: good += 1
                    else: bad += 1
            dev_time += time.time() - dev_start
            train_time = time.time() - start - dev_time
            print ("tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f" % (good/(good+bad), good_sent/(good_sent+bad_sent), train_time, all_tagged/train_time))
            if all_time > args.TIMEOUT:
                sys.exit(0)
        # batch / loss
        words, golds = zip(*s)
        preds = model(words)
        loss = F.cross_entropy(preds, get_var(torch.LongTensor([vt.w2i[t] for t in golds])))
        # log / optim
        this_loss += loss.data[0]
        this_tagged += len(golds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch %r finished" % ITER)
