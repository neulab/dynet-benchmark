from __future__ import print_function
import time
start = time.time()

from collections import Counter, defaultdict
from itertools import count
import random
import sys
import argparse

from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

parser = argparse.ArgumentParser()
parser.add_argument('--chainer_gpu', type=int, default=-1, help='GPU id')
parser.add_argument('CEMBED_SIZE', type=int, help='char embedding size')
parser.add_argument('WEMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('MLP_SIZE', type=int, help='embedding size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

if args.chainer_gpu >= 0:
  # use GPU
  from chainer.cuda import cupy as xp, get_device
  get_device(args.chainer_gpu).use()
else:
  # use CPU
  import numpy as xp

def makevar(x):
  return Variable(xp.array([x], dtype=xp.int32))

# format of files: each line is "word1|tag2 word2|tag2 ..."
train_file="data/tags/train.txt"
dev_file="data/tags/dev.txt"

class Vocab:
  def __init__(self, w2i=None):
    if w2i is None: w2i = defaultdict(count(0).next)
    self.w2i = dict(w2i)
    self.i2w = {i:w for w,i in w2i.iteritems()}
  @classmethod
  def from_corpus(cls, corpus):
    w2i = defaultdict(count(0).next)
    for sent in corpus:
      [w2i[word] for word in sent]
    return Vocab(w2i)

  def size(self): return len(self.w2i.keys())

def read(fname):
  """
  Read a POS-tagged file where each line is of the form "word1|tag2 word2|tag2 ..."
  Yields lists of the form [(word1,tag1), (word2,tag2), ...]
  """
  with file(fname) as fh:
    for line in fh:
      line = line.strip().split()
      sent = [tuple(x.rsplit("|",1)) for x in line]
      yield sent

train=list(read(train_file))
dev=list(read(dev_file))
words=[]
tags=[]
chars=set()
wc=Counter()
for sent in train:
  for w,p in sent:
    words.append(w)
    tags.append(p)
    chars.update(w)
    wc[w]+=1
words.append("_UNK_")
chars.add("<*>")

vw = Vocab.from_corpus([words])
vt = Vocab.from_corpus([tags])
vc = Vocab.from_corpus([chars])
UNK = vw.w2i["_UNK_"]

nwords = vw.size()
ntags  = vt.size()
nchars  = vc.size()
print("nwords=%r, ntags=%r, nchars=%r" % (nwords, ntags, nchars))

# Chainer Starts

class Tagger(Chain):
  def __init__(self):
    super(Tagger, self).__init__(
        embedW=L.EmbedID(nwords, args.WEMBED_SIZE),
        embedC=L.EmbedID(nwords, args.CEMBED_SIZE),
        # MLP on top of biLSTM outputs 100 -> 32 -> ntags
        WH=L.Linear(args.HIDDEN_SIZE*2, args.MLP_SIZE, nobias=True),
        WO=L.Linear(args.MLP_SIZE, ntags, nobias=True),
        # word-level LSTMs
        fwdRNN=L.LSTM(args.WEMBED_SIZE, args.HIDDEN_SIZE),
        bwdRNN=L.LSTM(args.WEMBED_SIZE, args.HIDDEN_SIZE),
        # char-level LSTMs,
        cFwdRNN=L.LSTM(args.CEMBED_SIZE, args.WEMBED_SIZE/2),
        cBwdRNN=L.LSTM(args.CEMBED_SIZE, args.WEMBED_SIZE/2),
    )

  def word_rep(self, w):
    if wc[w] > 5:
      return self.embedW(makevar(vw.w2i[w]))
    else:
      pad_char = vc.w2i["<*>"]
      char_ids = [pad_char] + [vc.w2i[c] for c in w] + [pad_char]
      char_embs = [self.embedC(makevar(cid)) for cid in char_ids]
      self.cFwdRNN.reset_state()
      self.cBwdRNN.reset_state()
      for e in char_embs:
        fw_exp = self.cFwdRNN(e)
      for e in reversed(char_embs):
        bw_exp = self.cBwdRNN(e)
      return F.concat([fw_exp, bw_exp])

  def build_tagging_graph(self, words):
    # initialize the RNNs
    self.fwdRNN.reset_state()
    self.bwdRNN.reset_state()

    # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    wembs = [self.word_rep(w) for w in words]

    # feed word vectors into biLSTM
    fw_exps = []
    for e in wembs:
      fw_exps.append(self.fwdRNN(e))
    bw_exps = []
    for e in reversed(wembs):
      bw_exps.append(self.bwdRNN(e))

    # biLSTM states
    bi_exps = [F.concat([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]

    # feed each biLSTM state to an MLP
    exps = [self.WO(F.tanh(self.WH(x))) for x in bi_exps]
    return exps

  def sent_loss(self, words, tags):
    vecs = self.build_tagging_graph(words)
    return sum(F.softmax_cross_entropy(v, makevar(vt.w2i[t])) for v, t in zip(vecs, tags))

  def tag_sent(self, words):
    vecs = self.build_tagging_graph(words)
    tags = [vt.i2w[v.data.argmax()] for v in vecs]
    return zip(words, tags)

tagger = Tagger()
trainer = O.Adam()
trainer.use_cleargrads()
trainer.setup(tagger)

print("startup time: %r" % (time.time() - start))
start = time.time()
i = all_time = dev_time = all_tagged = this_tagged = this_loss = 0
for ITER in xrange(100):
  random.shuffle(train)
  for s in train:
    i += 1
    if i % 500 == 0:   # print status
      print(this_loss / this_tagged)
      all_tagged += this_tagged
      this_loss = this_tagged = 0
      all_time = time.time() - start
    if i % 10000 == 0 or all_time > args.TIMEOUT: # eval on dev
      dev_start = time.time()
      good_sent = bad_sent = good = bad = 0.0
      for sent in dev:
        words = [w for w, _ in sent]
        golds = [t for _, t in sent]
        tags = [t for _, t in tagger.tag_sent(words)]
        if tags == golds:
          good_sent += 1
        else:
          bad_sent += 1
        for go,gu in zip(golds,tags):
          if go == gu:
            good += 1
          else:
            bad += 1
      dev_time += time.time() - dev_start 
      train_time = time.time() - start - dev_time
      print("tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f" % (good/(good+bad), good_sent/(good_sent+bad_sent), train_time, all_tagged/train_time))
      if all_time > args.TIMEOUT:
        sys.exit(0)
    # train on sent
    words = [w for w, _ in s]
    golds = [t for _, t in s]

    loss_exp = tagger.sent_loss(words, golds)
    this_loss += float(loss_exp.data)
    this_tagged += len(golds)
    tagger.cleargrads()
    loss_exp.backward()
    trainer.update()

  print("epoch %r finished" % ITER)
  trainer.update_epoch(1.0)
