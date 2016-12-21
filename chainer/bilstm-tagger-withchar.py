from collections import Counter, defaultdict
from itertools import count
import random
import time
import sys

from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

if len(sys.argv) != 2:
  print("usage: %s (GPU-ID or -1 to use CPU)" % sys.argv[0])
  sys.exit(1)

GPUID = int(sys.argv[1])

if GPUID >= 0:
  # use GPU
  from chainer.cuda import cupy as xp, get_device
  get_device(GPUID).use()
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
print ("nwords=%r, ntags=%r, nchars=%r" % (nwords, ntags, nchars))

# Chainer Starts

class Tagger(Chain):
  def __init__(self):
    super(Tagger, self).__init__(
        embedW=L.EmbedID(nwords, 128),
        embedC=L.EmbedID(nwords, 20),
        # MLP on top of biLSTM outputs 100 -> 32 -> ntags
        WH=L.Linear(50*2, 32, nobias=True),
        WO=L.Linear(32, ntags, nobias=True),
        # word-level LSTMs
        fwdRNN=L.LSTM(128, 50),
        bwdRNN=L.LSTM(128, 50),
        # char-level LSTMs,
        cFwdRNN=L.LSTM(20, 64),
        cBwdRNN=L.LSTM(20, 64),
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

start = time.time()
i = all_time = all_tagged = this_tagged = this_loss = 0
for ITER in xrange(10):
  random.shuffle(train)
  for s in train:
    i += 1
    if i % 500 == 0:   # print status
      print this_loss / this_tagged
      all_tagged += this_tagged
      this_loss = this_tagged = 0
    if i % 10000 == 0: # eval on dev
      all_time += time.time() - start
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
      print ("tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f" % (good/(good+bad), good_sent/(good_sent+bad_sent), all_time, all_tagged/all_time))
      if all_time > 3600:
        sys.exit(0)
      start = time.time()
    # train on sent
    words = [w for w, _ in s]
    golds = [t for _, t in s]

    loss_exp = tagger.sent_loss(words, golds)
    this_loss += float(loss_exp.data)
    this_tagged += len(golds)
    tagger.cleargrads()
    loss_exp.backward()
    trainer.update()

  print "epoch %r finished" % ITER
  trainer.update_epoch(1.0)
