from __future__ import print_function
import time
start = time.time()

from collections import Counter, defaultdict
import random
import sys
import argparse

import dynet as dy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dynet_seed", default=0, type=int)
parser.add_argument("--dynet_mem", default=512, type=int)
parser.add_argument('WEMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('MLP_SIZE', type=int, help='embedding size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

# format of files: each line is "word1|tag2 word2|tag2 ..."
train_file="data/tags/train.txt"
dev_file="data/tags/dev.txt"

class Vocab:
  def __init__(self, w2i=None):
    if w2i is None: w2i = defaultdict(lambda: len(w2i))
    self.w2i = dict(w2i)
    self.i2w = {i:w for w,i in w2i.items()}
  @classmethod
  def from_corpus(cls, corpus):
    w2i = defaultdict(lambda: len(w2i))
    for sent in corpus:
      [w2i[word] for word in sent]
    return Vocab(w2i)

  def size(self): return len(self.w2i.keys())

def read(fname):
  """
  Read a POS-tagged file where each line is of the form "word1|tag2 word2|tag2 ..."
  Yields lists of the form [(word1,tag1), (word2,tag2), ...]
  """
  with open(fname, "r") as fh:
    for line in fh:
      line = line.strip().split()
      sent = [tuple(x.rsplit("|",1)) for x in line]
      yield sent

train=list(read(train_file))
dev=list(read(dev_file))
words=[]
tags=[]
wc=Counter()
for sent in train:
  for w,p in sent:
    words.append(w)
    tags.append(p)
    wc[w]+=1
words.append("_UNK_")

vw = Vocab.from_corpus([words])
vt = Vocab.from_corpus([tags])
UNK = vw.w2i["_UNK_"]

nwords = vw.size()
ntags  = vt.size()
print ("nwords=%r, ntags=%r" % (nwords, ntags))

# DyNet Starts

model = dy.Model()
trainer = dy.AdamTrainer(model)
trainer.set_clip_threshold(-1.0)
trainer.set_sparse_updates(True if args.SPARSE == 1 else False)

WORDS_LOOKUP = model.add_lookup_parameters((nwords, args.WEMBED_SIZE))

# MLP on top of biLSTM outputs 100 -> 32 -> ntags
pH = model.add_parameters((args.MLP_SIZE, args.HIDDEN_SIZE*2))
pO = model.add_parameters((ntags, args.MLP_SIZE))

# word-level LSTMs
fwdRNN = dy.VanillaLSTMBuilder(1, args.WEMBED_SIZE, args.HIDDEN_SIZE, model) # layers, in-dim, out-dim, model
bwdRNN = dy.VanillaLSTMBuilder(1, args.WEMBED_SIZE, args.HIDDEN_SIZE, model)

def word_rep(w):
  widx = vw.w2i[w] if wc[w] > 5 else UNK
  return WORDS_LOOKUP[widx]

def build_tagging_graph(words):
  dy.renew_cg()
  # parameters -> expressions
  H = dy.parameter(pH)
  O = dy.parameter(pO)

  # initialize the RNNs
  f_init = fwdRNN.initial_state()
  b_init = bwdRNN.initial_state()

  # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
  wembs = []
  for i, w in enumerate(words):
    wembs.append(word_rep(w))

  # feed word vectors into biLSTM
  fw_exps = f_init.transduce(wembs)
  bw_exps = b_init.transduce(reversed(wembs))

  # biLSTM states
  bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

  # feed each biLSTM state to an MLP
  exps = []
  for x in bi_exps:
    r_t = O*(dy.tanh(H * x))
    exps.append(r_t)

  return exps

def sent_loss_precalc(words, tags, vecs):
  errs = []
  for v,t in zip(vecs,tags):
    tid = vt.w2i[t]
    err = dy.pickneglogsoftmax(v, tid)
    errs.append(err)
  return dy.esum(errs)

def sent_loss(words, tags):
  return sent_loss_precalc(words, tags, build_tagging_graph(words))

def tag_sent_precalc(words, vecs):
  log_probs = [v.npvalue() for v in vecs]
  tags = []
  for prb in log_probs:
    tag = np.argmax(prb)
    tags.append(vt.i2w[tag])
  return zip(words, tags)

def tag_sent(words):
  return tag_sent_precalc(words, build_tagging_graph(words))

print ("startup time: %r" % (time.time() - start))
start = time.time()
i = all_time = dev_time = all_tagged = this_tagged = this_loss = 0
for ITER in range(50):
  random.shuffle(train)
  for s in train:
    i += 1
    if i % 500 == 0:   # print status
      trainer.status()
      print(this_loss / this_tagged, file=sys.stderr)
      all_tagged += this_tagged
      this_loss = this_tagged = 0
      all_time = time.time() - start
    if i % 10000 == 0 or all_time > args.TIMEOUT: # eval on dev
      dev_start = time.time()
      good_sent = bad_sent = good = bad = 0.0
      for sent in dev:
        words = [w for w,t in sent]
        golds = [t for w,t in sent]
        tags = [t for w,t in tag_sent(words)]
        if tags == golds: good_sent += 1
        else: bad_sent += 1
        for go,gu in zip(golds,tags):
          if go == gu: good += 1
          else: bad += 1
      dev_time += time.time() - dev_start 
      train_time = time.time() - start - dev_time
      print ("tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f" % (good/(good+bad), good_sent/(good_sent+bad_sent), train_time, all_tagged/train_time))
      if train_time > args.TIMEOUT:
        sys.exit(0)
    # train on sent
    words = [w for w,t in s]
    golds = [t for w,t in s]

    loss_exp = sent_loss(words, golds)
    my_loss = loss_exp.scalar_value()
    this_loss += my_loss;
    this_tagged += len(golds)
    loss_exp.backward()
    trainer.update()
  print("epoch %r finished" % ITER)
  trainer.update_epoch(1.0)
