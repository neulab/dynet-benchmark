from __future__ import division, print_function
import time
start = time.time()

import random

import theano.tensor as T
import theano
import numpy as np
import sys
import argparse
from itertools import chain

from nn.layers.recurrent import LSTM, BiLSTM
from nn.layers.embeddings import Embedding
from nn.activations import softmax
from nn.optimizers import Adam
from nn.initializations import uniform

from collections import Counter, defaultdict
from itertools import count

parser = argparse.ArgumentParser()
parser.add_argument('WEMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('MLP_SIZE', type=int, help='embedding size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

MB_SIZE = 1

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

  def size(self):
    return len(self.w2i.keys())


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
wc=Counter()
words.append('_MASK_')
for sent in train:
  for w,p in sent:
    words.append(w)
    tags.append(p)
    wc[w]+=1
words.append("_UNK_")

vw = Vocab.from_corpus([words])
vt = Vocab.from_corpus([tags])
UNK = vw.w2i["_UNK_"]

# mask token must be of index 0
assert vw.w2i['_MASK_'] == 0

nwords = vw.size()
ntags  = vt.size()

print "nwords=%r, ntags=%r" % (nwords, ntags)


def word2id(w):
  if wc[w] > 5:
    w_index = vw.w2i[w]
    return w_index
  else:
    return UNK


def pad(seq):
  """
  pad a mini-batch input with ending zeros
  """
  batch_size = len(seq)
  max_len = max(len(seq[i]) for i in xrange(batch_size))
  padded_seq = np.zeros((batch_size, max_len), dtype='int32')
  for i in xrange(batch_size):
    padded_seq[i, :len(seq[i])] = seq[i]

  return padded_seq


def build_tag_graph():
  print >> sys.stderr, 'build graph..'

  # (batch_size, sentence_length)
  x = T.imatrix(name='sentence')

  # (batch_size, sentence_length)
  y = T.imatrix(name='tag')

  # Lookup parameters for word embeddings
  embedding_table = Embedding(nwords, args.WEMBED_SIZE)

  # bi-lstm
  lstm = BiLSTM(args.WEMBED_SIZE, args.HIDDEN_SIZE, return_sequences=True)

  # MLP
  W_mlp_hidden = uniform((args.HIDDEN_SIZE * 2, args.MLP_SIZE), name='W_mlp_hidden')
  W_mlp = uniform((args.MLP_SIZE, ntags), name='W_mlp')

  # (batch_size, sentence_length, embedding_dim)
  sent_embed, sent_mask = embedding_table(x, mask_zero=True)

  # (batch_size, sentence_length, lstm_hidden_dim)
  lstm_output = lstm(sent_embed, mask=sent_mask)

  # (batch_size, sentence_length, ntags)
  mlp_output = T.dot(T.tanh(T.dot(lstm_output, W_mlp_hidden)), W_mlp)

  # (batch_size * sentence_length, ntags)
  mlp_output = mlp_output.reshape((mlp_output.shape[0] * mlp_output.shape[1], -1))

  tag_prob_f = T.log(T.nnet.softmax(mlp_output))

  y_f = y.flatten()
  mask_f = sent_mask.flatten()

  tag_nll = - tag_prob_f[T.arange(tag_prob_f.shape[0]), y_f] * mask_f

  loss = tag_nll.sum()

  params = embedding_table.params + lstm.params + [W_mlp_hidden, W_mlp]
  updates = Adam().get_updates(params, loss)
  train_loss_func = theano.function([x, y], loss, updates=updates)

  # build the decoding graph
  tag_prob = tag_prob_f.reshape((x.shape[0], x.shape[1], -1))
  decode_func = theano.function([x], tag_prob)

  return train_loss_func, decode_func


def data2ids(batch_data):
  batch_sent_ids = [[word2id(w) for w, t in sent] for sent in batch_data]
  batch_tag_ids = [[vt.w2i[t] for w, t in sent] for sent in batch_data]

  return batch_sent_ids, batch_tag_ids


def tag_sent(batch_sents, decode_func):
  batch_sent_ids = [[word2id(w) for w in sent] for sent in batch_sents]
  batch_sents_x = pad(batch_sent_ids)
  batch_sents_len = [len(sent) for sent in batch_sents]

  # (batch_size, sentence_length, tag_num)
  tag_prob = decode_func(batch_sents_x)
  batch_tag_results = []

  for i, sent in enumerate(batch_sents):
    sent_len = batch_sents_len[i]
    tag_results = tag_prob[i].argmax(axis=-1)[:sent_len]
    tag_results = [vt.i2w[tid] for tid in tag_results]
    batch_tag_results.append(tag_results)

  return batch_tag_results

train_func, decode_func = build_tag_graph()

batch_num = int(np.ceil(len(train) / float(MB_BATCH)))
batches = [(i * MB_BATCH, min(len(train), (i + 1) * MB_BATCH)) for i in range(0, batch_num)]

print ("startup time: %r" % (time.time() - start))
start = time.time()
i = all_time = all_tagged = this_tagged = this_loss = 0

for ITER in xrange(50):
  random.shuffle(train)
  for batch_id, (batch_start, batch_end) in enumerate(batches):
    i += MB_BATCH

    if i % 500 == 0:  # print status
      print this_loss / this_tagged
      all_tagged += this_tagged
      this_loss = this_tagged = 0

    if i % 10000 == 0:  # eval on dev
      all_time += time.time() - start
      good_sent = bad_sent = good = bad = 0.0
      for sent in dev:
        words = [w for w, t in sent]
        golds = [t for w, t in sent]

        # package words in a batch
        tags = tag_sent([words], decode_func)
        tags = tags[0]

        if tags == golds:
          good_sent += 1
        else:
          bad_sent += 1
        for go, gu in zip(golds, tags):
          if go == gu:
            good += 1
          else:
            bad += 1

      print ("tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f" % (
           good / (good + bad), good_sent / (good_sent + bad_sent), all_time, all_tagged / all_time))

      if all_time > args.TIMEOUT:
        sys.exit(0)
      start = time.time()

    # train on training sentences

    batch_data = train[batch_start:batch_end]
    batch_sent_ids, batch_tag_ids = data2ids(batch_data)

    batch_x = pad(batch_sent_ids)
    batch_y = pad(batch_tag_ids)

    batch_loss = train_func(batch_x, batch_y)

    this_loss += batch_loss
    this_tagged += len(list(chain(*batch_data)))

  print "epoch %r finished" % ITER
