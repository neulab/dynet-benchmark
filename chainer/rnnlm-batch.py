from collections import Counter, defaultdict
from itertools import count
import random
import time
import math
import sys

from chainer import Chain, Variable
from chainer.cuda import cupy, get_device
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
import numpy as np

if len(sys.argv) != 2:
  print("usage: %s (GPU-ID or -1 to use CPU)" % sys.argv[0])
  sys.exit(1)

GPUID = int(sys.argv[1])

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file="data/text/train.txt"
test_file="data/text/dev.txt"

MB_SIZE = 10

w2i = defaultdict(count(0).next)

def read(fname):
  """
  Read a file where each line is of the form "word1 word2 ..."
  Yields lists of the form [word1, word2, ...]
  """
  with file(fname) as fh:
    for line in fh:
      sent = [w2i[x] for x in line.strip().split()]
      sent.append(w2i["<s>"])
      yield sent

train=list(read(train_file))
nwords = len(w2i)
test=list(read(test_file))
S = w2i["<s>"]
assert(nwords == len(w2i))

# Chainer Starts

class RNNLM(Chain):
  def __init__(self):
    super(RNNLM, self).__init__(
        embed=L.EmbedID(nwords, 64),
        rnn=L.LSTM(64, 128),
        h2y=L.Linear(128, nwords),
    )

  def reset(self):
    self.rnn.reset_state()

  def add_input(self, x):
    h = self.rnn(self.embed(x))
    return self.h2y(h)

def makevar(arr):
  xp = cupy if GPUID >= 0 else np
  return Variable(xp.array(arr, dtype=xp.int32))

lm = RNNLM()
if GPUID >= 0:
  get_device(GPUID).use()
  lm.to_gpu()

init_alpha = 0.001
trainer = O.Adam(init_alpha)
trainer.use_cleargrads()
trainer.setup(lm)

# Build the language model graph
#
# Note: Chainer could not consider masking using default cross entropy function
#       which returns an integrated scalar.
#
def calc_lm_loss(sents):
  # initialize the RNN
  lm.reset()

  # get the wids and masks for each step
  tot_words = 0
  wids = []
  masks = []
  for i in range(len(sents[0])):
    wids.append([
      (sent[i] if len(sent)>i else S) for sent in sents])
    mask = [(1 if len(sent)>i else 0) for sent in sents]
    masks.append(mask)
    tot_words += sum(mask)
    
  # start the rnn by inputting "<s>"
  init_ids = [S] * len(sents)
  y = lm.add_input(makevar(init_ids))

  # feed word vectors into the RNN and predict the next word
  losses = []
  for wid, mask in zip(wids, masks):
    # calculate the softmax and loss
    t = makevar(wid)
    loss = F.softmax_cross_entropy(y, t) * len(sents)
    #
    # TODO: Implementing masking
    #
    losses.append(loss)
    # update the state of the RNN        
    y = lm.add_input(t)
  
  return sum(losses), tot_words

i = all_time = all_tagged = this_words = this_loss = 0
# Sort training sentences in descending order and count minibatches
train.sort(key=lambda x: -len(x))
test.sort(key=lambda x: -len(x))
train_order = [x*MB_SIZE for x in range((len(train)-1)/MB_SIZE + 1)]
test_order = [x*MB_SIZE for x in range((len(test)-1)/MB_SIZE + 1)]
# Perform training
start = time.time()
for ITER in xrange(10):
  random.shuffle(train_order)
  trainer.alpha = init_alpha / (1.0 + ITER)
  for sid in train_order: 
    i += 1
    if i % (500/MB_SIZE) == 0:
      print(this_loss / this_words)
      all_tagged += this_words
      this_loss = this_words = 0
    if i % (10000/MB_SIZE) == 0:
      all_time += time.time() - start
      dev_loss = dev_words = 0
      for sid in test_order:
        loss_exp, mb_words = calc_lm_loss(test[sid:sid+MB_SIZE])
        dev_loss += float(loss_exp.data)
        dev_words += mb_words
      print("nll=%.4f, ppl=%.4f, words=%r, time=%.4f, word_per_sec=%.4f" % (dev_loss/dev_words, math.exp(dev_loss/dev_words), dev_words, all_time, all_tagged/all_time))
      if all_time > 3600:
        sys.exit(0)
      start = time.time()
    # train on the minibatch
    loss_exp, mb_words = calc_lm_loss(train[sid:sid+MB_SIZE])
    this_loss += float(loss_exp.data)
    this_words += mb_words
    lm.cleargrads()
    loss_exp.backward()
    trainer.update()
  print "epoch %r finished" % ITER


