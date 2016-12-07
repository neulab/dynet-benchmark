from collections import Counter, defaultdict
from itertools import count
import random
import time
import math
import sys

from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

if len(sys.argv) != 2:
  print("usage: %s (GPU-ID or -1 to use CPU)" % sys.argv[0])
  sys.exit(1)

GPUID = int(sys.argv[1])

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file="data/text/train.txt"
test_file="data/text/dev.txt"

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

lm = RNNLM()

if GPUID >= 0:
  # use GPU
  from chainer.cuda import cupy as xp, get_device
  get_device(GPUID).use()
  lm.to_gpu()
else:
  # use CPU
  import numpy as xp

def makevar(arr):
  return Variable(xp.array(arr, dtype=xp.int32))

init_alpha = 0.001
trainer = O.Adam(init_alpha)
trainer.use_cleargrads()
trainer.setup(lm)

# Build the language model graph
def calc_lm_loss(sent):
  # initialize the RNN
  lm.reset()

  # start the rnn by inputting "<s>"
  y = lm.add_input(makevar([sent[-1]]))

  # feed word vectors into the RNN and predict the next word
  losses = []
  for wid in sent:
    # calculate the softmax and loss
    t = makevar([wid])
    loss = F.softmax_cross_entropy(y, t)
    losses.append(loss)
    # update the state of the RNN
    y = lm.add_input(t)

  return sum(losses)

start = time.time()
i = all_time = all_tagged = this_words = this_loss = 0
for ITER in xrange(50):
  random.shuffle(train)
  trainer.alpha = init_alpha / (1.0 + ITER)
  for s in train:
    i += 1
    if i % 500 == 0:
      print(this_loss / this_words)
      all_tagged += this_words
      this_loss = this_words = 0
    if i % 5000 == 0:
      all_time += time.time() - start
      dev_loss = dev_words = 0
      for sent in test:
        loss_exp = calc_lm_loss(sent)
        dev_loss += float(loss_exp.data)
        dev_words += len(sent)
      print("nll=%.4f, ppl=%.4f, time=%.4f, word_per_sec=%.4f" % (dev_loss/dev_words, math.exp(dev_loss/dev_words), all_time, all_tagged/all_time))
      if all_time > 300:
        sys.exit(0)
      start = time.time()
    # train on sent
    loss_exp = calc_lm_loss(s)
    this_loss += float(loss_exp.data)
    this_words += len(s)
    lm.cleargrads()
    loss_exp.backward()
    trainer.update()
