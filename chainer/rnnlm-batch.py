import time
start = time.time()

from collections import Counter, defaultdict
from itertools import count
import random
import math
import sys

from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

if len(sys.argv) != 5:
  print("Usage: %s MB_SIZE EMBED_SIZE HIDDEN_SIZE TIMEOUT" % sys.argv[0])
  sys.exit(1)
MB_SIZE = int(sys.argv[1])
EMBED_SIZE = int(sys.argv[2])
HIDDEN_SIZE = int(sys.argv[3])
TIMEOUT = int(sys.argv[4]) 

GPUID = -1

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
        embed=L.EmbedID(nwords, EMBED_SIZE),
        rnn=L.LSTM(EMBED_SIZE, HIDDEN_SIZE),
        h2y=L.Linear(HIDDEN_SIZE, nwords),
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

trainer = O.Adam()
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

  # get the wids for each step
  tot_words = 0
  wids = []
  for i in range(len(sents[0])):
    # Note: -1 is the default padding tag in Chainer.
    wids.append([
      (sent[i] if len(sent)>i else -1) for sent in sents])
    mask = [(1 if len(sent)>i else 0) for sent in sents]
    tot_words += sum(mask)

  # start the rnn by inputting "<s>"
  init_ids = [S] * len(sents)
  y = lm.add_input(makevar(init_ids))

  # feed word vectors into the RNN and predict the next word
  losses = []
  for wid in wids:
    # calculate the softmax and loss
    t = makevar(wid)
    # Note: Chainer calculates the average. We have to multiply the batch size
    #       to adjust dynamic range of the loss.
    loss = F.softmax_cross_entropy(y, t, normalize=False) * len(sents)
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
print ("startup time: %r" % (time.time() - start))
start = time.time()
for ITER in xrange(10):
  random.shuffle(train_order)
  for sid in train_order:
    i += 1
    if i % int(500/MB_SIZE) == 0:
      print(this_loss / this_words)
      all_tagged += this_words
      this_loss = this_words = 0
    if i % int(10000/MB_SIZE) == 0:
      all_time += time.time() - start
      dev_loss = dev_words = 0
      for sid in test_order:
        loss_exp, mb_words = calc_lm_loss(test[sid:sid+MB_SIZE])
        dev_loss += float(loss_exp.data)
        dev_words += mb_words
      print("nll=%.4f, ppl=%.4f, words=%r, time=%.4f, word_per_sec=%.4f" % (dev_loss/dev_words, math.exp(dev_loss/dev_words), dev_words, all_time, all_tagged/all_time))
      if all_time > TIMEOUT:
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


