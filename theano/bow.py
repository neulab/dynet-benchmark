from __future__ import division, print_function
import time
start = time.time()

import theano.tensor as T
import theano
import numpy as np
import sys
import random

from nn.optimizers import SGD, Adam
from nn.initializations import uniform, zero

from collections import defaultdict


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
  with open(filename, "r") as f:
  for line in f:
    tag, words = line.lower().strip().split(" ||| ")
    yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train = list(read_dataset("data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)


# Define the model
W_sm = zero((nwords, ntags))  # Word weights
b_sm = zero((ntags))      # Softmax bias

# bag of words input
x = T.ivector('words')
# gold class
y = T.iscalar('class')

score = T.sum(W_sm[x], axis=0) + b_sm
# log likelihood
ll = T.log(T.nnet.softmax(score)).flatten()
# negative log likelihood loss
loss = - ll[y]

params = [W_sm, b_sm]
updates = Adam(lr=0.001).get_updates(params, loss)

train_func = theano.function([x, y], loss, updates=updates)
test_func = theano.function([x], score)

print ("startup time: %r" % (time.time() - start))
for ITER in range(100):
  # Perform training
  random.shuffle(train)
  train_loss = 0.0
  start = time.time()
  for i, (words, tag) in enumerate(train):
    my_loss = train_func(words, tag)
    train_loss += my_loss
    # print(b_sm.get_value())
    # if i > 5:
    #   sys.exit(0)

  print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss/len(train), time.time()-start))

  # Perform testing
  test_correct = 0.0
  for words, tag in dev:
    scores = test_func(words)
    predict = np.argmax(scores)
    if predict == tag:
      test_correct += 1

  print("iter %r: test acc=%.4f" % (ITER, test_correct/len(dev)))



