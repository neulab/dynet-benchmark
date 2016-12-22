from __future__ import print_function
import time
start = time.time()

from collections import Counter, defaultdict
import random
import math
import sys
import argparse

import dynet as dy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dynet_mem", default=512, type=int)
parser.add_argument('MB_SIZE', type=int, help='minibatch size')
parser.add_argument('EMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file="data/text/train.txt"
test_file="data/text/dev.txt"

w2i = defaultdict(lambda: len(w2i))

def read(fname):
  """
  Read a file where each line is of the form "word1 word2 ..."
  Yields lists of the form [word1, word2, ...]
  """
  with open(fname, "r") as fh:
    for line in fh:
      sent = [w2i[x] for x in line.strip().split()]
      sent.append(w2i["<s>"])
      yield sent

train=list(read(train_file))
nwords = len(w2i)
test=list(read(test_file))
S = w2i["<s>"]
assert(nwords == len(w2i))

# DyNet Starts

model = dy.Model()
trainer = dy.AdamTrainer(model)
trainer.set_clip_threshold(-1.0)
trainer.set_sparse_updates(True if args.SPARSE == 1 else False)

# Lookup parameters for word embeddings
WORDS_LOOKUP = model.add_lookup_parameters((nwords, args.EMBED_SIZE))

# Word-level LSTM (layers=1, input=64, output=128, model)
RNN = dy.VanillaLSTMBuilder(1, args.EMBED_SIZE, args.HIDDEN_SIZE, model)

# Softmax weights/biases on top of LSTM outputs
W_sm = model.add_parameters((nwords, args.HIDDEN_SIZE))
b_sm = model.add_parameters(nwords)

# Build the language model graph
def calc_lm_loss(sents):

  dy.renew_cg()
  # parameters -> expressions
  W_exp = dy.parameter(W_sm)
  b_exp = dy.parameter(b_sm)

  # initialize the RNN
  f_init = RNN.initial_state()

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
  s = f_init.add_input(dy.lookup_batch(WORDS_LOOKUP,init_ids))

  # feed word vectors into the RNN and predict the next word
  losses = []
  for wid, mask in zip(wids, masks):
    # calculate the softmax and loss
    score = dy.affine_transform([b_exp, W_exp, s.output()])
    loss = dy.pickneglogsoftmax_batch(score, wid)
    # mask the loss if at least one sentence is shorter
    if mask[-1] != 1:
      mask_expr = dy.inputVector(mask)
      mask_expr = dy.reshape(mask_expr, (1,), args.MB_SIZE)
      loss = loss * mask_expr
    losses.append(loss)
    # update the state of the RNN    
    wemb = dy.lookup_batch(WORDS_LOOKUP, wid)
    s = s.add_input(wemb) 
  
  return dy.sum_batches(dy.esum(losses)), tot_words

# Sort training sentences in descending order and count minibatches
train.sort(key=lambda x: -len(x))
test.sort(key=lambda x: -len(x))
train_order = [x*args.MB_SIZE for x in range(int((len(train)-1)/args.MB_SIZE + 1))]
test_order = [x*args.MB_SIZE for x in range(int((len(test)-1)/args.MB_SIZE + 1))]

print ("startup time: %r" % (time.time() - start))
# Perform training
start = time.time()
i = all_time = dev_time = all_tagged = this_words = this_loss = 0
for ITER in range(10):
  random.shuffle(train_order)
  for sid in train_order: 
    i += 1
    if i % int(500/args.MB_SIZE) == 0:
      trainer.status()
      print (this_loss / this_words, sys.stderr)
      all_tagged += this_words
      this_loss = this_words = 0
      all_time = time.time() - start
    if i % int(10000 / args.MB_SIZE) == 0 or all_time > args.TIMEOUT:
      dev_start = time.time()
      dev_loss = dev_words = 0
      for sid in test_order:
        loss_exp, mb_words = calc_lm_loss(test[sid:sid+args.MB_SIZE])
        dev_loss += loss_exp.scalar_value()
        dev_words += mb_words
      dev_time += time.time() - dev_start 
      train_time = time.time() - start - dev_time
      print ("nll=%.4f, ppl=%.4f, words=%r, time=%.4f, word_per_sec=%.4f" % (dev_loss/dev_words, math.exp(dev_loss/dev_words), dev_words, train_time, all_tagged/train_time))
      if train_time > args.TIMEOUT:
        sys.exit(0)
    # train on the minibatch
    loss_exp, mb_words = calc_lm_loss(train[sid:sid+args.MB_SIZE])
    this_loss += loss_exp.scalar_value()
    # print("loss @ %r: %r" % (i, this_loss))
    this_words += mb_words
    loss_exp.backward()
    trainer.update()
  print ("epoch %r finished" % ITER)
  trainer.update_epoch(1.0)
