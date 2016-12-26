from __future__ import print_function
import time
start = time.time()

from collections import Counter, defaultdict
from itertools import count
import random
import math
import sys
import argparse

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('MB_SIZE', type=int, help='minibatch size')
parser.add_argument('EMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')  # sparse updates by default in tensorflow
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

NUM_LAYERS = 1
GPU = False

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file='data/text/train.txt'
test_file='data/text/dev.txt'
w2i = defaultdict(count(0).next)
eos = '<s>'

def read(fname):
  """
  Read a file where each line is of the form "word1 word2 ..."
  Yields lists of the form [word1, word2, ...]
  """
  with file(fname) as fh:
    for line in fh:
      sent = [w2i[eos]]
      sent += [w2i[x] for x in line.strip().split()]
      sent.append(w2i[eos])
      yield sent

train = list(read(train_file))
nwords = len(w2i)
test = list(read(test_file))
S = w2i[eos]
assert(nwords == len(w2i))

train.sort(key=lambda x: -len(x))
test.sort(key=lambda x: -len(x))

if args.MB_SIZE != 0:
  train_order = [x*args.MB_SIZE for x in range((len(train)-1)/args.MB_SIZE + 1)]
  test_order = [x*args.MB_SIZE for x in range((len(test)-1)/args.MB_SIZE + 1)]
else:
  train_order = range(len(train))
  test_order = range(len(test))

#max_length = len(max(train, key=len))
#assert len(max(test, key=len)) < max_length, 'There should be no test sentences longer than the longest training sentence (%d words)' % max_length

def pad(seq, element, length):
  assert len(seq) <= length
  r = seq + [element] * (length - len(seq))
  assert len(r) == length
  return r 

if GPU:
  cpu_or_gpu = '/gpu:0'
else:
  cpu_or_gpu = '/cpu:0'

with tf.device(cpu_or_gpu):
  # Lookup parameters for word embeddings
  WORDS_LOOKUP = tf.Variable(tf.random_uniform([nwords, 1, args.EMBED_SIZE], -1.0, 1.0))

  # Word-level LSTM (configurable number of layers, input is unspecified,
  # but will be equal to the embedding dim, output=128)
  cell = tf.nn.rnn_cell.BasicLSTMCell(args.HIDDEN_SIZE) 
  cell = tf.nn.rnn_cell.MultiRNNCell([cell] * NUM_LAYERS, state_is_tuple=True)

  # input sentence placeholder
  x_input = tf.placeholder(tf.int32, [None, None], name="x_input")
  x_lens = tf.placeholder(tf.int32, [None], name='x_lens')

  x_embs = tf.squeeze(tf.nn.embedding_lookup(WORDS_LOOKUP, x_input))
  x_embs.set_shape([None, None, args.EMBED_SIZE])
  cell_out = tf.nn.rnn_cell.OutputProjectionWrapper(cell, nwords)
  outputs, _ = tf.nn.dynamic_rnn(cell_out, x_embs, sequence_length=x_lens, dtype=tf.float32)

  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, x_input)
  loss = tf.reduce_mean(losses)
  optimizer = tf.train.AdamOptimizer().minimize(loss)

  print('Graph created.' , file=sys.stderr)

sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=1))
tf.global_variables_initializer().run()
print('Session initialized.' , file=sys.stderr)
train_losses = [] 
print("startup time: %r" % (time.time() - start))
start = time.time()
i = all_time = dev_time = all_tagged = train_words = 0
for ITER in range(10):
  random.shuffle(train_order)
  for i,sid in enumerate(train_order, start=1):
    if i % int(500/args.MB_SIZE) == 0:
      print("Updates so far:", (i-1), "Loss:" , sum(train_losses) / train_words)
      all_tagged += train_words
      train_losses = []
      train_words = 0
      all_time = time.time() - start
    if i % int(10000 / args.MB_SIZE) == 0 or all_time > args.TIMEOUT:
      dev_start = time.time()
      test_losses = []
      test_words = 0
      all_time += time.time() - start
      print("Testing on dev set...")

      for tid in test_order:
        t_examples = test[tid:tid+args.MB_SIZE]
        x_lens_in = [len(example) for example in t_examples]
        x_in = [pad(example, S, max(x_lens_in)) for example in t_examples]
        test_loss = sess.run(loss, feed_dict={x_input: x_in, x_lens: x_lens_in})
        tot_words = sum([len(t_example) for t_example in t_examples])
        test_losses.append(test_loss * tot_words)
        test_words += tot_words
      nll = sum(test_losses) / test_words
      dev_time += time.time() - dev_start 
      train_time = time.time() - start - dev_time
      print('nll=%.4f, ppl=%.4f, time=%.4f, words_per_sec=%.4f' % (nll, math.exp(nll), train_time, all_tagged/train_time), file=sys.stderr)
      if all_time > args.TIMEOUT:
        sys.exit(0)

    # train on sent
    examples = train[sid : sid+args.MB_SIZE]
    x_lens_in = [len(example) for example in examples]
    x_in = [pad(example, S, max(x_lens_in)) for example in examples]
    train_loss, _ = sess.run([loss, optimizer], feed_dict={x_input: x_in, x_lens: x_lens_in})
    tot_words = sum([len(example) for example in examples])
    train_losses.append(train_loss * tot_words)
    train_words += tot_words
