import time
start = time.time()

from collections import Counter, defaultdict
from itertools import count
import random
import math
import sys

import numpy as np
import tensorflow as tf

if len(sys.argv) != 5:
  print("Usage: %s MB_SIZE EMBED_SIZE HIDDEN_SIZE TIMEOUT" % sys.argv[0])
  sys.exit(1)
MB_SIZE = int(sys.argv[1])
EMBED_SIZE = int(sys.argv[2])
HIDDEN_SIZE = int(sys.argv[3])
TIMEOUT = int(sys.argv[4]) 
NUM_LAYERS = 1

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
      sent = [w2i[x] for x in line.strip().split()]
      sent.append(w2i[eos])
      yield sent

train = list(read(train_file))
nwords = len(w2i)
test = list(read(test_file))
S = w2i[eos]
assert(nwords == len(w2i))

train.sort(key=lambda x: -len(x))
test.sort(key=lambda x: -len(x))

if MB_SIZE != 0:
  train_order = [x*MB_SIZE for x in range((len(train)-1)/MB_SIZE + 1)]
  test_order = [x*MB_SIZE for x in range((len(test)-1)/MB_SIZE + 1)]
else:
  train_order = range(len(train))
  test_order = range(len(test))

max_length = len(max(train, key=len))
assert len(max(test, key=len)) < max_length, 'There should be no test sentences longer than the longest training sentence (%d words)' % max_length

def pad(seq, element, length):
  r = seq + [element] * (length - len(seq))
  assert len(r) == length
  return r 

# Lookup parameters for word embeddings
WORDS_LOOKUP = tf.Variable(tf.random_uniform([nwords, 1, EMBED_SIZE], -1.0, 1.0))

# Word-level LSTM (configurable number of layers, input is unspecified,
# but will be equal to the embedding dim, output=128)
cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) 
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * NUM_LAYERS, state_is_tuple=True)

# Softmax weights/biases on top of LSTM outputs
W_sm = tf.Variable(tf.random_uniform([HIDDEN_SIZE, nwords]))
b_sm = tf.Variable(tf.random_uniform([nwords]))

# input sentence placeholder
x_input = tf.placeholder(tf.int32, [MB_SIZE, max_length], name="x_input")
mask_input = tf.placeholder(tf.float32, [MB_SIZE, max_length], name="mask_input")

# initialize the RNN
initial_state = cell.zero_state(MB_SIZE, tf.float32)

# start the rnn by inputting "<s>"
# print x.get_shape() => (mb_size, max_length)
x = tf.unstack(x_input, num=max_length, axis=1)

cell_output, state = cell(tf.squeeze(tf.nn.embedding_lookup(WORDS_LOOKUP, x[-1])), initial_state)
# feed word vectors into the RNN and produce the LSTM outputs
outputs = []
for time_step in range(max_length):
  outputs.append(cell_output)
  tf.get_variable_scope().reuse_variables()
  score = tf.matmul(cell_output, W_sm) + b_sm
  emb = tf.nn.embedding_lookup(WORDS_LOOKUP, x[time_step])
  # update the state of the RNN
  cell_output, state = cell(tf.squeeze(emb), state)

# Compute the unnormalized log distribution for each word for each sent in the batch
output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_SIZE])

logits = tf.matmul(tf.squeeze(output), W_sm) + b_sm
# calculate the loss
logits_as_list = tf.split(0, max_length, logits)

# Mask loss weights using input mask
loss_weights = tf.mul(tf.ones(shape=(MB_SIZE, max_length)), mask_input)
loss_weights = tf.unstack(loss_weights,axis=1)

x = tf.stack(x, axis=1)

x_as_list = [tf.split(0, max_length, sent) for sent in tf.unstack(x, axis=0)]
x_as_list = tf.squeeze(tf.stack(x_as_list, axis=1))

#Average log perplexity  
losses = tf.nn.seq2seq.sequence_loss_by_example(logits_as_list, tf.unstack(x_as_list, axis=0), loss_weights)
loss = tf.reduce_mean(losses)

optimizer = tf.train.AdamOptimizer().minimize(loss)

print >>sys.stderr, 'Graph created.' 
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print >>sys.stderr, 'Session initialized.' 

train_losses = [] 
print ("startup time: %r" % (time.time() - start))
start = time.time()
i = all_time = all_tagged = train_words = 0
for ITER in xrange(10):
  random.shuffle(train_order)
  for i,sid in enumerate(train_order, start=1):
    if i % int(500/MB_SIZE) == 0:
      print "Updates so far:", (i-1), "Loss:" , sum(train_losses) / train_words
      all_tagged += train_words
      train_losses = []
      train_words = 0

    if i % int(10000/MB_SIZE) == 0:
      test_losses = []
      test_words = 0
      all_time += time.time() - start
      print "Testing on dev set..."

      for tid in test_order:
        t_examples = test[tid:tid+MB_SIZE]
        x_in = [pad(example, S, max_length) for example in t_examples]
        masks = [[1.0] * len(example) + [0.0] * (max_length - len(example)) for example in examples]
        test_loss = sess.run(loss, feed_dict={x_input: x_in, mask_input: masks})
        tot_words = sum([len(t_example) for t_example in t_examples])
        test_losses.append(test_loss * tot_words)
        test_words += tot_words
      nll = sum(test_losses) / test_words
      print >>sys.stderr, 'nll=%.4f, ppl=%.4f, time=%.4f, words_per_sec=%.4f' % (nll, math.exp(nll), all_time, all_tagged/all_time)
      if all_time > TIMEOUT:
        sys.exit(0)
      start = time.time()

    # train on sent
    examples = train[sid : sid+MB_SIZE]
    x_in = [pad(example, S, max_length) for example in examples]
    masks = [[1.0] * len(example) + [0.0] * (max_length - len(example)) for example in examples]
    train_loss, _ = sess.run([loss, optimizer], feed_dict={x_input: x_in, mask_input: masks})
    tot_words = sum([len(example) for example in examples])
    train_losses.append(train_loss * tot_words)
    train_words += tot_words

