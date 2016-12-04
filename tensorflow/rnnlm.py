from collections import Counter, defaultdict
from itertools import count
import random
import time
import math
import sys

import numpy as np
import tensorflow as tf

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file='data/text/train.txt'
test_file='data/text/dev.txt'

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

train = list(read(train_file))
nwords = len(w2i)
test = list(read(test_file))
S = w2i['<s>']
assert(nwords == len(w2i))

max_length = len(max(train, key=len))
assert len(max(test, key=len)) < max_length, 'There should be no test sentences longer than the longest training sentence (%d words)' % max_length

def pad(seq, element, length):
  r = seq + [element] * (length - len(seq))
  assert len(r) == length
  return r 

def main(_):
  # TODO: What is the dynet initializer?

  # Lookup parameters for word embeddings
  WORDS_LOOKUP = tf.Variable(tf.random_uniform([nwords, 1, 64], -1.0, 1.0))

  # Word-level LSTM (one layer, input is unspecified,
  # but will be equal to the embedding dim, output=128)
  cell = tf.nn.rnn_cell.BasicLSTMCell(128) 

  # Softmax weights/biases on top of LSTM outputs
  W_sm = tf.Variable(tf.random_uniform([128, nwords]))
  b_sm = tf.Variable(tf.random_uniform([nwords]))

  # input sentence placeholder
  x = tf.placeholder(tf.int32, [max_length])

  # initialize the RNN
  initial_state = cell.zero_state(1, tf.float32)

  # start the rnn by inputting "<s>"
  cell_output, state = cell(tf.nn.embedding_lookup(WORDS_LOOKUP, x[-1]), initial_state)

  # feed word vectors into the RNN and produce the LSTM outputs
  outputs = []
  for time_step in range(max_length):
    outputs.append(cell_output)
    tf.get_variable_scope().reuse_variables()
    score = tf.matmul(cell_output, W_sm) + b_sm
    emb = tf.nn.embedding_lookup(WORDS_LOOKUP, x[time_step])
    # update the state of the RNN
    cell_output, state = cell(emb, state)

  # Compute the unnormalized log distribution for each word
  logits = tf.matmul(tf.squeeze(outputs), W_sm) + b_sm

  # calculate the loss
  logits_as_list = tf.split(0, max_length, logits)
  loss_weights = [tf.ones([1]) for i in range(max_length)]
  x_as_list = tf.split(0, max_length, x)
  losses = tf.nn.seq2seq.sequence_loss_by_example(logits_as_list, x_as_list, loss_weights)
  loss = tf.reduce_mean(losses)

  optimizer = tf.train.AdamOptimizer().minimize(loss)

  print >>sys.stderr, 'Graph created.' 
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  print >>sys.stderr, 'Session initialized.' 

  train_losses = [] 
  start = time.time()
  i = all_time = all_tagged = train_words = 0
  for ITER in xrange(50):
    random.shuffle(train)
    for s in train:
      i += 1
      if i % 500 == 0:
        print sum(train_losses) / train_words
        all_tagged += train_words
        train_losses = []
        train_words = 0

      if i % 10000 == 9999:
        test_losses = []
        test_words = 0
        all_time += time.time() - start
        for i in range(len(test)):
          example = test[i]
          x_in = pad(example, S, max_length)
          test_loss = sess.run(loss, feed_dict={x: x_in})
          test_losses.append(test_loss * len(example))
          test_words += len(example)
        nll = sum(test_losses) / test_words
        print >>sys.stderr, 'nll=%.4f, ppl=%.4f, time=%.4f, word_per_sec=%.4f' % (nll, math.exp(nll), all_time, all_tagged/all_time)
        if all_time > 300:
          sys.exit(0)
        start = time.time()

      # train on sent
      example = train[i]
      x_in = pad(example, S, max_length)
      train_loss, _ = sess.run([loss, optimizer], feed_dict={x: x_in})
      train_losses.append(train_loss * len(example))
      train_words += len(example)

    # TODO: update_epoch

if __name__ == "__main__":
  tf.app.run()
