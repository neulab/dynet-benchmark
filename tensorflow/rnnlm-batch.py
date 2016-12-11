from collections import Counter, defaultdict
from itertools import count
import random
import time
import math
import sys
import numpy as np
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", default=1, type=int)
parser.add_argument("--input_dim", default=64, type=int)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--minibatch_size", default=10, type=int)
parser.add_argument("--learning_rate", default=1.0, type=int)
args = parser.parse_args()
print "ARGS:", args



# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file='../data/text/train.txt'
test_file='../data/text/dev.txt'
w2i = defaultdict(count(0).next)
eos = "<s>"



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

if args.minibatch_size != 0:
  train_order = [x*args.minibatch_size for x in range((len(train)-1)/args.minibatch_size + 1)]
  test_order = [x*args.minibatch_size for x in range((len(test)-1)/args.minibatch_size + 1)]
else:
  train_order = range(len(train))
  test_order = range(len(test))




max_length = len(max(train, key=len))
assert len(max(test, key=len)) < max_length, 'There should be no test sentences longer than the longest training sentence (%d words)' % max_length


def pad(seq, element, length):
  r = seq + [element] * (length - len(seq))
  assert len(r) == length
  return r 


def main(_):
  # TODO: What is the dynet initializer?


  # Lookup parameters for word embeddings
  WORDS_LOOKUP = tf.Variable(tf.random_uniform([nwords, 1, args.input_dim], -1.0, 1.0))


  # Word-level LSTM (configurable number of layers, input is unspecified,
  # but will be equal to the embedding dim, output=128)
  cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_dim) 
  cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)


  # Softmax weights/biases on top of LSTM outputs
  W_sm = tf.Variable(tf.random_uniform([args.hidden_dim, nwords]))
  b_sm = tf.Variable(tf.random_uniform([nwords]))


  # input sentence placeholder
  x_input = tf.placeholder(tf.int32, [args.minibatch_size, max_length], name="x_input")
  # initialize the RNN
  initial_state = cell.zero_state(args.minibatch_size, tf.float32)


  # start the rnn by inputting "<s>"
  # print x.get_shape() => (64,83)
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


  output = tf.reshape(tf.concat(1, outputs), [-1, args.hidden_dim])


  logits = tf.matmul(tf.squeeze(output), W_sm) + b_sm
  # calculate the loss
  logits_as_list = tf.split(0, max_length, logits)


  # loss_weights = [tf.ones([1]) for i in range(max_length)]
  loss_weights = tf.unstack(tf.ones(shape=(args.minibatch_size, max_length)), axis=1)


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
  start = time.time()
  i = all_time = all_tagged = train_words = 0
  for ITER in xrange(args.epochs):
    random.shuffle(train_order)
    for i,sid in enumerate(train_order, start=1):


      if i % 5 == 0:
        print "Loss:" , sum(train_losses) / train_words
        all_tagged += train_words
        train_losses = []
        train_words = 0


      if i % 50 == 0:
        test_losses = []
        test_words = 0
        all_time += time.time() - start
        print "Testing on dev set..."

        for tid in test_order:
          t_examples = test[tid:tid+args.minibatch_size]
          x_in = [pad(example, S, max_length) for example in t_examples]
          test_loss = sess.run(loss, feed_dict={x_input: x_in})
          tot_words = sum([len(t_example) for t_example in t_examples])
          test_losses.append(test_loss * tot_words)
          test_words += tot_words
        nll = sum(test_losses) / test_words
        print >>sys.stderr, 'nll=%.4f, ppl=%.4f, time=%.4f, words_per_sec=%.4f' % (nll, math.exp(nll), all_time, all_tagged/all_time)
        if all_time > 300:
          sys.exit(0)
        start = time.time()


      # train on sent
      examples = train[sid : sid+args.minibatch_size]
      x_in = [pad(example, S, max_length) for example in examples]
      train_loss, _ = sess.run([loss, optimizer], feed_dict={x_input: x_in})
      tot_words = sum([len(example) for example in examples])
      train_losses.append(train_loss * tot_words)
      train_words += tot_words


    # TODO: update_epoch




if __name__ == "__main__":
  tf.app.run()
