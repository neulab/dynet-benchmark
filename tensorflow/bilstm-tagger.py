from __future__ import print_function
import time
start = time.time()

from collections import Counter, defaultdict
from itertools import count
import random
import math
import sys
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.add_argument('WEMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('MLP_SIZE', type=int, help='embedding size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')  # sparse updates by default in tensorflow
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

NUM_LAYERS = 1

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file='/data/tags/train.txt'
test_file='/data/tags/dev.txt'

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

    def size(self): return len(self.w2i.keys())

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
test=list(read(test_file))
words=[]
tags=[]
wc=Counter()
for sent in train:
    for w,p in sent:
        words.append(w)
        tags.append(p)
        wc[w]+=1
words.append("_UNK_")

vw = Vocab.from_corpus([words]) 
vt = Vocab.from_corpus([tags])
UNK = vw.w2i["_UNK_"]

nwords = vw.size()
ntags  = vt.size()
print ("nwords=%r, ntags=%r" % (nwords, ntags))

def get_tags(log_probs):
  sent_tags = []
  for word_probs in log_probs:
    tag = np.argmax(word_probs, axis=0)
    sent_tags.append(tag)
  return sent_tags

if args.gpu:
  cpu_or_gpu = '/gpu:0'
else:
  cpu_or_gpu = '/cpu:0'

with tf.device(cpu_or_gpu):

  # Lookup parameters for word embeddings
  WORDS_LOOKUP = tf.Variable(tf.random_uniform([nwords, 1, args.WEMBED_SIZE], -1.0, 1.0))

  mlp_hidden = tf.Variable(tf.random_uniform([args.HIDDEN_SIZE*2, args.MLP_SIZE], -1.0, 1.0))
  mlp_out = tf.Variable(tf.random_uniform([args.MLP_SIZE, ntags], -1.0, 1.0))

  # input sentence placeholder
  words_in = tf.placeholder(tf.int32, [None], name="input_sentence")
  golds = tf.placeholder(tf.int32, [None], name="golds")
  sent_len = tf.placeholder(tf.int32, shape=(1,), name="sent_len")

  wembs = tf.squeeze(tf.nn.embedding_lookup(WORDS_LOOKUP, words_in), axis=1)
  wembs = tf.expand_dims(wembs, axis=0)
  wembs.set_shape([1, words_in.get_shape()[0], args.WEMBED_SIZE])

  # Word-level LSTM (configurable number of layers, input is unspecified,
  # but will be equal to the embedding dim, output=128)

  cell = tf.nn.rnn_cell.BasicLSTMCell(args.HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=True)
  cell = tf.nn.rnn_cell.MultiRNNCell([cell] * NUM_LAYERS, state_is_tuple=True)

  outputs, _ =  tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                cell_bw=cell,
                                                dtype=tf.float32,
                                                sequence_length=sent_len,
                                                inputs=wembs)

  output_fw, output_bw = outputs
  output_concat = tf.squeeze(tf.concat(2, [output_fw, output_bw]), axis=0)  # (input_length, 2 * HIDDEN_SIZE)
  output_concat.set_shape([None, 2*args.HIDDEN_SIZE])

  # Pass to MLP
  mlp_activation = tf.tanh(tf.matmul(output_concat, mlp_hidden))
  mlp_output = tf.matmul(mlp_activation, mlp_out)

  ## calculate the loss
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(mlp_output, golds)
  loss = tf.reduce_sum(losses)

  optimizer = tf.train.AdamOptimizer().minimize(loss)
  print('Graph created.' , file=sys.stderr)

sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
tf.global_variables_initializer().run()
print('Session initialized.' , file=sys.stderr)
train_losses = [] 
print ("startup time: %r" % (time.time() - start))
start_train = time.time()
i = all_time = dev_time = all_tagged = this_tagged = this_loss = 0

for ITER in range(100):
  random.shuffle(train)
  start = time.time()
  for s in train:
    i += 1
    if i % 500 == 0:   # print status
      print('Updates so far: %d Loss: %f wps: %f' % (i - 1, this_loss / this_tagged, this_tagged/(time.time() - start)))
      all_tagged += this_tagged
      this_loss = this_tagged = 0
      all_time = time.time() - start_train
      start = time.time()
    if i % 10000 == 0 or all_time > args.TIMEOUT: # eval on dev
      dev_start = time.time()
      good_sent = bad_sent = good = bad = 0.0
      for sent in test:
        x_in = [vw.w2i[w] if wc[w]>5 else UNK for w,_ in sent]
        golds_in = [vt.w2i[t] for _,t in sent]
        # log_probs = sess.run(mlp_output, feed_dict={words_in: x_in, golds: golds_in, sent_len: [len(sent)]})
        log_probs = mlp_output.eval(feed_dict={words_in: x_in, golds: golds_in, sent_len: [len(sent)]}, session=sess)
        tags = get_tags(log_probs)
        if tags == golds_in: good_sent += 1
        else: bad_sent += 1
        for go,gu in zip(golds_in,tags):
          if go == gu: good += 1
          else: bad += 1
      dev_time += time.time() - dev_start
      train_time = time.time() - start_train - dev_time
      print ("tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f" % (good/(good+bad), good_sent/(good_sent+bad_sent), train_time, all_tagged/train_time))
      start = start + (time.time() - dev_start)
      if all_time > args.TIMEOUT:
        sys.exit(0)        
    # train on sent         
    x_in = [vw.w2i[w] if wc[w]>5 else UNK for w,_ in s]
    golds_in = [vt.w2i[t] for _,t in s]
    train_loss, _ = sess.run([loss, optimizer], feed_dict={words_in: x_in, golds: golds_in, sent_len: [len(s)]})
    this_loss += train_loss
    this_tagged += len(golds_in)
  print("epoch %r finished" % ITER)
