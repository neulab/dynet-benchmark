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
parser.add_argument("--input_dim", default=50, type=int)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--mlp_dim", default=32, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--minibatch_size", default=10, type=int)
parser.add_argument("--learning_rate", default=1.0, type=int)
args = parser.parse_args()
print "ARGS:", args

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file='../data/tags/train.txt'
test_file='../data/tags/dev.txt'

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

def pad(seq, element, length):
  r = seq + [element] * (length - len(seq))
  assert len(r) == length
  return r 


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

max_length = len(max(train, key=len))
assert len(max(test, key=len)) < max_length, 'There should be no test sentences longer than the longest training sentence (%d words)' % max_length

train.sort(key=lambda x: -len(x))
test.sort(key=lambda x: -len(x))

if args.minibatch_size != 0:
  train_order = [x*args.minibatch_size for x in range((len(train)-1)/args.minibatch_size + 1)]
  test_order = [x*args.minibatch_size for x in range((len(test)-1)/args.minibatch_size + 1)]
else:
  train_order = range(len(train))
  test_order = range(len(test))



def main(_):

  # Lookup parameters for word/tag embeddings
  WORDS_LOOKUP = tf.Variable(tf.random_uniform([nwords, args.input_dim], -1.0, 1.0))

  mlp_hidden = tf.Variable(tf.random_uniform([args.mlp_dim, args.hidden_dim*2], -1.0, 1.0))
  mlp_out = tf.Variable(tf.random_uniform([args.mlp_dim, ntags], -1.0, 1.0))


  # input sentence placeholder
  words_in = tf.placeholder(tf.int32, [args.minibatch_size, max_length], name="words_in")
  golds_in = tf.placeholder(tf.int32, [args.minibatch_size, max_length], name="golds_in")
  sentLengths_in = tf.placeholder(tf.int32, [args.minibatch_size], name="sentLengths_in")
  mask_input = tf.placeholder(tf.float32, [args.minibatch_size, max_length], name="mask_input")
  
  wembs = []
  gold_tags = []
  for sid in range(args.minibatch_size):
    emb = tf.nn.embedding_lookup(WORDS_LOOKUP, words_in[sid])
    gold_tags.append(golds_in[sid])
    wembs.append(emb)
  # Word-level LSTM (configurable number of layers, input is unspecified,
  # but will be equal to the embedding dim, output=128)
  cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_dim) 
  cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)

  # print len(gold_tags), gold_tags[0]
  gold_tags_as_list = tf.stack(gold_tags, axis=1)
  # print gold_tags_as_list.get_shape()
  gold_tags_as_list = tf.unstack(gold_tags_as_list, axis=0)
  # print len(gold_tags_as_list)
  

  # W_sm = tf.Variable(tf.random_uniform([args.hidden_dim, nwords]))
  # b_sm = tf.Variable(tf.random_uniform([nwords]))

  # initial_state_fwd = cell.zero_state(args.minibatch_size, tf.float32)
  # initial_state_bwd = cell.zero_state(args.minibatch_size, tf.float32)
  # cell_output_fwd, state = cell((tf.nn.embedding_lookup(WORDS_LOOKUP, x[-1])), initial_state)

  # for time_step in range(max_length):
  #   output_fw.append(cell_output)
  #   tf.get_variable_scope().reuse_variables()
  #   score = tf.matmul(cell_output, W_sm) + b_sm
  #   emb = tf.nn.embedding_lookup(WORDS_LOOKUP, x[time_step])
  #   # update the state of the RNN
  #   cell_output, state = cell(tf.squeeze(emb), state)


  outputs, states =  tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                     cell_bw=cell,
                                                     dtype=tf.float32,
                                                     sequence_length=sentLengths_in,
                                                     inputs=tf.stack(wembs, axis=0))


  # Each (args.minibatch_size, max_length, args.hidden_dim)
  output_fw, output_bw = outputs
  # (args.minibatch_size, max_length, 2 * args.hidden_dim)
  output_concat = tf.expand_dims(tf.concat(2, [output_fw, output_bw]), axis=0)
  
  mlp_hidden = tf.expand_dims(mlp_hidden, axis=1)
  mlp_hidden = tf.expand_dims(mlp_hidden, axis=1)
  mlp_out = tf.expand_dims(mlp_out, axis=1)
  mlp_out = tf.expand_dims(mlp_out, axis=1)
  
  # (batch_size, max_length, ntags)
  mlp_prod = tf.reduce_sum(tf.mul(output_concat, mlp_hidden), axis=3)
  mlp_activation = tf.tanh(tf.expand_dims(mlp_prod, axis=3))
  mlp_output = tf.reduce_sum(tf.mul(mlp_out, mlp_activation) , axis=0)
  print mlp_output.get_shape()

  # # calculate the loss
  logits_as_list = tf.split(1, max_length, mlp_output)
  print len(logits_as_list), logits_as_list[0].get_shape()

  # # Mask loss weights using input mask
  loss_weights = tf.mul(tf.ones(shape=(args.minibatch_size, max_length)), mask_input)
  loss_weights = tf.unstack(loss_weights,axis=1)

  #Average log perplexity  
  losses = tf.nn.seq2seq.sequence_loss_by_example(logits_as_list, gold_tags_as_list , loss_weights)
  loss = tf.reduce_mean(losses)

  optimizer = tf.train.AdamOptimizer().minimize(loss)
  # gvs = optimizer.compute_gradients(cost)
  # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
  # optimizer.apply_gradients(capped_gvs)

  print >>sys.stderr, 'Graph created.' 
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  print >>sys.stderr, 'Session initialized.' 

  train_losses = [] 
  start = time.time()
  i = all_time = all_tagged = this_tagged = this_loss = 0
  for ITER in xrange(args.epochs):
    random.shuffle(train_order)
    for i,s in enumerate(train_order, start=1):

      if i % (500/args.minibatch_size) == 0:
        print "Loss:", this_loss / this_tagged
        all_tagged += this_tagged
        this_loss = this_tagged = 0

      if i % (10000/args.minibatch_size) == 0:
        all_time += time.time() - start
        good_sent = bad_sent = good = bad = 0.0
        for tid in test_order:
          t_examples = test[tid:tid+args.minibatch_size]
          words = [[w for w,t in sent] for sent in t_examples]
          golds = [[t for w,t in sent] for sent in t_examples]
          _, tags = sess.run(loss, feed_dict={words_in: words})
          if tags == golds: good_sent += 1
          else: bad_sent += 1
          for go,gu in zip(golds,tags):
            if go == gu: good += 1
            else: bad += 1
        print ("tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f" % (good/(good+bad), good_sent/(good_sent+bad_sent), all_time, all_tagged/all_time))
        if all_time > 300:
          sys.exit(0)
          start = time.time()
        
      # train on sent
      examples = train[s : s+args.minibatch_size]
      words = [[vw.w2i[w] if wc[w]>5 else "_UNK_" for w,t in sent] for sent in examples]
      golds = [[vt.w2i[t] for w,t in sent] for sent in examples]
      sentLengths = [len(sent) for sent in examples]
      masks = [[1.0] * len(example) + [0.0] * (max_length - len(example)) for example in examples]
      train_loss, _ = sess.run([loss, optimizer], feed_dict={words_in:words, golds_in:golds, sentLengths_in:sentLengths, masks_in:masks})
      this_loss += train_loss
      this_tagged += len(golds)

    print "epoch %r finished" % ITER

if __name__ == "__main__":
  tf.app.run()
