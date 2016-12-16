from collections import defaultdict
from operator import itemgetter
import time
import random
import tensorflow as tf
import numpy as np
import sys

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
train = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)
EPOCHS = 100

print ("nwords=%r, ntags=%r" % (nwords, ntags))
# Determine max length across train and dev set
max_length = 0
for sent in train:
  if len(sent[0]) > max_length:
    max_length = len(sent[0])

for sent in dev:
  if len(sent[0]) > max_length:
    max_length = len(sent[0])

def pad(seq, element, length):
  r = seq + [element] * (length - len(seq))
  assert len(r) == length
  return r 

def main(_):

  W_sm = tf.Variable(tf.random_uniform([nwords, ntags], -1.0, 1.0)) # Word weights 
  b_sm = tf.Variable(tf.random_uniform([ntags], -1.0, 1.0))  # Softmax bias
  words_in = tf.placeholder(tf.int32, shape=[max_length])
  tags_in = tf.placeholder(tf.int32, shape=[1])
  masks_in = tf.placeholder(tf.float32, shape=[max_length])
  
  ##Calculate scores
  embs = [tf.expand_dims(tf.nn.embedding_lookup(W_sm, x), axis=1) for x in tf.unstack(words_in)]
  embs_concat = tf.concat(1, embs)
  score = tf.mul(embs_concat, masks_in)
  score = tf.reduce_sum(score, axis=1)
  score_out = tf.add(score, b_sm)

  # Add dims to match sequence_loss_by_examples func definition
  score_to_loss = tf.expand_dims(score_out, axis=0)
  score_to_loss = tf.expand_dims(score_to_loss, axis=0)

  # Calculate loss 
  # loss_weights = tf.unstack(tf.Variable(tf.ones(1)))
  loss_weights = tf.unstack(tf.random_uniform([1], -1.0, 1.0))

  losses = tf.nn.seq2seq.sequence_loss_by_example(tf.unstack(score_to_loss), tf.unstack(tags_in), loss_weights)
  loss = tf.reduce_mean(losses)

  optimizer = tf.train.AdamOptimizer().minimize(loss)

  print >>sys.stderr, 'Graph created.' 
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  print >>sys.stderr, 'Session initialized.'

  for ITER in range(EPOCHS):
    
    # Perform training
    # random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for i, (words, tag) in enumerate(train):
      padded_words = pad(words, UNK, max_length)
      mask = [1.0] * len(words) + [0.0] * (max_length - len(words))
      _, cur_loss, _ = sess.run([score_out, loss, optimizer], feed_dict={words_in: padded_words, tags_in: [tag], masks_in: mask})
      train_loss += cur_loss

      # print(b_sm.as_array())
      # if i > 5:
      #     sys.exit(0)
    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss/len(train), time.time()-start))
    
    # Perform testing
    test_correct = 0.0
    for words, tag in dev:
      padded_words = pad(words, UNK, max_length)
      mask = [1.0] * len(words) + [0.0] * (max_length - len(words))
      prob_scores = sess.run(score_out, feed_dict={words_in: padded_words, tags_in: [tag], masks_in: mask})
      predict = np.argmax(prob_scores)
      if predict == tag:
        test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct/len(dev)))

if __name__ == "__main__":
    tf.app.run()
