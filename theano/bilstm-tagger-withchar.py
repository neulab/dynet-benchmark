from __future__ import division, print_function
import time
start = time.time()

import random

import theano.tensor as T
import theano
from theano.ifelse import ifelse
import numpy as np
import sys
import argparse
from itertools import chain

from nn.layers.recurrent import LSTM, BiLSTM
from nn.layers.embeddings import Embedding
from nn.activations import softmax
from nn.optimizers import Adam
from nn.initializations import uniform

from collections import Counter, defaultdict
from itertools import count

parser = argparse.ArgumentParser()
parser.add_argument("--dynet_mem", default=512, type=int)
parser.add_argument('CEMBED_SIZE', type=int, help='embedding size')
parser.add_argument('WEMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('MLP_SIZE', type=int, help='embedding size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

# format of files: each line is "word1|tag2 word2|tag2 ..."
train_file="data/tags/train.txt"
dev_file="data/tags/dev.txt"


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

  def size(self):
    return len(self.w2i.keys())


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
dev=list(read(dev_file))
words=[]
tags=[]
chars=set()
wc=Counter()
for sent in train:
  for w,p in sent:
    words.append(w)
    tags.append(p)
    chars.update(w)
    wc[w]+=1
words.append("_UNK_")
chars.add("<*>")

vw = Vocab.from_corpus([words])
vt = Vocab.from_corpus([tags])
vc = Vocab.from_corpus([['_CHAR_MASK_'] + list(chars)])
UNK = vw.w2i["_UNK_"]

char_mask = vc.w2i['_CHAR_MASK_']
# mask of chars must be zero
assert char_mask == 0

nwords = vw.size()
ntags  = vt.size()
nchars  = vc.size()
print("nwords=%r, ntags=%r, nchars=%r" % (nwords, ntags, nchars))


def word2id(w):
  if wc[w] > 5:
    w_index = vw.w2i[w]
    return w_index
  else:
    return UNK


def build_tag_graph():
  print('build graph..', file=sys.stderr)

  # (sentence_length)
  # word indices for a sentence
  x = T.ivector(name='sentence')

  # (sentence_length, max_char_num_per_word)
  # character indices for each word in a sentence
  x_chars = T.imatrix(name='sent_word_chars')

  # (sentence_length)
  # target tag
  y = T.ivector(name='tag')

  # Lookup parameters for word embeddings
  word_embeddings = Embedding(nwords, args.WEMBED_SIZE, name='word_embeddings')

  # Lookup parameters for character embeddings
  char_embeddings = Embedding(nchars, args.CEMBED_SIZE, name='char_embeddings')

  # lstm for encoding word characters
  char_lstm = BiLSTM(args.CEMBED_SIZE, int(args.WEMBED_SIZE / 2), name='char_lstm')

  # bi-lstm
  lstm = BiLSTM(args.WEMBED_SIZE, args.HIDDEN_SIZE, return_sequences=True, name='lstm')

  # MLP
  W_mlp_hidden = uniform((args.HIDDEN_SIZE * 2, args.MLP_SIZE), name='W_mlp_hidden')
  W_mlp = uniform((args.MLP_SIZE, ntags), name='W_mlp')

  # def get_word_embed_from_chars(word_chars):
  #   # (max_char_num_per_word, char_embed_dim)
  #   # (max_char_num_per_word)
  #   word_char_embeds, word_char_masks = char_embeddings(word_chars, mask_zero=True)
  #   word_embed = char_lstm(T.unbroadcast(word_char_embeds[None, :, :], 0), mask=T.unbroadcast(word_char_masks[None, :], 0))[0]
  #
  #   return word_embed

  # def word_embed_look_up_step(word_id, word_chars):
  #   word_embed = ifelse(T.eq(word_id, UNK),
  #             get_word_embed_from_chars(word_chars),  # if it's a unk
  #             word_embeddings(word_id))
  #
  #   return word_embed

  word_embed_src = T.eq(x, UNK).astype('float32')[:, None]

  # (sentence_length, word_embedding_dim)
  word_embed = word_embeddings(x)

  # (sentence_length, max_char_num_per_word, char_embed_dim)
  # (sentence_length, max_char_num_per_word)
  word_char_embeds, word_char_masks = char_embeddings(x_chars, mask_zero=True)

  # (sentence_length, word_embedding_dim)
  word_embed_from_char = char_lstm(word_char_embeds, mask=word_char_masks)

  sent_embed = word_embed_src * word_embed_from_char + (1 - word_embed_src) * word_embed

  # # (sentence_length, embedding_dim)
  # sent_embed, _ = theano.scan(word_embed_look_up_step, sequences=[x, x_chars])

  # (sentence_length, lstm_hidden_dim)
  lstm_output = lstm(T.unbroadcast(sent_embed[None, :, :], 0))[0]

  # (sentence_length, ntags)
  mlp_output = T.dot(T.tanh(T.dot(lstm_output, W_mlp_hidden)), W_mlp)

  tag_prob = T.log(T.nnet.softmax(mlp_output))

  tag_nll = - tag_prob[T.arange(tag_prob.shape[0]), y]

  loss = tag_nll.sum()

  params = word_embeddings.params + char_embeddings.params + char_lstm.params + lstm.params + [W_mlp_hidden, W_mlp]
  updates = Adam().get_updates(params, loss)
  train_loss_func = theano.function([x, x_chars, y], loss, updates=updates)

  # build the decoding graph
  decode_func = theano.function([x, x_chars], tag_prob)

  return train_loss_func, decode_func


def sent_to_theano_input(sent):
  tags = np.asarray([vt.w2i[t] for w, t in sent], dtype='int32')
  words = np.asarray([word2id(w) for w, t in sent], dtype='int32')

  max_char_num_per_word = max(len(w) + 2 for w, t in sent)
  word_chars = np.zeros((len(words), max_char_num_per_word), dtype='int32')
  pad_char = vc.w2i["<*>"]
  for i, (word, tag) in enumerate(sent):
    word_chars[i, :len(word) + 2] = [pad_char] + [vc.w2i[c] for c in word] + [pad_char]

  return words, word_chars, tags


def tag_sent(sent, decode_func):
  words, word_chars, ref_tags = sent_to_theano_input(sent)

  # (sentence_length, tag_num)
  tag_prob = decode_func(words, word_chars)

  tag_results = tag_prob.argmax(axis=-1)
  tag_results = [vt.i2w[tid] for tid in tag_results]

  return tag_results

train_func, decode_func = build_tag_graph()

print("startup time: %r" % (time.time() - start))
start = time.time()
i = all_time = dev_time = all_tagged = this_tagged = this_loss = 0

for ITER in range(100):
  random.shuffle(train)
  for s in train:
    i += 1

    if i % 500 == 0:  # print status
      print(this_loss / this_tagged)
      all_tagged += this_tagged
      this_loss = this_tagged = 0
      all_time = time.time() - start
    if i % 10000 == 0 or all_time > args.TIMEOUT: # eval on dev
      dev_start = time.time()
      all_time += time.time() - start
      good_sent = bad_sent = good = bad = 0.0
      for sent in dev:
        golds = [t for w, t in sent]

        # package words in a batch
        tags = tag_sent(sent, decode_func)

        if tags == golds:
          good_sent += 1
        else:
          bad_sent += 1
        for go, gu in zip(golds, tags):
          if go == gu:
            good += 1
          else:
            bad += 1

      dev_time += time.time() - dev_start 
      train_time = time.time() - start - dev_time
      print("tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f" % (
           good / (good + bad), good_sent / (good_sent + bad_sent), train_time, all_tagged / train_time))

      if all_time > args.TIMEOUT:
        sys.exit(0)

    # train on training sentences

    # word indices
    # char indices for each word
    # gold tags
    words, word_chars, tags = sent_to_theano_input(s)

    loss = train_func(words, word_chars, tags)

    this_loss += loss
    this_tagged += len(s)
    # print('loss: %f' % loss)

  print("epoch %r finished" % ITER)
