from __future__ import print_function
import time
start = time.time()

import re
import codecs
from collections import Counter
import random
import sys
import argparse

from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

parser = argparse.ArgumentParser()
parser.add_argument('--chainer_gpu', type=int, default=-1, help='GPU id')
parser.add_argument('WEMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

if args.chainer_gpu >= 0:
  # use GPU
  from chainer.cuda import cupy as xp, get_device
  get_device(args.chainer_gpu).use()
else:
  # use CPU
  import numpy as xp

def makevar(x):
  return Variable(xp.array([x], dtype=xp.int32))

def zeros(dim):
  return Variable(xp.zeros(dim, dtype=xp.float32))

def _tokenize_sexpr(s):
  tokker = re.compile(r" +|[()]|[^ ()]+")
  toks = [t for t in [match.group(0) for match in tokker.finditer(s)] if t[0] != " "]
  return toks

def _within_bracket(toks):
  label = next(toks)
  children = []
  for tok in toks:
    if tok == "(":
      children.append(_within_bracket(toks))
    elif tok == ")":
      return Tree(label, children)
    else: children.append(Tree(tok, None))
  assert(False),list(toks)

class Tree(object):
  def __init__(self, label, children=None):
    self.label = label
    self.children = children

  @staticmethod
  def from_sexpr(string):
    toks = iter(_tokenize_sexpr(string))
    assert next(toks) == "("
    return _within_bracket(toks)

  def __str__(self):
    if self.children is None: return self.label
    return "[%s %s]" % (self.label, " ".join([str(c) for c in self.children]))

  def isleaf(self): return self.children==None

  def leaves_iter(self):
    if self.isleaf():
      yield self
    else:
      for c in self.children:
        for l in c.leaves_iter(): yield l

  def leaves(self): return list(self.leaves_iter())

  def nonterms_iter(self):
    if not self.isleaf():
      yield self
      for c in self.children:
        for n in c.nonterms_iter(): yield n

  def nonterms(self): return list(self.nonterms_iter())

def read_dataset(filename):
  return [Tree.from_sexpr(line.strip()) for line in codecs.open(filename,"r")]

def get_vocabs(trees):
  label_vocab = Counter()
  word_vocab  = Counter()
  for tree in trees:
    label_vocab.update([n.label for n in tree.nonterms()])
    word_vocab.update([l.label for l in tree.leaves()])
  labels = [x for x,c in label_vocab.iteritems() if c > 0]
  words  = ["_UNK_"] + [x for x,c in word_vocab.iteritems() if c > 0]
  l2i = {l:i for i,l in enumerate(labels)}
  w2i = {w:i for i,w in enumerate(words)}
  return l2i, w2i, labels, words

class TreeRNN(Chain):
  def __init__(self, word_vocab, hdim, nc):
    super(TreeRNN, self).__init__(
        embed=L.EmbedID(len(word_vocab), hdim),
        WR=L.Linear(2*hdim, hdim, nobias=True),
        WO=L.Linear(hdim, nc, nobias=True),
    )
    self.w2i = word_vocab

  def expr_for_tree(self, tree, decorate=False):
    if tree.isleaf():
      return self.embed(makevar(self.w2i.get(tree.label, 0)))
    if len(tree.children) == 1:
      assert(tree.children[0].isleaf())
      expr = self.expr_for_tree(tree.children[0])
      if decorate:
        tree._e = expr
      return expr
    assert(len(tree.children) == 2), tree.children[0]
    e1 = self.expr_for_tree(tree.children[0], decorate)
    e2 = self.expr_for_tree(tree.children[1], decorate)
    expr = F.tanh(self.WR(F.concat(e1, e2)))
    if decorate:
      tree._e = expr
    return expr

  def classify(self, e):
    return self.WO(e)

class TreeLSTM(Chain):
  def __init__(self, word_vocab, wdim, hdim, nc):
    super(TreeLSTM, self).__init__(
        embed=L.EmbedID(len(word_vocab), wdim),
        WU=L.Linear(wdim, 4 * hdim), # i,f,o,u with bias (semiterminal)
        W1=L.Linear(hdim, 4 * hdim), # i,f,o,u with bias (left)
        W2=L.Linear(hdim, 4 * hdim), # i,f,o,u with bias (right)
        WO=L.Linear(hdim, nc, nobias=True),
    )
    self.w2i = word_vocab
    self.hdim = hdim

  def expr_for_tree(self, tree, decorate=False):
    if tree.isleaf():
      return zeros((1, self.hdim)), self.embed(makevar(self.w2i.get(tree.label, 0)))
    if len(tree.children) == 1:
      assert(tree.children[0].isleaf())
      c0, e0 = self.expr_for_tree(tree.children[0])
      c, h = F.lstm(c0, self.WU(e0))
      if decorate:
        tree._e = (c, h)
      return c, h
    assert(len(tree.children) == 2), tree.children[0]
    c1, e1 = self.expr_for_tree(tree.children[0], decorate)
    c2, e2 = self.expr_for_tree(tree.children[1], decorate)
    c, h = F.slstm(c1, c2, self.W1(e1), self.W2(e2))
    if decorate:
      tree._e = (c, h)
    return c, h

  def classify(self, e):
    return self.WO(e[1])

train = read_dataset("data/trees/train.txt")
dev = read_dataset("data/trees/dev.txt")

l2i, w2i, i2l, i2w = get_vocabs(train)

tlm = TreeLSTM(w2i, args.WEMBED_SIZE, args.HIDDEN_SIZE, len(l2i))
if args.chainer_gpu >= 0:
  tlm.to_gpu()

trainer = O.Adam()
trainer.use_cleargrads()
trainer.setup(tlm)

print("startup time: %r" % (time.time() - start))
sents = 0
all_time = 0
for ITER in range(100):
  random.shuffle(train)
  closs = 0.0
  cwords = 0
  start = time.time()
  for i,tree in enumerate(train,1):
    sents += 1
    d = tlm.expr_for_tree(tree,True)
    nodes = tree.nonterms()
    losses = [F.softmax_cross_entropy(tlm.classify(nt._e), makevar(l2i[nt.label])) for nt in nodes]
    loss = sum(losses)
    closs += float(loss.data)
    cwords += len(nodes)
    tlm.cleargrads()
    loss.backward()
    trainer.update()
    if sents % 1000 == 0:
      print(closs / cwords)
      closs = 0.0
      cwords = 0
  all_time += time.time() - start
  good = bad = 0.0
  for tree in dev:
    my_data = tlm.classify(tlm.expr_for_tree(tree,False)).data
    if args.chainer_gpu >= 0:
      my_data = xp.asnumpy(my_data)
    pred = i2l[my_data.argmax()]
    if pred == tree.label:
      good += 1
    else:
      bad += 1
  print("acc=%.4f, time=%.4f, sent_per_sec=%.4f" % (good/(good+bad), all_time, sents/all_time))
  if all_time > args.TIMEOUT:
    sys.exit(0)
