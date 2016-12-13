from collections import defaultdict
import time
import random
import dynet as dy
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
train = list(read_dataset("data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

# Start DyNet and define trainer
model = dy.Model()
trainer = dy.AdamTrainer(model, 0.001)
trainer.set_clip_threshold(-1.0)
trainer.set_sparse_updates(False)

# Define the model
W_sm = model.add_lookup_parameters((nwords, ntags), dy.ConstInitializer(0.0)) # Word weights
b_sm = model.add_parameters((ntags), dy.ConstInitializer(0.0))                # Softmax bias

# A function to calculate scores for one value
def calc_scores(words):
  dy.renew_cg()
  score = dy.esum([dy.lookup(W_sm, x) for x in words])
  b_sm_exp = dy.parameter(b_sm)
  return score + b_sm_exp

for ITER in range(100):
  # Perform training
  # random.shuffle(train)
  train_loss = 0.0
  start = time.time()
  for i, (words, tag) in enumerate(train):
    scores = calc_scores(words)
    my_loss = dy.pickneglogsoftmax(scores, tag)
    train_loss += my_loss.value()
    my_loss.backward()
    trainer.update()
    # print(b_sm.as_array())
    # if i > 5:
    #     sys.exit(0)
  print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss/len(train), time.time()-start))
  # Perform testing
  test_correct = 0.0
  for words, tag in dev:
    scores = calc_scores(words).npvalue()
    predict = np.argmax(scores)
    if predict == tag:
      test_correct += 1
  print("iter %r: test acc=%.4f" % (ITER, test_correct/len(dev)))
