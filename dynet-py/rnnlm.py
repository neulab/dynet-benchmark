from collections import Counter, defaultdict
from itertools import count
import random
import time
import math
import sys

import dynet as dy
import numpy as np

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file="data/text/train.txt"
test_file="data/text/dev.txt"

class Vocab:
    def __init__(self):
        self.w2i = defaultdict(count(0).next)
        self.i2w = []

    def size(self): return len(self.w2i.keys())

def read(fname, vw):
    """
    Read a file where each line is of the form "word1 word2 ..."
    Yields lists of the form [word1, word2, ...]
    """
    with file(fname) as fh:
        for line in fh:
            sent = [vw.w2i[x] for x in line.strip().split()]
            sent.append(vw.w2i["<s>"])
            yield sent

vw = Vocab()
train=list(read(train_file, vw))
nwords_train = vw.size()
test=list(read(test_file, vw))
S = vw.w2i["<s>"]

nwords = vw.size()
assert(nwords == nwords_train)

# DyNet Starts

model = dy.Model()
trainer = dy.AdamTrainer(model)

# Lookup parameters for word embeddings
WORDS_LOOKUP = model.add_lookup_parameters((nwords, 64))

# Word-level LSTM (layers=1, input=64, output=128, model)
RNN = dy.LSTMBuilder(1, 64, 128, model)

# Softmax weights/biases on top of LSTM outputs
W_sm = model.add_parameters((nwords, 128))
b_sm = model.add_parameters(nwords)

# Build the language model graph
def calc_lm_loss(sent):

    dy.renew_cg()
    # parameters -> expressions
    W_exp = dy.parameter(W_sm)
    b_exp = dy.parameter(b_sm)

    # initialize the RNN
    f_init = RNN.initial_state()

    # start the rnn by inputting "<s>"
    s = f_init.add_input(WORDS_LOOKUP[sent[-1]]) 

    # feed word vectors into the RNN and predict the next word
    losses = []
    for wid in sent:
        # calculate the softmax and loss
        score = W_exp * s.output() + b_exp
        loss = dy.pickneglogsoftmax(score, wid)
        losses.append(loss)
        # update the state of the RNN
        s = s.add_input(WORDS_LOOKUP[wid]) 
    
    return dy.esum(losses)

start = time.time()
i = all_time = all_tagged = this_words = this_loss = 0
for ITER in xrange(50):
    random.shuffle(train)
    for s in train:
        i += 1
        if i % 500 == 0:
            trainer.status()
            print this_loss / this_words
            all_tagged += this_words
            this_loss = this_words = 0
        if i % 10000 == 0:
            all_time += time.time() - start
            dev_loss = dev_words = 0
            for sent in test:
                loss_exp = calc_lm_loss(sent)
                dev_loss += loss_exp.scalar_value()
                dev_words += len(sent)
            print ("nll=%.4f, ppl=%.4f, time=%.4f, word_per_sec=%.4f" % (dev_loss/dev_words, math.exp(dev_loss/dev_words), all_time, all_tagged/all_time))
            if all_time > 300:
                sys.exit(0)
            start = time.time()
        # train on sent
        loss_exp = calc_lm_loss(s)
        this_loss += loss_exp.scalar_value()
        this_words += len(s)
        loss_exp.backward()
        trainer.update()
    trainer.update_epoch(1.0)
