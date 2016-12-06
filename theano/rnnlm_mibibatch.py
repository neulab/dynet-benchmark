from __future__ import division
import theano.tensor as T
import theano
import numpy as np
import sys, time
from itertools import chain

from nn.layers.recurrent import LSTM
from nn.layers.embeddings import Embedding
from nn.optimizers import Adam

from collections import Counter, defaultdict
from itertools import count


EMBEDDING_DIM = 64
LSTM_HIDDEN_DIM = 128


train_file = '../data/text/train.txt'
test_file = '../data/text/dev.txt'

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

mask = w2i['<MASK>']
assert mask == 0

train = list(read(train_file))
vocab_size = len(w2i)
test = list(read(test_file))
S = w2i['<s>']

max_length = len(max(train, key=len))
assert len(max(test, key=len)) < max_length, \
    'There should be no test sentences longer than the longest training sentence (%d words)' % max_length


def pad(seq):
    """
    pad a mini-batch input with ending zeros
    """
    batch_size = len(seq)
    max_len = max(len(seq[i]) for i in xrange(batch_size))
    padded_seq = np.zeros((batch_size, max_len), dtype='int32')
    for i in xrange(batch_size):
        padded_seq[i, :len(seq[i])] = seq[i]

    return padded_seq


def build_graph():
    print >>sys.stderr, 'build graph..'
    # Lookup parameters for word embeddings
    embedding_table = Embedding(vocab_size, EMBEDDING_DIM)

    lstm = LSTM(EMBEDDING_DIM, LSTM_HIDDEN_DIM, return_sequences=True)

    # Softmax weights/biases on top of LSTM outputs
    W_sm = theano.shared(np.random.uniform(low=-.5, high=.5, size=(128, vocab_size)), name='W_sm')
    b_sm = theano.shared(np.random.uniform(low=-.5, high=.5, size=vocab_size), name='b_sm')

    # (batch_size, sentence_length)
    x = T.imatrix(name='sentence')

    # (batch_size, sentence_length, embedding_dim)
    sent_embed, sent_mask = embedding_table(x, mask_zero=True)

    lstm_input = T.set_subtensor(T.zeros_like(sent_embed)[:, 1:, :], sent_embed[:, :-1, :])
    lstm_input = T.set_subtensor(lstm_input[:, 1, :], sent_embed[:, -1, :])

    # (batch_size, sentence_length, output_dim)
    lstm_output = lstm(lstm_input)

    # (batch_size, sentence_length, vocab_size)
    logits = T.dot(lstm_output, W_sm) + b_sm
    logits = T.nnet.softmax(logits.reshape((logits.shape[0] * logits.shape[1], vocab_size))).reshape(logits.shape)

    loss = T.log(logits).reshape((-1, logits.shape[-1]))
    # (batch_size * sentence_length)
    loss = loss[T.arange(loss.shape[0]), x.flatten()]
    # (batch_size, sentence_length)
    loss = loss.reshape((x.shape[0], x.shape[1])) * sent_mask
    loss = loss.sum(axis=-1) / sent_mask.sum(axis=-1)
    loss = -T.mean(loss)

    params = embedding_table.params + lstm.params + [W_sm, b_sm]
    updates = Adam().get_updates(params, loss)
    train_loss_func = theano.function([x], loss, updates=updates)
    test_loss_func = theano.function([x], loss)

    return x, train_loss_func, test_loss_func


def train_model():
    x, train_loss_func, test_loss_func = build_graph()
    batch_size = 10

    batch_num = int(np.ceil(len(train) / float(batch_size)))
    batches = [(i*batch_size, min(len(train), (i+1)*batch_size)) for i in range(0, batch_num)]

    start = time.time()
    all_time = all_tagged = 0
    for epoch in xrange(30):
        cum_loss = 0.
        for batch_id, (batch_start, batch_end) in enumerate(batches):
            batch_sents = train[batch_start:batch_end]
            batch_sents_x = pad(batch_sents)

            loss = train_loss_func(batch_sents_x)

            cum_loss += len(batch_sents) * loss
            batch_id += 1
            all_tagged += sum(len(s) for s in batch_sents)

            if batch_id % 10 == 0:
                print >>sys.stderr, 'batch %d, loss=%f' % (batch_id, loss)

            if batch_id % 1000 == 0:
                test_losses = []
                test_words_num = 0
                all_time += time.time() - start

                for i in xrange(len(test)):
                    example = [test[i]]
                    test_loss = test_loss_func(example)
                    test_losses.append(test_loss * len(example))
                    test_words_num += len(example)

                nll = sum(test_losses) / test_words_num
                print >>sys.stderr, 'nll=%.4f, ppl=%.4f, time=%.4f, word_per_sec=%.4f' % (nll, np.exp(nll), all_time, all_tagged / all_time)

                if all_time > 300:
                    sys.exit(0)
                start = time.time()

        avg_loss = cum_loss / len(train)
        print >>sys.stderr, 'epoch %d, avg. loss=%f' % (epoch, avg_loss)


if __name__ == '__main__':
    train_model()
