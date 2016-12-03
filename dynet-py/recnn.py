import re
import codecs
from collections import Counter
import random
import numpy as np
import dynet as dy


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

class TreeRNNBuilder(object):
    def __init__(self, model, word_vocab, hdim):
        self.W = model.add_parameters((hdim, 2*hdim))
        self.E = model.add_lookup_parameters((len(word_vocab),hdim))
        self.w2i = word_vocab

    def expr_for_tree(self, tree, decorate=False):
        if tree.isleaf():
            return self.E[self.w2i.get(tree.label,0)]
        if len(tree.children) == 1:
            assert(tree.children[0].isleaf())
            expr = self.expr_for_tree(tree.children[0])
            if decorate: tree._e = expr
            return expr
        assert(len(tree.children) == 2),tree.children[0]
        e1 = self.expr_for_tree(tree.children[0], decorate)
        e2 = self.expr_for_tree(tree.children[1], decorate)
        W = dy.parameter(self.W)
        expr = dy.tanh(W*dy.concatenate([e1,e2]))
        if decorate: tree._e = expr
        return expr

class TreeLSTMBuilder(object):
    def __init__(self, model, word_vocab, wdim, hdim):
        self.WS = [model.add_parameters((hdim, wdim)) for _ in "iou"]
        self.US = [model.add_parameters((hdim, 2*hdim)) for _ in "iou"]
        self.UFS =[model.add_parameters((hdim, hdim)) for _ in "ff"]
        self.BS = [model.add_parameters(hdim) for _ in "iouf"]
        self.E = model.add_lookup_parameters((len(word_vocab),wdim))
        self.w2i = word_vocab

    def expr_for_tree(self, tree, decorate=False):
        if tree.isleaf():
            return self.E[self.w2i.get(tree.label,0)]
        if len(tree.children) == 1:
            assert(tree.children[0].isleaf())
            emb = self.expr_for_tree(tree.children[0])
            Wi,Wo,Wu   = [dy.parameter(w) for w in self.WS]
            bi,bo,bu,_ = [dy.parameter(b) for b in self.BS]
            i = dy.logistic(Wi*emb + bi)
            o = dy.logistic(Wo*emb + bo)
            u = dy.tanh(    Wu*emb + bu)
            c = dy.cmult(i,u)
            expr = dy.cmult(o,dy.tanh(c))
            if decorate: tree._e = expr
            return expr
        assert(len(tree.children) == 2),tree.children[0]
        e1 = self.expr_for_tree(tree.children[0], decorate)
        e2 = self.expr_for_tree(tree.children[1], decorate)
        Ui,Uo,Uu = [dy.parameter(u) for u in self.US]
        Uf1,Uf2 = [dy.parameter(u) for u in self.UFS]
        bi,bo,bu,bf = [dy.parameter(b) for b in self.BS]
        e = dy.concatenate([e1,e2])
        i = dy.logistic(Ui*e + bi)
        o = dy.logistic(Uo*e + bo)
        f1 = dy.logistic(Uf1*e1 + bf)
        f2 = dy.logistic(Uf2*e2 + bf)
        u = dy.tanh(    Uu*e + bu)
        c = dy.cmult(i,u) + dy.cmult(f1,e1) + dy.cmult(f2,e2)
        h = dy.cmult(o,dy.tanh(c))
        expr = h
        if decorate: tree._e = expr
        return expr

train = read_dataset("data/trees/train.txt")
dev = read_dataset("data/trees/dev.txt")

l2i, w2i, i2l, i2w = get_vocabs(train)

model = dy.Model()
#builder = TreeRNNBuilder(model, w2i, 30)
builder = TreeLSTMBuilder(model, w2i, 300, 30)
W_ = model.add_parameters((len(l2i),30))
trainer = dy.AdamTrainer(model)

for ITER in xrange(100):
    random.shuffle(train)
    closs = 0.0
    for i,tree in enumerate(train,1):
        dy.renew_cg()
        W = dy.parameter(W_)
        d = builder.expr_for_tree(tree,True)
        nodes = tree.nonterms()
        losses = [dy.pickneglogsoftmax(W*nt._e,l2i[nt.label]) for nt in nodes]
        loss = dy.esum(losses)
        closs += loss.value()/len(nodes)
        loss.backward()
        trainer.update()
        if i % 1000 == 0:
            trainer.status()
            print closs / 1000
            closs = 0.0
    trainer.update_epoch(1.0)
    good = bad = 0.0
    for tree in dev:
        dy.renew_cg()
        W = dy.parameter(W_)
        pred = i2l[np.argmax((W*builder.expr_for_tree(tree,False)).npvalue())]
        if pred == tree.label: good += 1
        else: bad += 1
    print "iter %s dev accuracy: %s" % (ITER, good/(good+bad))
