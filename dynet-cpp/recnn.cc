#include <regex>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>

#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/dict.h>

using namespace std;
using namespace std::chrono;
using namespace dynet;
using namespace dynet::expr;

class Tree {
public:

  Tree(const string & label, vector<Tree*> children = vector<Tree*>())
                      : label(label), children(children) { }
  ~Tree() {
    for(auto child : children) delete child;
  }

  static Tree* from_sexpr(const string & str) {
    vector<string> toks = tokenize_sexpr(str);
    vector<string>::const_iterator tokit = toks.begin();
    if(*(tokit++) != "(") throw runtime_error("Poorly structured tree");
    return Tree::within_bracket(tokit);
  }

  static vector<string> tokenize_sexpr(const string & s) {
    regex tokker(" +|[()]|[^ ()]+");
    vector<string> toks;
    for(auto it = std::sregex_iterator(s.begin(), s.end(), tokker); it != std::sregex_iterator(); ++it) {
      string m = it->str();
      if(m != " ")
        toks.push_back(m);
    }
    return toks;
  }
  
  static Tree* within_bracket(vector<string>::const_iterator & tokit) {
    const string & label = *(tokit++);
    vector<Tree*> children;
    while(true) {
      const string & tok = *(tokit++);
      if(tok == "(") {
        children.push_back(within_bracket(tokit));
      } else if(tok == ")") {
        return new Tree(label, children);
      } else {
        children.push_back(new Tree(tok));
      }
    }
    throw runtime_error("Poorly structured tree");
  }

  void nonterms(vector<Tree*> & ret) {
    if(!isleaf()) {
      ret.push_back(this);
      for(Tree* child : children) child->nonterms(ret);
    }
  }

  bool isleaf() const { return children.size() == 0; }

  void make_vocab(Dict & nonterm_voc, Dict & term_voc) {
    (isleaf() ? term_voc : nonterm_voc).convert(label);
    for(Tree* tr : children) tr->make_vocab(nonterm_voc, term_voc);
  }

  string label;
  vector<Tree*> children;
  Expression expr;

};

ostream& operator<<(ostream& os, const Tree& tr) {
  if(tr.isleaf()) {
    os << tr.label;
  } else {
    os << '(' << tr.label;
    for(auto child : tr.children) os << ' ' << *child;
    os << ')';
  }
  return os;
}

vector<Tree*> read_dataset(const string & filename) {
  ifstream file(filename);
  if(!file) throw runtime_error("Missing file");
  string line;
  vector<Tree*> ret;
  while(getline(file, line)) ret.push_back(Tree::from_sexpr(line));
  return ret;
}

class TreeRNNBuilder {
public:
  TreeRNNBuilder(Model & model, Dict & word_vocab, unsigned hdim) :
          model(model), word_vocab(word_vocab), hdim(hdim) {
    W = model.add_parameters({hdim, 2*hdim});
    E = model.add_lookup_parameters(word_vocab.size(), {hdim});
  }

  void start_graph(ComputationGraph & c) {
    cg = &c;
    cg_W = parameter(*cg, W);
  }

  Expression expr_for_tree(Tree & tree, bool decorate = false) {
    if(tree.isleaf())
      return lookup(*cg, E, word_vocab.convert(tree.label));
    if(tree.children.size() == 1) {
      assert(tree.children[0]->isleaf());
      Expression expr = expr_for_tree(*tree.children[0]);
      if(decorate) tree.expr = expr;
      return expr;
    }
    assert(tree.children.size() == 2);
    Expression e1 = expr_for_tree(*tree.children[0], decorate);
    Expression e2 = expr_for_tree(*tree.children[1], decorate);
    Expression expr = tanh(cg_W*concatenate({e1,e2}));
    if(decorate) tree.expr = expr;
    return expr;
  }

  Model & model;
  Dict & word_vocab;
  unsigned hdim;
  Parameter W;
  LookupParameter E;

  ComputationGraph * cg;
  Expression cg_W;

};

class TreeLSTMBuilder {
public:
  TreeLSTMBuilder(Model & model, Dict & word_vocab, unsigned wdim, unsigned hdim) :
          model(model), word_vocab(word_vocab), wdim(wdim), hdim(hdim) {
    WS = {model.add_parameters({hdim, wdim}), // 0: Wi
          model.add_parameters({hdim, wdim}), // 1: Wo
          model.add_parameters({hdim, wdim}), // 2: Wu
          model.add_parameters({hdim, 2*hdim}), // 3: Ui
          model.add_parameters({hdim, 2*hdim}), // 4: Uo
          model.add_parameters({hdim, 2*hdim}), // 5: Uu
          model.add_parameters({hdim, hdim}), // 6: UFS1
          model.add_parameters({hdim, hdim}), // 7: UFS2
          model.add_parameters({hdim}), // 8: Bi
          model.add_parameters({hdim}), // 9: Bo
          model.add_parameters({hdim}), // 10: Bu
          model.add_parameters({hdim})};// 11: Bf
    E = model.add_lookup_parameters(word_vocab.size(),{wdim});
    cg_WS.resize(WS.size());
  }

  void start_graph(ComputationGraph & c) {
    cg = &c;
    for(size_t i = 0; i < WS.size(); ++i)
      cg_WS[i] = parameter(*cg, WS[i]);
  }

  Expression expr_for_tree(Tree & tree, bool decorate = false) {
    if(tree.isleaf()) {
      return lookup(*cg, E, word_vocab.convert(tree.label));
    } else if(tree.children.size() == 1) {
      assert(tree.children[0]->isleaf());
      Expression emb, i, o, u, c, expr;
      emb = expr_for_tree(*tree.children[0]);
      i = logistic(cg_WS[0]*emb + cg_WS[8]);
      o = logistic(cg_WS[1]*emb + cg_WS[9]);
      u = tanh(    cg_WS[2]*emb + cg_WS[10]);
      c = cmult(i,u);
      expr = cmult(o,tanh(c));
      if(decorate) { tree.expr = expr; }
      return expr;
    } else {
      assert(tree.children.size() == 2);
      Expression e1, e2, e, i, o, u, f1, f2, c, expr;
      e1 = expr_for_tree(*tree.children[0], decorate);
      e2 = expr_for_tree(*tree.children[1], decorate);
      e = concatenate({e1,e2});
      i = logistic(cg_WS[3]*e + cg_WS[8]);
      o = logistic(cg_WS[4]*e + cg_WS[9]);
      u = tanh(    cg_WS[5]*e + cg_WS[10]);
      f1 = logistic(cg_WS[6]*e1 + cg_WS[11]);
      f2 = logistic(cg_WS[7]*e2 + cg_WS[11]);
      c = cmult(i,u) + cmult(f1,e1) + cmult(f2,e2);
      expr = cmult(o,tanh(c));
      if(decorate) { tree.expr = expr; }
      return expr;
    }
  }

  Model & model;
  Dict & word_vocab;
  unsigned wdim, hdim;
  vector<Parameter> WS;
  LookupParameter E;

  ComputationGraph * cg;
  vector<Expression> cg_WS;

};

int main(int argc, char**argv) {
  vector<Tree*> train = read_dataset("data/trees/train.txt");
  vector<Tree*> dev = read_dataset("data/trees/dev.txt");
  Dict nonterm_voc, term_voc;
  for(auto tree : train) tree->make_vocab(nonterm_voc, term_voc);
  nonterm_voc.freeze();
  term_voc.convert("<unk>"); term_voc.freeze(); term_voc.set_unk("<unk>");

  // DyNet Starts
  dynet::initialize(argc, argv);
  Model model;
  AdamTrainer trainer(&model, 0.001);

  // Builder
  Parameter W_param = model.add_parameters({nonterm_voc.size(), 30});
  // TreeRNNBuilder builder(model, term_voc, 30);
  TreeLSTMBuilder builder(model, term_voc, 300, 30);

  int i = 0, all_tagged = 0, this_nodes = 0;
  float this_loss = 0.f, all_time = 0.f;
  for(int iter = 0; iter < 100; iter++) {
    shuffle(train.begin(), train.end(), *dynet::rndeng);
    time_point<system_clock> start = system_clock::now();
    for(auto tree : train) {
      ComputationGraph cg;
      builder.start_graph(cg);
      Expression W = parameter(cg, W_param);
      Expression d = builder.expr_for_tree(*tree, true);
      vector<Expression> losses;
      vector<Tree*> nodes; tree->nonterms(nodes);
      for(auto nt : nodes)
        losses.push_back(pickneglogsoftmax(W*nt->expr, nonterm_voc.convert(nt->label)));
      Expression loss = sum(losses);
      cg.forward(loss);
      this_loss += as_scalar(loss.value());
      this_nodes += nodes.size();
      cg.backward(loss);
      trainer.update();
      if(++i % 1000 == 0) {
        trainer.status();
        cout << this_loss / this_nodes << endl;
        this_loss = 0; this_nodes = 0;
      }
    }
    all_time += (system_clock::now() - start).count() / 1000000.f;
    trainer.update_epoch(1.0);
    int good = 0, bad = 0;
    for(auto tree : dev) {
      ComputationGraph cg;
      builder.start_graph(cg);
      Expression W = parameter(cg, W_param);
      vector<float> scores = as_vector((W*builder.expr_for_tree(*tree, false)).value());
      size_t max_id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
      (nonterm_voc.convert(max_id) == tree->label ? good : bad)++;
    }
    cout << "accuracy=" << good/float(good+bad) << ", time=" << all_time << ", sent_per_sec=" << i/all_time << endl;
  }
}
