#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>
#ifdef BOOST_REGEX
  #include <boost/regex.hpp>
  using namespace boost;
#else
  #include <regex>
#endif

#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/dict.h>

using namespace std;
using namespace std::chrono;
using namespace dynet;

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
    for(auto it = sregex_iterator(s.begin(), s.end(), tokker); it != sregex_iterator(); ++it) {
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

class TreeLSTMBuilder {
public:
  TreeLSTMBuilder(ParameterCollection & model, Dict & word_vocab, unsigned wdim, unsigned hdim) :
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

  pair<Expression,Expression> expr_for_tree(Tree & tree, bool decorate = false) {
    assert(!tree.isleaf());
    pair<Expression,Expression> hc_ret;
    if(tree.children.size() == 1) {
      assert(tree.children[0]->isleaf());
      Expression emb, i, o, u, c, expr;
      emb = lookup(*cg, E, word_vocab.convert(tree.children[0]->label));
      i = logistic(affine_transform({cg_WS[8], cg_WS[0], emb}));
      o = logistic(affine_transform({cg_WS[9], cg_WS[1], emb}));
      u = tanh(    affine_transform({cg_WS[10], cg_WS[2], emb}));
      hc_ret.second = cmult(i,u);
      hc_ret.first = cmult(o,tanh(hc_ret.second));
    } else {
      assert(tree.children.size() == 2);
      Expression e, i, o, u, f1, f2, c, expr;
      pair<Expression,Expression> hc1, hc2; 
      hc1 = expr_for_tree(*tree.children[0], decorate);
      hc2 = expr_for_tree(*tree.children[1], decorate);
      e = concatenate({hc1.first,hc2.first});
      i = logistic(affine_transform({cg_WS[8], cg_WS[3], e}));
      o = logistic(affine_transform({cg_WS[9], cg_WS[4], e}));
      u = tanh(    affine_transform({cg_WS[10], cg_WS[5], e}));
      f1 = logistic(affine_transform({cg_WS[11], cg_WS[6], hc1.first}));
      f2 = logistic(affine_transform({cg_WS[11], cg_WS[7], hc2.first}));
      hc_ret.second = cmult(i,u) + cmult(f1,hc1.second) + cmult(f2,hc2.second);
      hc_ret.first = cmult(o,tanh(hc_ret.second));
    }
    if(decorate) { tree.expr = hc_ret.first; }
    return hc_ret;
  }

  ParameterCollection & model;
  Dict & word_vocab;
  unsigned wdim, hdim;
  vector<Parameter> WS;
  LookupParameter E;

  ComputationGraph * cg;
  vector<Expression> cg_WS;

};

int main(int argc, char**argv) {

  time_point<system_clock> start = system_clock::now();

  vector<Tree*> train = read_dataset("data/trees/train.txt");
  vector<Tree*> dev = read_dataset("data/trees/dev.txt");
  Dict nonterm_voc, term_voc;
  for(auto tree : train) tree->make_vocab(nonterm_voc, term_voc);
  nonterm_voc.freeze();
  term_voc.convert("<unk>"); term_voc.freeze(); term_voc.set_unk("<unk>");

  // DyNet Starts
  dynet::initialize(argc, argv);
  ParameterCollection model;
  AdamTrainer trainer(model, 0.001);
  trainer.clipping_enabled = false;

  if(argc != 7) {
    cerr << "Usage: " << argv[0] << " WEMBED_SIZE HIDDEN_SIZE SPARSE BATCH_SIZE LAST_STEP TIMEOUT" << endl;
    return 1;
  }
  unsigned WEMBED_SIZE = atoi(argv[1]);
  unsigned HIDDEN_SIZE = atoi(argv[2]);
  trainer.sparse_updates_enabled = atoi(argv[3]);
  int BATCH_SIZE = atoi(argv[4]);
  int LAST_STEP = atoi(argv[5]);
  int TIMEOUT = atoi(argv[6]);

  // Builder
  Parameter W_param = model.add_parameters({nonterm_voc.size(), HIDDEN_SIZE});
  TreeLSTMBuilder builder(model, term_voc, WEMBED_SIZE, HIDDEN_SIZE);

  {
    duration<float> fs = (system_clock::now() - start);
    float startup_time = duration_cast<milliseconds>(fs).count() / float(1000);
    cout << "startup time: " << startup_time << endl;
  }

  shuffle(train.begin(), train.end(), *dynet::rndeng);
  int i = 0, bi = 0, all_tagged = 0, this_nodes = 0;
  float this_loss = 0.f, all_time = 0.f;
  for(int iter = 0; iter < 100; iter++) {
    size_t batch = BATCH_SIZE;
    start = system_clock::now();
    for(size_t j1 = 0; j1 <= train.size()-batch; j1 += batch) {
      ComputationGraph cg;
      builder.start_graph(cg);
      Expression W = parameter(cg, W_param);
      vector<Expression> losses;
      for(size_t j2 = 0; j2 < batch; ++j2) {
        auto & tree = train[j1+j2];
        pair<Expression,Expression> hc = builder.expr_for_tree(*tree, true);
        vector<Tree*> nodes; tree->nonterms(nodes);
        for(auto nt : nodes)
          losses.push_back(pickneglogsoftmax(W*nt->expr, nonterm_voc.convert(nt->label)));
        this_nodes += nodes.size();
        ++i;
      }
      Expression loss = sum(losses);
      cg.forward(loss);
      this_loss += as_scalar(loss.value());
      if(LAST_STEP > 0) {
        cg.backward(loss);
        if(LAST_STEP > 1)
          trainer.update();
      }
      bi++;
      if(bi % (1000/BATCH_SIZE) == 0) {
        trainer.status();
        cout << this_loss / this_nodes << endl;
        this_loss = 0; this_nodes = 0;
      }
    }
    std::chrono::duration<float> fs = (system_clock::now() - start);
    all_time += duration_cast<milliseconds>(fs).count() / float(1000);
    int good = 0, bad = 0;
    for(auto tree : dev) {
      ComputationGraph cg;
      builder.start_graph(cg);
      Expression W = parameter(cg, W_param);
      pair<Expression,Expression> hc = builder.expr_for_tree(*tree, false);
      vector<float> scores = as_vector((W*hc.first).value());
      size_t max_id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
      (nonterm_voc.convert(max_id) == tree->label ? good : bad)++;
    }
    cout << "acc=" << good/float(good+bad) << ", time=" << all_time << ", sent_per_sec=" << i/all_time << ", sec_per_sent=" << all_time/i << endl;
    if(all_time > TIMEOUT)
      exit(0);
  }
}
