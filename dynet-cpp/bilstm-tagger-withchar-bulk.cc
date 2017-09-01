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
#include <dynet/lstm.h>

using namespace std;
using namespace std::chrono;
using namespace dynet;

// Read a file where each line is of the form "word1|tag1 word2|tag2 ..."
// Yields pairs of lists of the form < [word1, word2, ...], [tag1, tag2, ...] >
vector<pair<vector<string>, vector<string> > > read(const string & fname) {
  ifstream fh(fname);
  if(!fh) throw std::runtime_error("Could not open file");
  string str;
  regex re("[ |]");
  vector<pair<vector<string>, vector<string> > > sents;
  while(getline(fh, str)) {
    pair<vector<string>,vector<string> > word_tags;
    sregex_token_iterator first{str.begin(), str.end(), re, -1}, last;
    while(first != last) {
      word_tags.first.push_back(*first++);
      assert(first != last);
      word_tags.second.push_back(*first++);
    }
    sents.push_back(word_tags);
  }
  return sents;
}

class BiLSTMTagger {
public:

  BiLSTMTagger(unsigned layers, unsigned cembed_dim, unsigned wembed_dim, unsigned hidden_dim, unsigned mlp_dim, ParameterCollection & model, Dict & wv, Dict & cv, Dict & tv, unordered_map<string,int> & wc)
                        : wv(wv), cv(cv), tv(tv), wc(wc) {
    unsigned nwords = wv.size();
    unsigned ntags  = tv.size();
    unsigned nchars  = cv.size();
    word_lookup = model.add_lookup_parameters(nwords, {wembed_dim});
    char_lookup = model.add_lookup_parameters(nchars, {cembed_dim});

    // MLP on top of biLSTM outputs 100 -> mlp_dim -> ntags
    pH = model.add_parameters({mlp_dim, hidden_dim*2});
    pO = model.add_parameters({ntags, mlp_dim});

    // word-level LSTMs
    fwdRNN = VanillaLSTMBuilder(1, wembed_dim, hidden_dim, model); // layers, in-dim, out-dim, model
    bwdRNN = VanillaLSTMBuilder(1, wembed_dim, hidden_dim, model);

    // char-level LSTMs
    cFwdRNN = VanillaLSTMBuilder(1, cembed_dim, wembed_dim/2, model);
    cBwdRNN = VanillaLSTMBuilder(1, cembed_dim, wembed_dim/2, model);
  }

  Dict &wv, &cv, &tv;
  unordered_map<string,int> & wc;
  LookupParameter word_lookup, char_lookup;
  Parameter p_t1, pH, pO;
  VanillaLSTMBuilder fwdRNN, bwdRNN, cFwdRNN, cBwdRNN;

  // Do word representation
  Expression word_rep(ComputationGraph & cg, const string & w) {
    if(wc[w] > 5) {
      return lookup(cg, word_lookup, wv.convert(w));
    } else {
      Expression pad = lookup(cg, char_lookup, cv.convert("<*>"));
      vector<Expression> cembs(w.size()+2, pad);
      for(size_t i = 0; i < w.size(); ++i)
        cembs[i+1] = lookup(cg, char_lookup, cv.convert(w.substr(i, 1)));
      cFwdRNN.start_new_sequence();
      for(size_t i = 0; i < cembs.size(); ++i) cFwdRNN.add_input(cembs[i]);
      cBwdRNN.start_new_sequence();
      for(size_t i = cembs.size(); i > 0; --i) cBwdRNN.add_input(cembs[i-1]);
      return concatenate({cFwdRNN.back(), cBwdRNN.back()});
    }
  }

  vector<Expression> build_tagging_graph(ComputationGraph & cg, const vector<string> & words) {
    // parameters -> expressions
    Expression H = parameter(cg, pH);
    Expression O = parameter(cg, pO);

    // initialize the RNNs
    fwdRNN.new_graph(cg);
    bwdRNN.new_graph(cg);
    cFwdRNN.new_graph(cg);
    cBwdRNN.new_graph(cg);

    // get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    vector<Expression> wembs(words.size()), fwds(words.size()), bwds(words.size()), fbwds(words.size());
    for(size_t i = 0; i < words.size(); ++i)
      wembs[i] = word_rep(cg, words[i]);

    // feed word vectors into biLSTM
    fwdRNN.start_new_sequence();
    for(size_t i = 0; i < wembs.size(); ++i)
      fwds[i] = fwdRNN.add_input(wembs[i]);
    bwdRNN.start_new_sequence();
    for(size_t i = wembs.size(); i > 0; --i) {
      bwds[i-1] = bwdRNN.add_input(wembs[i-1]);
      fbwds[i-1] = O * tanh( H * concatenate({fwds[i-1], bwds[i-1]}) );
    }

    return fbwds;
  }

  Expression sent_loss(ComputationGraph & cg, vector<string> & words, vector<string> & tags) {
    vector<Expression> exprs = build_tagging_graph(cg, words), errs(words.size());
    for(size_t i = 0; i < tags.size(); ++i)
      errs[i] = pickneglogsoftmax(exprs[i], tv.convert(tags[i]));
    return sum(errs);
  }

  vector<string> tag_sent(vector<string> & words) {
    ComputationGraph cg;
    vector<Expression> exprs = build_tagging_graph(cg, words), errs(words.size());
    vector<string> tags(words.size());
    for(size_t i = 0; i < words.size(); ++i) {
      vector<float> scores = as_vector(exprs[i].value());
      size_t max_id = distance(scores.begin(), max_element(scores.begin(), scores.end()));
      tags[i] = tv.convert(max_id);
    }
    return tags;
  }

};

int main(int argc, char**argv) {

  time_point<system_clock> start = system_clock::now();

  // DyNet Starts
  dynet::initialize(argc, argv);
  ParameterCollection model;
  AdamTrainer trainer(model, 0.001);
  trainer.clipping_enabled = false;

  if(argc != 9) {
    cerr << "Usage: " << argv[0] << " CEMBED_SIZE WEMBED_SIZE HIDDEN_SIZE MLP_SIZE SPARSE BATCH_SIZE LAST_STEP TIMEOUT" << endl;
    return 1;
  }
  int CEMBED_SIZE = atoi(argv[1]);
  int WEMBED_SIZE = atoi(argv[2]);
  int HIDDEN_SIZE = atoi(argv[3]);
  int MLP_SIZE = atoi(argv[4]);
  trainer.sparse_updates_enabled = atoi(argv[5]);
  int BATCH_SIZE = atoi(argv[6]);
  int LAST_STEP = atoi(argv[7]);
  int TIMEOUT = atoi(argv[8]);

  vector<pair<vector<string>, vector<string> > > train = read("data/tags/train.txt");
  vector<pair<vector<string>, vector<string> > > dev = read("data/tags/dev.txt");
  Dict word_voc, tag_voc, char_voc;
  unordered_map<string, int> word_cnt;
  for(auto & sent : train) {
    for(auto & w : sent.first) {
      word_voc.convert(w);
      word_cnt[w]++;
      for(size_t i = 0; i < w.size(); ++i)
        char_voc.convert(w.substr(i,1));
    }
    for(auto & t : sent.second)
      tag_voc.convert(t);
  }
  tag_voc.freeze();
  word_voc.convert("<unk>"); word_voc.freeze(); word_voc.set_unk("<unk>");
  char_voc.convert("<*>"); char_voc.freeze();

  // Initilaize the tagger
  BiLSTMTagger tagger(1, CEMBED_SIZE, WEMBED_SIZE, HIDDEN_SIZE, MLP_SIZE, model, word_voc, char_voc, tag_voc, word_cnt);

  {
    duration<float> fs = (system_clock::now() - start);
    float startup_time = duration_cast<milliseconds>(fs).count() / float(1000);
    cout << "startup time: " << startup_time << endl;
  }

  // Do training
  shuffle(train.begin(), train.end(), *dynet::rndeng);
  start = system_clock::now();
  int i = 0, bi = 0, all_tagged = 0, this_words = 0;
  int batch = BATCH_SIZE;
  float this_loss = 0.f, all_time = 0.f;
  for(int iter = 0; iter < 100; iter++) {
    for(size_t id1 = 0; id1 <= train.size()-batch; id1 += batch) {
      i += batch;
      bi++;
      if(bi % (500/BATCH_SIZE) == 0) {
        trainer.status();
        cout << this_loss/this_words << endl;
        all_tagged += this_words;
        this_loss = 0.f;
        this_words = 0;
      }
      if(bi % (5000/BATCH_SIZE) == 0) {
        duration<float> fs = (system_clock::now() - start);
        all_time += duration_cast<milliseconds>(fs).count() / float(1000);
        int dev_words = 0, dev_good = 0;
        float dev_loss = 0;
        for(auto & sent : dev) {
          vector<string> tags = tagger.tag_sent(sent.first);
          for(size_t j = 0; j < tags.size(); ++j)
            if(tags[j] == sent.second[j])
              dev_good++;
          dev_words += sent.second.size();
        }
        cout << "acc=" << dev_good/float(dev_words) << ", time=" << all_time << ", word_per_sec=" << all_tagged/all_time << ", sent_per_sec=" << i/all_time << ", sec_per_sent=" << all_time/i << endl;
        if(all_time > TIMEOUT)
          exit(0);
        start = system_clock::now();
      }

      ComputationGraph cg;
      vector<Expression> losses;
      for(size_t id2 = 0; id2 < batch; ++id2) {
        auto & s = train[id1+id2];
        losses.push_back(tagger.sent_loss(cg, s.first, s.second));
        this_words += s.first.size();
      }
      Expression loss_exp = sum(losses);
      this_loss += as_scalar(cg.forward(loss_exp));
      if(LAST_STEP > 0) {
        cg.backward(loss_exp);
        if(LAST_STEP > 1)
          trainer.update();
      }
    }
  }
  return 0;
}
