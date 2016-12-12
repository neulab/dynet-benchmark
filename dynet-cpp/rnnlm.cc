#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <chrono>

#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>
#include <dynet/training.h>

using namespace std;
using namespace std::chrono;
using namespace dynet;
using namespace dynet::expr;

// Read a file where each line is of the form "word1 word2 ..."
// Yields lists of the form [word1, word2, ...]
vector<vector<int> > read(const string & fname, Dict & vw) {
  ifstream fh(fname);
  if(!fh) throw std::runtime_error("Could not open file");
  string str; 
  vector<vector<int> > sents;
  while(getline(fh, str)) {
    istringstream iss(str);
    vector<int> tokens;
    while(iss >> str)
      tokens.push_back(vw.convert(str));
    tokens.push_back(vw.convert("<s>"));
    sents.push_back(tokens);
  }
  return sents;
}

struct RNNLanguageModel {
  LookupParameter p_c;
  Parameter W_sm;
  Parameter b_sm;
  VanillaLSTMBuilder builder;
  explicit RNNLanguageModel(unsigned layers, unsigned input_dim, unsigned hidden_dim, unsigned vocab_size, Model& model) : builder(layers, input_dim, hidden_dim, model) {
    p_c = model.add_lookup_parameters(vocab_size, {input_dim}); 
    W_sm = model.add_parameters({vocab_size, hidden_dim});
    b_sm = model.add_parameters({vocab_size});
  }

  Expression calc_lm_loss(const vector<int> & sent, ComputationGraph & cg) {
  
    // parameters -> expressions
    Expression W_exp = parameter(cg, W_sm);
    Expression b_exp = parameter(cg, b_sm);
  
    // initialize the RNN
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
  
    // start the rnn by inputting "<s>"
    Expression s = builder.add_input(lookup(cg, p_c, *sent.rbegin())); 
  
    // feed word vectors into the RNN and predict the next word
    vector<Expression> losses;
    for(auto wid : sent) {
      // calculate the softmax and loss
      Expression score = affine_transform({b_exp, W_exp, s});
      Expression loss = pickneglogsoftmax(score, wid);
      losses.push_back(loss);
      // update the state of the RNN
      s = builder.add_input(lookup(cg, p_c, wid));
    }
    
    return sum(losses);
  }

};

int main(int argc, char** argv) {

  // format of files: each line is "word1 word2 ..."
  string train_file = "data/text/train.txt";
  string test_file = "data/text/dev.txt";

  Dict vw;
  vector<vector<int> > train = read(train_file, vw);
  vw.freeze();
  vector<vector<int> > test = read(test_file, vw);

  int nwords = vw.size();

  // DyNet Starts
  dynet::initialize(argc, argv);
  Model model;
  AdamTrainer trainer(model, 0.001);
  trainer.sparse_updates_enabled = false;

  RNNLanguageModel rnnlm(1, 64, 128, nwords, model);

  time_point<system_clock> start = system_clock::now();
  int i = 0, all_tagged = 0, this_words = 0;
  float this_loss = 0.f, all_time = 0.f;
  for(int iter = 0; iter < 10; iter++) {
    shuffle(train.begin(), train.end(), *dynet::rndeng);
    for(auto & s : train) {
      i++;
      if(i % 500 == 0) {
        trainer.status();
        cout << this_loss/this_words << endl;
        all_tagged += this_words;
        this_loss = 0.f;
        this_words = 0;
      }
      if(i % 5000 == 0) {
        duration<float> fs = (system_clock::now() - start);
        all_time += duration_cast<milliseconds>(fs).count() / float(1000);
        int test_words = 0;
        float test_loss = 0;
        for(auto & sent : test) {
          ComputationGraph cg;
          Expression loss_exp = rnnlm.calc_lm_loss(sent, cg);
          test_loss += as_scalar(cg.forward(loss_exp));
          test_words += sent.size();
        }
        cout << "nll=" << test_loss/test_words << ", ppl=" << exp(test_loss/test_words) << ", words=" << test_words << ", time=" << all_time << ", word_per_sec=" << all_tagged/all_time << endl;
        if(all_time > 3600)
          exit(0);
        start = system_clock::now();
      }

      ComputationGraph cg;
      Expression loss_exp = rnnlm.calc_lm_loss(s, cg);
      this_loss += as_scalar(cg.forward(loss_exp));
      this_words += s.size();
      cg.backward(loss_exp);
      trainer.update();
    }
    trainer.update_epoch(1.0);
  }
}
