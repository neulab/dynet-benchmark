

CUDA_PATH=/usr/local/cuda/targets/x86_64-linux/lib
EIGEN_PATH=${HOME}/usr/local/eigen
DYNET_PATH=${HOME}/work/dynet

DYNET_LIB=-ldynet

# *** Mac or linux
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -lc++ -Ofast
else
  CC=g++
  CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -DBOOST_REGEX -lboost_regex -Ofast
endif

# The -seq version is commented out for compatibility with master. If using the sequence-ops
# branch you can compile this program as well
all: rnnlm-batch treenn treenn-bulk bilstm-tagger bilstm-tagger-bulk bilstm-tagger-withchar bilstm-tagger-withchar-bulk

gpu: rnnlm-batch-gpu treenn-gpu treenn-bulk-gpu bilstm-tagger-gpu bilstm-tagger-bulk-gpu bilstm-tagger-withchar-gpu bilstm-tagger-withchar-bulk-gpu

clean:
	rm -f rnnlm-batch treenn treenn-bulk bilstm-tagger bilstm-tagger-bulk bilstm-tagger-withchar bilstm-tagger-withchar-bulk rnnlm-batch-gpu treenn-gpu treenn-bulk-gpu bilstm-tagger-gpu bilstm-tagger-bulk-gpu bilstm-tagger-withchar-gpu bilstm-tagger-withchar-bulk-gpu

rnnlm-batch: rnnlm-batch.cc
	${CC} -o rnnlm-batch rnnlm-batch.cc ${CXX_FLAGS} ${DYNET_LIB}

rnnlm-seq: rnnlm-seq.cc
	${CC} -o rnnlm-seq rnnlm-seq.cc ${CXX_FLAGS} ${DYNET_LIB}

treenn: treenn.cc
	${CC} -o treenn treenn.cc ${CXX_FLAGS} ${DYNET_LIB}

treenn-bulk: treenn-bulk.cc
	${CC} -o treenn-bulk treenn-bulk.cc ${CXX_FLAGS} ${DYNET_LIB}

bilstm-tagger: bilstm-tagger.cc
	${CC} -o bilstm-tagger bilstm-tagger.cc ${CXX_FLAGS} ${DYNET_LIB}

bilstm-tagger-bulk: bilstm-tagger-bulk.cc
	${CC} -o bilstm-tagger-bulk bilstm-tagger-bulk.cc ${CXX_FLAGS} ${DYNET_LIB}

bilstm-tagger-withchar: bilstm-tagger-withchar.cc
	${CC} -o bilstm-tagger-withchar bilstm-tagger-withchar.cc ${CXX_FLAGS} ${DYNET_LIB}

bilstm-tagger-withchar-bulk: bilstm-tagger-withchar-bulk.cc
	${CC} -o bilstm-tagger-withchar-bulk bilstm-tagger-withchar-bulk.cc ${CXX_FLAGS} ${DYNET_LIB}

rnnlm-batch-gpu: rnnlm-batch.cc
	${CC} -o rnnlm-batch-gpu rnnlm-batch.cc ${CXX_FLAGS} ${DYNET_LIB}

rnnlm-seq-gpu: rnnlm-seq.cc
	${CC} -o rnnlm-seq-gpu rnnlm-seq.cc ${CXX_FLAGS} ${DYNET_LIB}

treenn-gpu: treenn.cc
	${CC} -o treenn-gpu treenn.cc ${CXX_FLAGS} ${DYNET_LIB}

treenn-bulk-gpu: treenn-bulk.cc
	${CC} -o treenn-bulk-gpu treenn-bulk.cc ${CXX_FLAGS} ${DYNET_LIB}

bilstm-tagger-gpu: bilstm-tagger.cc
	${CC} -o bilstm-tagger-gpu bilstm-tagger.cc ${CXX_FLAGS} ${DYNET_LIB}

bilstm-tagger-bulk-gpu: bilstm-tagger-bulk.cc
	${CC} -o bilstm-tagger-bulk-gpu bilstm-tagger-bulk.cc ${CXX_FLAGS} ${DYNET_LIB}

bilstm-tagger-withchar-gpu: bilstm-tagger-withchar.cc
	${CC} -o bilstm-tagger-withchar-gpu bilstm-tagger-withchar.cc ${CXX_FLAGS} ${DYNET_LIB}

bilstm-tagger-withchar-bulk-gpu: bilstm-tagger-withchar-bulk.cc
	${CC} -o bilstm-tagger-withchar-bulk-gpu bilstm-tagger-withchar-bulk.cc ${CXX_FLAGS} ${DYNET_LIB}
