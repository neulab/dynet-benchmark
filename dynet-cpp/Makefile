

CUDA_PATH=/usr/local/cuda-8.0/targets/x86_64-linux/lib
EIGEN_PATH=${HOME}/usr/local/eigen
DYNET_PATH=${HOME}/work/dynet

DYNET_LIB_CPU=-ldynet
DYNET_LIB_GPU=-lgdynet -L${CUDA_PATH} -lcublas -lcudart

# *** Mac or linux
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -lc++ -Ofast
else
  CC=g++
  CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -DBOOST_REGEX -lboost_regex -Ofast
endif

all: rnnlm-batch treenn treenn-bulk bilstm-tagger bilstm-tagger-bulk bilstm-tagger-withchar bilstm-tagger-withchar-bulk

gpu: rnnlm-batch-gpu treenn-gpu treenn-bulk-gpu bilstm-tagger-gpu bilstm-tagger-bulk-gpu bilstm-tagger-withchar-gpu bilstm-tagger-withchar-bulk-gpu

clean:
	rm -f rnnlm-batch treenn treenn-bulk bilstm-tagger bilstm-tagger-bulk bilstm-tagger-withchar bilstm-tagger-withchar-bulk rnnlm-batch-gpu treenn-gpu treenn-bulk-gpu bilstm-tagger-gpu bilstm-tagger-bulk-gpu bilstm-tagger-withchar-gpu bilstm-tagger-withchar-bulk-gpu

rnnlm-batch: rnnlm-batch.cc
	${CC} -o rnnlm-batch rnnlm-batch.cc ${CXX_FLAGS} ${DYNET_LIB_CPU}

rnnlm-seq: rnnlm-seq.cc
	${CC} -o rnnlm-seq rnnlm-seq.cc ${CXX_FLAGS} ${DYNET_LIB_CPU}

treenn: treenn.cc
	${CC} -o treenn treenn.cc ${CXX_FLAGS} ${DYNET_LIB_CPU}

treenn-bulk: treenn-bulk.cc
	${CC} -o treenn-bulk treenn-bulk.cc ${CXX_FLAGS} ${DYNET_LIB_CPU}

bilstm-tagger: bilstm-tagger.cc
	${CC} -o bilstm-tagger bilstm-tagger.cc ${CXX_FLAGS} ${DYNET_LIB_CPU}

bilstm-tagger-bulk: bilstm-tagger-bulk.cc
	${CC} -o bilstm-tagger-bulk bilstm-tagger-bulk.cc ${CXX_FLAGS} ${DYNET_LIB_CPU}

bilstm-tagger-withchar: bilstm-tagger-withchar.cc
	${CC} -o bilstm-tagger-withchar bilstm-tagger-withchar.cc ${CXX_FLAGS} ${DYNET_LIB_CPU}

bilstm-tagger-withchar-bulk: bilstm-tagger-withchar-bulk.cc
	${CC} -o bilstm-tagger-withchar-bulk bilstm-tagger-withchar-bulk.cc ${CXX_FLAGS} ${DYNET_LIB_CPU}

rnnlm-batch-gpu: rnnlm-batch.cc
	${CC} -o rnnlm-batch-gpu rnnlm-batch.cc ${CXX_FLAGS} ${DYNET_LIB_GPU}

rnnlm-seq-gpu: rnnlm-seq.cc
	${CC} -o rnnlm-seq-gpu rnnlm-seq.cc ${CXX_FLAGS} ${DYNET_LIB_GPU}

treenn-gpu: treenn.cc
	${CC} -o treenn-gpu treenn.cc ${CXX_FLAGS} ${DYNET_LIB_GPU}

treenn-bulk-gpu: treenn-bulk.cc
	${CC} -o treenn-bulk-gpu treenn-bulk.cc ${CXX_FLAGS} ${DYNET_LIB_GPU}

bilstm-tagger-gpu: bilstm-tagger.cc
	${CC} -o bilstm-tagger-gpu bilstm-tagger.cc ${CXX_FLAGS} ${DYNET_LIB_GPU}

bilstm-tagger-bulk-gpu: bilstm-tagger-bulk.cc
	${CC} -o bilstm-tagger-bulk-gpu bilstm-tagger-bulk.cc ${CXX_FLAGS} ${DYNET_LIB_GPU}

bilstm-tagger-withchar-gpu: bilstm-tagger-withchar.cc
	${CC} -o bilstm-tagger-withchar-gpu bilstm-tagger-withchar.cc ${CXX_FLAGS} ${DYNET_LIB_GPU}

bilstm-tagger-withchar-bulk-gpu: bilstm-tagger-withchar-bulk.cc
	${CC} -o bilstm-tagger-withchar-bulk-gpu bilstm-tagger-withchar-bulk.cc ${CXX_FLAGS} ${DYNET_LIB_GPU}
