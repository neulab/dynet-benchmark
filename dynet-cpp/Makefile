

CUDA_PATH=/usr/local/cuda-7.5
EIGEN_PATH=${HOME}/usr/local/eigen
DYNET_PATH=${HOME}/work/dynet

# *** CPU
DYNET_LIB=-ldynet
# *** GPU
# DYNET_LIB=-lgdynet -ldynetcuda -L${CUDA_PATH}/lib64 -lcublas -lcudart

# *** Mac or linux
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -lc++ ${DYNET_LIB} -Ofast
else
  CC=g++
  CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -DBOOST_REGEX -lboost_regex ${DYNET_LIB} -Ofast
endif

all: rnnlm-batch recnn bilstm-tagger bilstm-tagger-withchar

clean:
	rm rnnlm-batch recnn bilstm-tagger bilstm-tagger-withchar

rnnlm-batch: rnnlm-batch.cc
	${CC} -o rnnlm-batch rnnlm-batch.cc ${CXX_FLAGS}

recnn: recnn.cc
	${CC} -o recnn recnn.cc ${CXX_FLAGS}

bilstm-tagger: bilstm-tagger.cc
	${CC} -o bilstm-tagger bilstm-tagger.cc ${CXX_FLAGS}

bilstm-tagger-withchar: bilstm-tagger-withchar.cc
	${CC} -o bilstm-tagger-withchar bilstm-tagger-withchar.cc ${CXX_FLAGS}
