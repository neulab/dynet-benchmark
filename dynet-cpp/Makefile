

CUDA_PATH=/usr/local/cuda-7.5
EIGEN_PATH=${HOME}/usr/local/eigen
DYNET_PATH=${HOME}/work/dynet

# *** CPU
DYNET_LIB=-ldynet
# *** GPU
# DYNET_LIB=-lgdynet -ldynetcuda -L${CUDA_PATH}/lib64 -lcublas -lcudart
# GPUSUF=-gpu

# *** Mac or linux
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -lc++ ${DYNET_LIB} -Ofast
else
  CC=g++
  CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -DBOOST_REGEX -lboost_regex ${DYNET_LIB} -Ofast
endif

all: rnnlm-batch treenn bilstm-tagger bilstm-tagger-withchar

clean:
	rm rnnlm-batch${GPUSUF} treenn${GPUSUF} bilstm-tagger${GPUSUF} bilstm-tagger-withchar${GPUSUF}

rnnlm-batch: rnnlm-batch.cc
	${CC} -o rnnlm-batch${GPUSUF} rnnlm-batch.cc ${CXX_FLAGS}

treenn: treenn.cc
	${CC} -o treenn${GPUSUF} treenn.cc ${CXX_FLAGS}

bilstm-tagger: bilstm-tagger.cc
	${CC} -o bilstm-tagger${GPUSUF} bilstm-tagger.cc ${CXX_FLAGS}

bilstm-tagger-withchar: bilstm-tagger-withchar.cc
	${CC} -o bilstm-tagger-withchar${GPUSUF} bilstm-tagger-withchar.cc ${CXX_FLAGS}
