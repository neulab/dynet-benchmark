

EIGEN_PATH=${HOME}/usr/local/eigen
DYNET_PATH=${HOME}/work/dynet
# *** Mac
# CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -lc++ -ldynet -Ofast
# *** Linux
CC=g++
CXX_FLAGS=-std=c++11 -I${EIGEN_PATH} -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -ldynet -Ofast

all: rnnlm rnnlm-batch recnn bilstm-tagger bilstm-tagger-withchar

clean:
	rm rnnlm rnnlm-batch recnn bilstm-tagger bilstm-tagger-withchar

rnnlm: rnnlm.cc
	${CC} -o rnnlm rnnlm.cc ${CXX_FLAGS}

rnnlm-batch: rnnlm-batch.cc
	${CC} -o rnnlm-batch rnnlm-batch.cc ${CXX_FLAGS}

recnn: recnn.cc
	${CC} -o recnn recnn.cc ${CXX_FLAGS}

bilstm-tagger: bilstm-tagger.cc
	${CC} -o bilstm-tagger bilstm-tagger.cc ${CXX_FLAGS}

bilstm-tagger-withchar: bilstm-tagger-withchar.cc
	${CC} -o bilstm-tagger-withchar bilstm-tagger-withchar.cc ${CXX_FLAGS}
