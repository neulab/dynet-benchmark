
DYNET_PATH=${HOME}/work/dynet
CXX_FLAGS=-std=c++11 -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -lc++ -ldynet -g

all: rnnlm rnnlm-batch

rnnlm: rnnlm.cc
	${CC} ${CXX_FLAGS} -o rnnlm rnnlm.cc

rnnlm-batch: rnnlm-batch.cc
	${CC} ${CXX_FLAGS} -o rnnlm-batch rnnlm-batch.cc
