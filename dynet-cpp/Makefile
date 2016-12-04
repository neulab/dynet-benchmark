
DYNET_PATH=${HOME}/work/dynet
CXX_FLAGS=-std=c++11 -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -lc++ -ldynet -Ofast

all: rnnlm rnnlm-batch recnn

clean:
	rm rnnlm rnnlm-batch recnn

rnnlm: rnnlm.cc
	${CC} ${CXX_FLAGS} -o rnnlm rnnlm.cc

rnnlm-batch: rnnlm-batch.cc
	${CC} ${CXX_FLAGS} -o rnnlm-batch rnnlm-batch.cc

recnn: recnn.cc
	${CC} ${CXX_FLAGS} -o recnn recnn.cc
