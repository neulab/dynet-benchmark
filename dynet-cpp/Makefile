
DYNET_PATH=${HOME}/work/dynet
CXX_FLAGS=-std=c++11 -I${DYNET_PATH} -L${DYNET_PATH}/build/dynet -lc++ -ldynet

rnnlm: rnnlm.cc
	${CC} ${CXX_FLAGS} -o rnnlm rnnlm.cc
