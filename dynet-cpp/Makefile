

CUDA_PATH=/usr/local/cuda/targets/x86_64-linux/lib
EIGEN_PATH=${HOME}/usr/local/eigen
DYNET_PATH=${HOME}/work/dynet
BUILD_DIR=build

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
all = $(addprefix ${BUILD_DIR}/,rnnlm-batch treenn treenn-bulk bilstm-tagger bilstm-tagger-bulk bilstm-tagger-withchar bilstm-tagger-withchar-bulk)

gpu = $(addprefix ${BUILD_DIR}/,rnnlm-batch-gpu treenn-gpu treenn-bulk-gpu bilstm-tagger-gpu bilstm-tagger-bulk-gpu bilstm-tagger-withchar-gpu bilstm-tagger-withchar-bulk-gpu)

all: $(all)
gpu: $(gpu)

clean:
	rm -f ${BUILD_DIR}

$(all) $(gpu): ${BUILD_DIR}/%: %.cc
	mkdir -p ${BUILD_DIR}
	${CC} -o $@ $< ${CXX_FLAGS} ${DYNET_LIB}
	ln -s $@

