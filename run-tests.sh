#!/bin/bash

CUDA_PATH=/usr/local/cuda-7.5
DYNET_PATH=$HOME/work/dynet
LD_LIBRARY_PATH=$DYNET_PATH/build/dynet:$CUDA_PATH/lib64
PYTHON=python2.7
DYNET_MEM=11000
CHAINER_GPUID=0

# # Run python tests
# $PYTHON dynet-py/bilstm-tagger.py --dynet-mem $DYNET_MEM
# $PYTHON dynet-py/bilstm-tagger-withchar.py --dynet-mem $DYNET_MEM
# $PYTHON dynet-py/rnnlm.py --dynet-mem $DYNET_MEM
# $PYTHON dynet-py/rnnlm-batch.py --dynet-mem $DYNET_MEM
# $PYTHON dynet-py/recnn.py --dynet-mem $DYNET_MEM

# Run C++ tests
# dynet-cpp/bilstm-tagger --dynet-mem $DYNET_MEM
# dynet-cpp/bilstm-tagger-withchar --dynet-mem $DYNET_MEM
# dynet-cpp/rnnlm --dynet-mem $DYNET_MEM
dynet-cpp/rnnlm-batch --dynet-mem $DYNET_MEM
dynet-cpp/recnn --dynet-mem $DYNET_MEM

# # Run Chainer tests
# $PYTHON chainer/rnnlm.py $CHAINER_GPUID
# $PYTHON chainer/rnnlm-batch.py $CHAINER_GPUID
