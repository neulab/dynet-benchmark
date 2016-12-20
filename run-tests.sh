#!/bin/bash

CUDA_PATH=/usr/local/cuda-7.5
DYNET_PATH=$HOME/work/dynet-vanilla
LD_LIBRARY_PATH=$DYNET_PATH/build/dynet:$CUDA_PATH/lib64
LIBRARY_PATH=$DYNET_PATH/build/dynet:$CUDA_PATH/lib64
PYTHONPATH=$DYNET_PATH/build/python
PYTHON=python2.7
DYNET_MEM=512
CHAINER_GPUID=0

# # Run python tests
# $PYTHON -u dynet-py/bilstm-tagger.py --dynet-mem $DYNET_MEM
# $PYTHON -u dynet-py/bilstm-tagger-withchar.py --dynet-mem $DYNET_MEM
# $PYTHON -u dynet-py/rnnlm-batch.py --dynet-mem $DYNET_MEM
# $PYTHON -u dynet-py/recnn.py --dynet-mem $DYNET_MEM

# Run C++ tests
# dynet-cpp/bilstm-tagger --dynet-mem $DYNET_MEM
# dynet-cpp/bilstm-tagger-withchar --dynet-mem $DYNET_MEM
dynet-cpp/rnnlm-batch --dynet-mem $DYNET_MEM 10 128 256
# dynet-cpp/recnn --dynet-mem $DYNET_MEM

# # Run Chainer tests
# $PYTHON -u chainer/rnnlm-batch.py $CHAINER_GPUID

# Run Theano tests
# THEANO_FLAGS=device=cpu,floatX=float32 $PYTHON -u theano/rnnlm-batch.py 10

