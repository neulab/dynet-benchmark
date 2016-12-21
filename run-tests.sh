#!/bin/bash

ANACONDA_PATH=$HOME/usr/local/anaconda3/envs/benchmark2/lib/
CUDA_PATH=/usr/local/cuda-7.5
DYNET_PATH=$HOME/work/dynet
LD_LIBRARY_PATH=$DYNET_PATH/build/dynet:$CUDA_PATH/lib64
LIBRARY_PATH=$DYNET_PATH/build/dynet:$CUDA_PATH/lib64
PYTHONPATH=$DYNET_PATH/build/python
PYTHON=python
DYNET_MEM=512

# Run RNNLM-Batch
mkdir -p log/rnnlm-batch
for trial in 1 2 3; do
  for embsize in 64 128; do
    hidsize=$(($embsize*2))
    for mbsize in 16 8 4 2 1; do
      for f in dynet-py theano tensorflow chainer; do
        echo "$PYTHON $f/rnnlm-batch.py $mbsize $embsize $hidsize 600 &> log/rnnlm-batch/$f-ms$mbsize-es$embsize-hs$hidsize-t$trial.log"
        $PYTHON $f/rnnlm-batch.py $mbsize $embsize $hidsize 600 &> log/rnnlm-batch/$f-ms$mbsize-es$embsize-hs$hidsize-t$trial.log
      done
      echo "dynet-cpp/rnnlm-batch &> log/rnnlm-batch/dynet-cpp-ms$mbsize-es$embsize-hs$hidsize-t$trial.log"
      dynet-cpp/rnnlm-batch &> log/rnnlm-batch/dynet-cpp-ms$mbsize-es$embsize-hs$hidsize-t$trial.log
    done
  done
done

# # Run python tests
# $PYTHON -u dynet-py/bilstm-tagger.py --dynet-mem $DYNET_MEM
# $PYTHON -u dynet-py/bilstm-tagger-withchar.py --dynet-mem $DYNET_MEM
# $PYTHON -u dynet-py/rnnlm-batch.py --dynet-mem $DYNET_MEM
# $PYTHON -u dynet-py/recnn.py --dynet-mem $DYNET_MEM

# Run C++ tests
# dynet-cpp/bilstm-tagger --dynet-mem $DYNET_MEM
# dynet-cpp/bilstm-tagger-withchar --dynet-mem $DYNET_MEM
# dynet-cpp/rnnlm-batch --dynet-mem $DYNET_MEM 10 128 256
# dynet-cpp/recnn --dynet-mem $DYNET_MEM

# # Run Chainer tests
# $PYTHON -u chainer/rnnlm-batch.py $CHAINER_GPUID

# Run Theano tests
# THEANO_FLAGS=device=cpu,floatX=float32 $PYTHON -u theano/rnnlm-batch.py 10

