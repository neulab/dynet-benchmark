#!/bin/bash

export ANACONDA_PATH=$HOME/usr/local/anaconda3/envs/benchmark2
export CUDA_PATH=/usr/local/cuda-7.5
export DYNET_PATH=$HOME/work/dynet
export LD_LIBRARY_PATH=$DYNET_PATH/build/dynet:$ANACONDA_PATH/lib:$CUDA_PATH/lib64
export LIBRARY_PATH=$DYNET_PATH/build/dynet:$ANACONDA_PATH/lib:$CUDA_PATH/lib64
export PYTHONPATH=$DYNET_PATH/build/python
PYTHON=python
DYNET_MEM=512

TIMEOUT=600

runcmd() {
  if [[ ! -e $4 ]]; then
    if [[ $1 == "dynet-cpp" ]]; then
      mycmd="$1/$2 --dynet_mem 1024"
    elif [[ $1 == "dynet-py" ]]; then
      mycmd="$PYTHON $1/$2.py --dynet_mem 1024"
    else
      mycmd="$PYTHON $1/$2.py"
    fi
    echo "$mycmd $3 &> $4"
    eval "$mycmd $3 &> $4"
  fi
}

# Run RNNLM-Batch
mkdir -p log/rnnlm-batch
for trial in 1 2 3; do
  for embsize in 64 128; do
    hidsize=$(($embsize*2))
    for mbsize in 16 8 4 2 1; do
      for f in dynet-cpp dynet-py theano tensorflow chainer; do
        runcmd $f rnnlm-batch "$mbsize $embsize $hidsize $TIMEOUT" log/rnnlm-batch/dynet-cpp-ms$mbsize-es$embsize-hs$hidsize-t$trial.log
      done
    done
  done
done

# Run bilstm-tagger
mkdir -p log/bilstm-tagger
for trial in 1 2 3; do
  wembsize=128
  hidsize=50
  mlpsize=32
  for f in dynet-cpp dynet-py theano tensorflow chainer; do
    runcmd $f bilstm-tagger "$wembsize $hidsize $mlpsize $TIMEOUT" log/bilstm-tagger/$f-ws$wembsize-hs$hidsize-mlps$mlpsize-t$trial.log
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

