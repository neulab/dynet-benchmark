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
      mycmd="$PYTHON -u $1/$2.py --dynet_mem 1024"
    elif [[ $1 == "theano" ]]; then
      mycmd="THEANO_FLAGS=device=cpu,floatX=float32 $PYTHON -u $1/$2.py"
    else
      mycmd="$PYTHON -u $1/$2.py"
    fi
    echo "$mycmd $3 &> $4"
    eval "$mycmd $3 &> $4"
  fi
}

# Run rnnlm-batch
mkdir -p log/rnnlm-batch
for trial in 1; do
  for embsize in 64 128; do
    hidsize=$(($embsize*2))
    for mbsize in 16 08 04 02 01; do
      for f in dynet-cpp dynet-py theano chainer tensorflow; do
        runcmd $f rnnlm-batch "$mbsize $embsize $hidsize 0 $TIMEOUT" log/rnnlm-batch/$f-ms$mbsize-es$embsize-hs$hidsize-sp0-t$trial.log
        if [[ $f == dynet* ]]; then
          runcmd $f rnnlm-batch "$mbsize $embsize $hidsize 1 $TIMEOUT" log/rnnlm-batch/$f-ms$mbsize-es$embsize-hs$hidsize-sp1-t$trial.log
        fi
      done
    done
  done
done

# Run bilstm-tagger
mkdir -p log/bilstm-tagger
for trial in 1; do
  wembsize=128
  hidsize=50
  mlpsize=32
  for f in dynet-cpp dynet-py theano tensorflow chainer; do
    runcmd $f bilstm-tagger "$wembsize $hidsize $mlpsize 0 $TIMEOUT" log/bilstm-tagger/$f-ws$wembsize-hs$hidsize-mlps$mlpsize-su0-t$trial.log
    if [[ $f == dynet* ]]; then
      runcmd $f bilstm-tagger "$wembsize $hidsize $mlpsize 1 $TIMEOUT" log/bilstm-tagger/$f-ws$wembsize-hs$hidsize-mlps$mlpsize-su1-t$trial.log
    fi
  done
done

# Run bilstm-tagger-withchar
mkdir -p log/bilstm-tagger-withchar
for trial in 1; do
  cembsize=20
  wembsize=128
  hidsize=50
  mlpsize=32
  for f in dynet-cpp dynet-py theano chainer; do
    runcmd $f bilstm-tagger-withchar "$cembsize $wembsize $hidsize $mlpsize 0 $TIMEOUT" log/bilstm-tagger-withchar/$f-cs$cembsize-ws$wembsize-hs$hidsize-mlps$mlpsize-su0-t$trial.log
    if [[ $f == dynet* ]]; then
      runcmd $f bilstm-tagger-withchar "$cembsize $wembsize $hidsize $mlpsize 1 $TIMEOUT" log/bilstm-tagger-withchar/$f-cs$cembsize-ws$wembsize-hs$hidsize-mlps$mlpsize-su1-t$trial.log
    fi
  done
done

# Run treenn
mkdir -p log/treenn
for trial in 1; do
  wembsize=128
  hidsize=128
  for f in dynet-cpp dynet-py chainer; do
    runcmd $f treenn "$wembsize $hidsize $mlpsize 0 $TIMEOUT" log/treenn/$f-ws$wembsize-hs$hidsize-su0-t$trial.log
    if [[ $f == dynet* ]]; then
      runcmd $f treenn "$wembsize $hidsize $mlpsize 1 $TIMEOUT" log/treenn/$f-ws$wembsize-hs$hidsize-su1-t$trial.log
    fi
  done
done
