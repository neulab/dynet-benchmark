#!/bin/bash

myrun() {
  if [[ ! -e $2 ]]; then
    echo "$1 &> $2"
    eval "$1 &> $2"
  fi
}

for run in 1 2 3; do
  for batch_size in 1 4 16 64 256; do
    for size in 256 512 1024; do
      for last_step in 2; do
        for autobatch in 0 1 2; do
          myrun "./treenn-bulk --dynet-seed $run --dynet-mem 10000 --dynet-autobatch $autobatch $size $size 1 $batch_size $last_step 60" log/tnn-cpu-ab$autobatch-ls$last_step-s$size-bs$batch_size-r$run.log
          myrun "./treenn-bulk-gpu --dynet-seed $run --dynet-mem 10000 --dynet-autobatch $autobatch $size $size 0 $batch_size $last_step 60" log/tnn-gpu-ab$autobatch-ls$last_step-s$size-bs$batch_size-r$run.log
          myrun "./bilstm-tagger-bulk --dynet-seed $run --dynet-mem 10000 --dynet-autobatch $autobatch $size $size $size 1 $batch_size $last_step 60" log/bt-cpu-ab$autobatch-ls$last_step-s$size-bs$batch_size-r$run.log
          myrun "./bilstm-tagger-bulk-gpu --dynet-seed $run --dynet-mem 10000 --dynet-autobatch $autobatch $size $size $size 0 $batch_size $last_step 60" log/bt-gpu-ab$autobatch-ls$last_step-s$size-bs$batch_size-r$run.log
          myrun "./bilstm-tagger-withchar-bulk --dynet-seed $run --dynet-mem 10000 --dynet-autobatch $autobatch 64 $size $size $size 1 $batch_size $last_step 60" log/btw-cpu-ab$autobatch-ls$last_step-s$size-bs$batch_size-r$run.log
          myrun "./bilstm-tagger-withchar-bulk-gpu --dynet-seed $run --dynet-mem 10000 --dynet-autobatch $autobatch 64 $size $size $size 0 $batch_size $last_step 60" log/btw-gpu-ab$autobatch-ls$last_step-s$size-bs$batch_size-r$run.log
        done
      done
    done
  done
done
