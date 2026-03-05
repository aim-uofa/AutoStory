#!/bin/bash

python main_batch.py \
	--model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 1 --dataset Custom --evaluate trained_models/table5_pidinet.pth --evaluate-converted \
	--savedir "./savedir/hina" --datadir "../Grounded-Segment-Anything/outputs/hina" \
	--infer_thr 0.5
