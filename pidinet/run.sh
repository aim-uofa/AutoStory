#!/bin/bash 

data_dir=(
	../Grounded-Segment-Anything/outputs/1 \
	../Grounded-Segment-Anything/outputs/2 \
	../Grounded-Segment-Anything/outputs/3 \
	../Grounded-Segment-Anything/outputs/4 \
	../Grounded-Segment-Anything/outputs/5 \
	../Grounded-Segment-Anything/outputs/6 \
	../Grounded-Segment-Anything/outputs/7 \
	../Grounded-Segment-Anything/outputs/8 \
	../Grounded-Segment-Anything/outputs/9 \
	../Grounded-Segment-Anything/outputs/10 \
	../Grounded-Segment-Anything/outputs/11 
	)

output_dir=(
	"./savedir/1" \
	"./savedir/2" \
	"./savedir/3" \
	"./savedir/4" \
	"./savedir/5" \
	"./savedir/6" \
	"./savedir/7" \
	"./savedir/8" \
	"./savedir/9" \
	"./savedir/10" \
	"./savedir/11" 
)

for((i=0;i<${#data_dir[@]};i++));  
do   
  python main.py \
    --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 1 --dataset Custom --evaluate trained_models/table5_pidinet.pth --evaluate-converted \
    --savedir ${output_dir[i]} --datadir ${data_dir[i]} \
    --infer_thr 0.5
done 
