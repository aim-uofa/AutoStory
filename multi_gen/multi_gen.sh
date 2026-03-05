# !/bin/bash

# model paths
PATH_TO_T2IADAPTOR="experiments/pretrained_models/t2i_adaptor"
PATH_TO_BASE_MODEL="experiments/pretrained_models/anything-v4.0"
PATH_TO_FUSED_MODEL="experiments/MixofShow_Results/Fused_Models/hina+kario+tezuka+mitsuha+son_anythingv4/combined_model.pth"
NUM_SAMPLE_PER_PANEL=2
WORK_DIR="data_bird"


python inference/multi_gen_sample.py \
    --config "$WORK_DIR/final_gen_manual.yaml" \
    --pretrained_model $PATH_TO_BASE_MODEL \
    --combined_model $PATH_TO_FUSED_MODEL \
    --keypose_adaptor_model "${PATH_TO_T2IADAPTOR}/t2iadapter_keypose_sd14v1.pth" \
    --sketch_adaptor_model "${PATH_TO_T2IADAPTOR}/t2iadapter_sketch_sd14v1.pth" \
    --save_dir "$WORK_DIR/final_gen_scattn" \
    --pipeline_type "pure_sd" \
    --num_samples_per_panel $NUM_SAMPLE_PER_PANEL \
    --image_width 512
