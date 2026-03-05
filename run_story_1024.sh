# !/bin/bash

# model paths
PATH_TO_T2IADAPTOR="experiments/pretrained_models/t2i_adaptor"
PATH_TO_BASE_MODEL="experiments/pretrained_models/coffeebreak"
PATH_TO_FUSED_MODEL="experiments/composed_lora/coffeebreak/mazaki_hayasaka_chisato/combined_model_.pth"

# set args
ROLE_1="chisato"
ROLE_2="mazaki"
ROLE_3="hayasaka"
ROLE_1_CLS="person"
ROLE_2_CLS="person"
ROLE_3_CLS="person"
USER_INPUT="Write a short story about 3 girls, their names are ${ROLE_1}, ${ROLE_2} and ${ROLE_3}"
NEGATIVE_PROMPT="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
ADAPTOR_WEIGHT=0.7

# default args
OUTPUT_DIR="output_stories"
GPT_MODEL="gpt-4"
IMAGE_WIDTH=1024
REPLICATE=0
NUM_SAMPLE_PER_SINGE_ROLE=10  # number of candidate samples per single role for single_gen
NUM_SAMPLE_PER_PANEL=10  # number of candidate samples per panel for final generation

# set work dir
WORK_DIR="${OUTPUT_DIR}/${GPT_MODEL}_${IMAGE_WIDTH}x512/${USER_INPUT}/replicate_${REPLICATE}"  # TODO: automatic
echo "WORK_DIR: $WORK_DIR"


# generate story
GENERATE_STORY=1
if [ $GENERATE_STORY -eq 1 ]
then
    python story_utils/generate_story_1024x512.py \
        --story "$USER_INPUT" \
        --output_dir "$OUTPUT_DIR" \
        --model "$GPT_MODEL"
fi


# postprocess for layout and local & global prompts  TODO: write to mixofshow script
PROCESS_BOX=0  # generate box layout image, generate layout compose config, generate final gen config
UPDATE_KEYPOSE_COMPOSE_YAML_FILE=0
UPDATE_FINAL_GEN_YAML_FILE=1
if [ $PROCESS_BOX -eq 1 ]
then
    python story_utils/process_bbox.py \
        --work_dir "$WORK_DIR" \
        --negative_prompt "$NEGATIVE_PROMPT" \
        --cond_type "keypose" \
        --adaptor_weight $ADAPTOR_WEIGHT \
        --image_width $IMAGE_WIDTH \
        --update_keypose_compose_config $UPDATE_KEYPOSE_COMPOSE_YAML_FILE \
        --update_final_gen_config $UPDATE_FINAL_GEN_YAML_FILE
fi
# NOTE: Please copy *.yaml and *.yaml to *_manual.yaml to avoid latter overwriting


# prepare single_gen scripts
mkdir -p "$WORK_DIR/single_gen"
role1_prompt_text_file="$WORK_DIR/single_gen/${ROLE_1}.txt"  # manually add local prompts according to ${WORK_DIR}/layout.txt
role2_prompt_text_file="$WORK_DIR/single_gen/${ROLE_2}.txt"  # manually add local prompts according to ${WORK_DIR}/layout.txt
role3_prompt_text_file="$WORK_DIR/single_gen/${ROLE_3}.txt"  # manually add local prompts according to ${WORK_DIR}/layout.txt
touch "$role1_prompt_text_file"
touch "$role2_prompt_text_file"
touch "$role3_prompt_text_file"
role1_yml_file="$WORK_DIR/single_gen/${ROLE_1}.yml"
role2_yml_file="$WORK_DIR/single_gen/${ROLE_2}.yml"
role3_yml_file="$WORK_DIR/single_gen/${ROLE_3}.yml"

# NOTE manually add local prompts according to ${WORK_DIR}/layout.txt before running the following scripts


# generate single_gen yaml files
UPDATE_ROLE_YAML_FILE=0
if [ $UPDATE_ROLE_YAML_FILE -eq 1 ]
then
    python story_utils/yaml_generator.py \
        --base_opt "options/test/MixofShow/EDLoRA/characters/anime/EDLoRA_hina_Anyv4_B4_Iter1K.yml" \
        --output_path "${role1_yml_file}" \
        --force_yml \
        "name=EDLoRA_${ROLE_1}" \
        "manual_seed=None" \
        "datasets.val_vis.prompts=$role1_prompt_text_file" \
        "datasets.val_vis.num_samples_per_prompt=$NUM_SAMPLE_PER_SINGE_ROLE" \
        "path.pretrain_network_g"="/root/STORY-ours/experiments/pretrained_models/edloras/chisato.pth" \
        "network_g.pretrained_path=$PATH_TO_BASE_MODEL"\
        "network_g.new_concept_token=<${ROLE_1}1>+<${ROLE_1}2>"\
        "network_g.initializer_token=<rand-0.013>+man"\
        "num_gpu=1" \
        "datasets.val_vis.replace_mapping.<TOK>=<${ROLE_1}1> <${ROLE_1}2>"


    python story_utils/yaml_generator.py \
        --base_opt "options/test/MixofShow/EDLoRA/characters/anime/EDLoRA_hina_Anyv4_B4_Iter1K.yml" \
        --output_path "${role2_yml_file}" \
        --force_yml \
        "name=EDLoRA_${ROLE_2}" \
        "manual_seed=None" \
        "datasets.val_vis.prompts=$role2_prompt_text_file" \
        "datasets.val_vis.num_samples_per_prompt=$NUM_SAMPLE_PER_SINGE_ROLE" \
        "path.pretrain_network_g"="/root/STORY-ours/experiments/pretrained_models/edloras/mazaki.pth"\
        "network_g.pretrained_path=$PATH_TO_BASE_MODEL"\
        "network_g.new_concept_token=<${ROLE_2}1>+<${ROLE_2}2>"\
        "network_g.initializer_token=<rand-0.013>+man"\
        "num_gpu=1" \
        "datasets.val_vis.replace_mapping.<TOK>=<${ROLE_2}1> <${ROLE_2}2>"



    python story_utils/yaml_generator.py \
        --base_opt "options/test/MixofShow/EDLoRA/characters/anime/EDLoRA_hina_Anyv4_B4_Iter1K.yml" \
        --output_path "${role3_yml_file}" \
        --force_yml \
        "name=EDLoRA_${ROLE_3}" \
        "manual_seed=None" \
        "datasets.val_vis.prompts=$role3_prompt_text_file" \
        "datasets.val_vis.num_samples_per_prompt=$NUM_SAMPLE_PER_SINGE_ROLE" \
        "path.pretrain_network_g"="/root/STORY-ours/experiments/pretrained_models/edloras/hayasaka.pth"\
        "network_g.pretrained_path=$PATH_TO_BASE_MODEL"\
        "network_g.new_concept_token=<${ROLE_3}1>+<${ROLE_3}2>"\
        "num_gpu=1" \
        "datasets.val_vis.replace_mapping.<TOK>=<${ROLE_3}1> <${ROLE_3}2>"
        # "network_g.finetune_cfg.unet.lora_cfg.rank=8" \
        # "network_g.finetune_cfg.text_encoder.lora_cfg.rank=8" \


fi


# run single_gen scripts
RUN_SINGLE_GEN=0
if [ $RUN_SINGLE_GEN -eq 1 ]
then
    python mixofshow/test.py -opt "${role1_yml_file}" --output_dir "${WORK_DIR}/single_gen/output_${ROLE_1}"/
    python mixofshow/test.py -opt "${role2_yml_file}" --output_dir "${WORK_DIR}/single_gen/output_${ROLE_2}"/
    python mixofshow/test.py -opt "${role3_yml_file}" --output_dir "${WORK_DIR}/single_gen/output_${ROLE_3}"/
fi


# run grounding-dino detection
mkdir -p "$WORK_DIR/single_det"
RUN_DET=0
if [ $RUN_DET -eq 1 ]
then
    cd Grounded-Segment-Anything

    python grounded_sam_demo_batch.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --input_dir "../$WORK_DIR/single_gen/output_${ROLE_1}/visualization" \
        --output_dir "../$WORK_DIR/single_det/output_${ROLE_1}/" \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "$ROLE_1_CLS" \
        --device "cuda"

    python grounded_sam_demo_batch.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --input_dir "../$WORK_DIR/single_gen/output_${ROLE_2}/visualization" \
        --output_dir "../$WORK_DIR/single_det/output_${ROLE_2}/" \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "$ROLE_2_CLS" \
        --device "cuda"

    python grounded_sam_demo_batch.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --input_dir "../$WORK_DIR/single_gen/output_${ROLE_3}/visualization" \
        --output_dir "../$WORK_DIR/single_det/output_${ROLE_3}/" \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "$ROLE_3_CLS" \
        --device "cuda"

    cd ../
fi


# run keypose detection
mkdir -p "$WORK_DIR/single_keypose"
RUN_KEYPOSE=0
if [ $RUN_KEYPOSE -eq 1 ]
then
    cd mmpose

    python demo/image_demo_batch.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --input_dir "../$WORK_DIR/single_det/output_${ROLE_1}/" \
    --image_type "image_crop.png" \
    --output_dir "../$WORK_DIR/single_keypose/output_${ROLE_1}/" \
    --kpt-thr 0.3

    python demo/image_demo_batch.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --input_dir "../$WORK_DIR/single_det/output_${ROLE_2}/" \
    --image_type "image_crop.png" \
    --output_dir "../$WORK_DIR/single_keypose/output_${ROLE_2}/" \
    --kpt-thr 0.3

    python demo/image_demo_batch.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --input_dir "../$WORK_DIR/single_det/output_${ROLE_3}/" \
    --image_type "image_crop.png" \
    --output_dir "../$WORK_DIR/single_keypose/output_${ROLE_3}/" \
    --kpt-thr 0.3

    cd ../
fi


# run sketch generation
mkdir -p "$WORK_DIR/single_sketch"
RUN_SKETCH=0
if [ $RUN_SKETCH -eq 1 ]
then
    cd pidinet

    python main_batch.py \
    --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 1 --dataset Custom --evaluate trained_models/table5_pidinet.pth --evaluate-converted \
    --datadir "../$WORK_DIR/single_det/output_${ROLE_1}/" \
    --savedir "../$WORK_DIR/single_sketch/output_${ROLE_1}/" \
    --infer_thr 0.5

    python main_batch.py \
        --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 1 --dataset Custom --evaluate trained_models/table5_pidinet.pth --evaluate-converted \
        --datadir "../$WORK_DIR/single_det/output_${ROLE_2}/" \
        --savedir "../$WORK_DIR/single_sketch/output_${ROLE_2}/" \
        --infer_thr 0.5

    python main_batch.py \
        --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 1 --dataset Custom --evaluate trained_models/table5_pidinet.pth --evaluate-converted \
        --datadir "../$WORK_DIR/single_det/output_${ROLE_3}/" \
        --savedir "../$WORK_DIR/single_sketch/output_${ROLE_3}/" \
        --infer_thr 0.5

    cd ../
fi


# compose condition
GET_KEYPOSE_PATHS=0
if [ $GET_KEYPOSE_PATHS -eq 1 ]
then
    # # create configs, moved to process_bbox.py
    # mkdir -p "$WORK_DIR/composed_keypose"
    # python story_utils/single_yaml_generator.py \
    #     --base_opt "story_utils/config_keypose_compose.yaml" \
    #     --output_path "$WORK_DIR/composed_keypose/config_keypose_compose.yaml" \
    #     --force_yml \
    #     "output_dir=$WORK_DIR/composed_keypose"

    # save potential paths to ease the manual writing
    python story_utils/get_potential_paths.py \
        --potential_dir "$WORK_DIR/single_keypose" \
        --depth 3 \
        --file_key "_pose.png" \
        --save_path "$WORK_DIR/composed_keypose/potential_paths.txt"

    # NOTE manually add local prompts according to ${WORK_DIR}/layout.txt before running the following scripts
fi

RUN_KEPOSE_COMPOSE=0
if [ $RUN_KEPOSE_COMPOSE -eq 1 ]
then
    python story_utils/compose_keypose.py --config "$WORK_DIR/composed_keypose/compose_keypose_manual.yaml"
fi


# run final generation
mkdir -p "$WORK_DIR/final_gen"
RUN_FINAL_GEN=0
if [ $RUN_FINAL_GEN -eq 1 ]
then
    python inference/mix_of_show_sample_batch.py \
        --config "$WORK_DIR/final_gen/final_gen.yaml" \
        --pretrained_model $PATH_TO_BASE_MODEL \
        --combined_model $PATH_TO_FUSED_MODEL \
        --keypose_adaptor_model "${PATH_TO_T2IADAPTOR}/t2iadapter_keypose_sd14v1.pth" \
        --sketch_adaptor_model "${PATH_TO_T2IADAPTOR}/t2iadapter_sketch_sd14v1.pth" \
        --save_dir "$WORK_DIR/final_gen" \
        --pipeline_type "adaptor_pplus" \
        --num_samples_per_panel $NUM_SAMPLE_PER_PANEL
fi
