
<p align="center">

  <h2 align="center">AutoStory: <br> Generating Diverse Storytelling Images with Minimal Human Effort</h2>
  <p align="center">
    <a href="https://github.com/encounter1997"><strong>Wen Wang*</strong></a>
    ·
    <a href="https://github.com/volcverse"><strong>Canyu Zhao*</strong></a>
    ·  
    <a href="https://scholar.google.com/citations?user=FaOqRpcAAAAJ"><strong>Hao Chen</strong></a>
    ·
    <a href="https://github.com/Aziily"><strong>Zhekai Chen</strong></a>
    ·
    <a href="https://zkcys001.github.io/"><strong>Kecheng Zheng</strong></a>
    ·
    <a href="https://cshen.github.io/"><strong>Chunhua Shen</strong></a>
    <br>
    Zhejiang University
    <br>
    </br>
        <a href="https://arxiv.org/abs/2311.11243">
        <img src='https://img.shields.io/badge/arxiv-AutoStory-blue' alt='Paper PDF'></a>
        <a href="https://aim-uofa.github.io/AutoStory/">
        <img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>
  </p>
</p>


<image src="docs/teaser.png" />

Story visualization aims to generate a series of images that match the story described in texts, and it requires the generated images to satisfy high quality, alignment with the text description, and consistency in character identities. Given the complexity of story visualization, existing methods drastically simplify the problem by considering only a few specific characters and scenarios, or requiring the users to provide per-image control conditions such as sketches. However, these simplifications render these methods incompetent for real applications. 

To this end, we propose an automated story visualization system that can effectively generate diverse, high-quality, and consistent sets of story images, with minimal human interactions. Specifically, we utilize the comprehension and planning capabilities of large language models for layout planning, and then leverage large-scale text-to-image models to generate sophisticated story images based on the layout. We empirically find that sparse control conditions, such as bounding boxes, are suitable for layout planning, while dense control conditions, e.g., sketches, and keypoints, are suitable for generating high-quality image content. To obtain the best of both worlds, we devise a dense condition generation module to transform simple bounding box layouts into sketch or keypoint control conditions for final image generation, which not only improves the image quality but also allows easy and intuitive user interactions. 

In addition, we propose a simple yet effective method to generate multi-view consistent character images, eliminating the reliance on human labor to collect or draw character images. This allows our method to obtain consistent story visualization even when only texts are provided as input. Both qualitative and quantitative experiments demonstrate the superiority of our method.


## Results

<image src="docs/results.png" />


## Overview

The full pipeline consists of the following stages:

1. **Train Single-Character LoRA** — Fine-tune ED-LoRA for each character using Mix-of-Show
2. **Gradient Fusion** — Fuse multiple ED-LoRAs into a single model
3. **Generate Single-Character Images** — Produce reference images per character
4. **Person Detection** — Detect characters using Grounding-DINO + SAM
5. **Pose Estimation** — Extract keypose via MMPose (HRNet)
6. **Layout Generation** — Use LLM (e.g., GPT-4) to plan bounding boxes
7. **Compose Keypose Layout** — Assemble per-character poses into a full panel
8. **Final Multi-Character Generation** — Regionally controllable generation with Mix-of-Show

---

## 🛠️ Installation

### Prerequisites

- Python >= 3.9 (recommend [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.12](https://pytorch.org/) with CUDA support
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Linux

### 1. Mix-of-Show (Core Framework)

```bash
# Install diffusers==0.14.0 with T2I-Adapter support
cd diffusers-t2i-adapter
pip install .
cd ..

# Install this repo
python setup.py install
```

### 2. Grounding-DINO + SAM

```bash
cd Grounded-Segment-Anything

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda  # e.g., /usr/local/cuda-11.8

python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

cd ..
```

### 3. MMPose

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"

cd mmpose
pip install -r requirements.txt
pip install -v -e .
cd ..
```

---

## ⏬ Pre-trained Weights

Download the following pre-trained models and place them under `experiments/pretrained_models/`:

| Model | Description | Link |
| --- | --- | --- |
| ChilloutMix | Base diffusion model (real-world style) | [HuggingFace](https://huggingface.co/windwhinny/chilloutmix) |
| T2I-Adapter (sketch) | Sketch condition adapter | [HuggingFace](https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_sketch_sd14v1.pth) |
| T2I-Adapter (openpose) | Keypose condition adapter | [HuggingFace](https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_openpose_sd14v1.pth) |
| Grounding-DINO | Zero-shot object detector | [GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| SAM (ViT-H) | Segment Anything Model | [GitHub](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| HRNet-w48 | Pose estimation model | [MMPose Model Zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html) |

```bash
cd experiments/pretrained_models

# ChilloutMix
git lfs clone https://huggingface.co/windwhinny/chilloutmix.git

# T2I-Adapters
mkdir -p t2i_adaptor && cd t2i_adaptor
wget https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_sketch_sd14v1.pth
wget https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_openpose_sd14v1.pth
cd ..

# Grounding-DINO weights (place in experiments/pretrained_models/)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

cd ../..
```

Expected directory structure:

```
STORY/
├── experiments/
│   └── pretrained_models/
│       ├── chilloutmix/
│       ├── t2i_adaptor/
│       │   ├── t2iadapter_sketch_sd14v1.pth
│       │   └── t2iadapter_openpose_sd14v1.pth
│       ├── groundingdino_swint_ogc.pth
│       └── sam_vit_h_4b8939.pth
├── mixofshow/
├── inference/
├── story_utils/
├── scripts/
├── datasets/
├── options/
├── Grounded-Segment-Anything/
├── mmpose/
└── ...
```

---

## 🚀 Pipeline Usage

Below is the full AutoStory pipeline. We use a movie character example (American Beauty) to illustrate each step.

### Step 1: Train Single-Character ED-LoRA

Train an ED-LoRA for each character following [Mix-of-Show](https://github.com/TencentARC/Mix-of-Show). Prepare per-character training images and configs in `datasets/data_cfgs/`.

```bash
# Example: Train ED-LoRA for one character (2 GPUs, ~5-10 min)
python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=2234 mixofshow/train.py \
    -opt options/train/EDLoRA/movie/EDLoRA_Lester_Burnham_Cmix_B4_Iter1K.yml \
    --launcher pytorch
```

Repeat for each character in your story.

### Step 2: Gradient Fusion (Multi-Character Model)

Fuse all single-character ED-LoRAs into one combined model:

```bash
export config_file="Jane_Burnham+Lester_Burnham+Carolyn_Burnham+Jim_Olmeyer+Jim_Berkley_Iter500"

python scripts/mixofshow_scripts/Gradient_Fusion_EDLoRA.py \
    --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/movie/${config_file}.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50
```

### Step 3: Generate Single-Character Reference Images

Generate diverse single-character images for pose extraction:

```bash
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=2234 mixofshow/test.py \
    -opt options/test/MixofShow/EDLoRA/characters/movie/EDLoRA_Lester_Burnham_Cmix_B4_Iter1K.yml \
    --launcher pytorch
```

Repeat for each character. Customize prompts in the corresponding `.txt` files specified in config.

### Step 4: Person Detection with Grounding-DINO + SAM

Detect and crop characters from the generated single-character images:

```bash
cd Grounded-Segment-Anything

export CUDA_VISIBLE_DEVICES=0

INPUT_DIR_LIST=(
    "../results/EDLoRA_Carolyn_Burnham_Cmix_B4_Iter1K/visualization/iters_EDLoRA_Carolyn_Burnham_Cmix_B4_Iter1K_negprompt"
    "../results/EDLoRA_Jane_Burnham_Cmix_B4_Iter1K/visualization/iters_EDLoRA_Jane_Burnham_Cmix_B4_Iter1K_negprompt"
    "../results/EDLoRA_Lester_Burnham_Cmix_B4_Iter1K/visualization/iters_EDLoRA_Lester_Burnham_Cmix_B4_Iter1K_negprompt"
)

for((i=0;i<${#INPUT_DIR_LIST[@]};i++));
do
    echo "Processing ${INPUT_DIR_LIST[i]}"
    python grounded_sam_demo_batch.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint ../experiments/pretrained_models/groundingdino_swint_ogc.pth \
        --sam_checkpoint ../experiments/pretrained_models/sam_vit_h_4b8939.pth \
        --input_dir ${INPUT_DIR_LIST[i]} \
        --output_dir ${INPUT_DIR_LIST[i]}_grounding_dino \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "person" \
        --device "cuda"
done

cd ..
```

See also: `Grounded-Segment-Anything/run_grounded_sam_batch.sh`

### Step 5: Pose Estimation with MMPose

Extract keypose from detected character crops:

```bash
cd mmpose

INPUT_DIR_LIST=(
    "../results/EDLoRA_Carolyn_Burnham_Cmix_B4_Iter1K/visualization/iters_EDLoRA_Carolyn_Burnham_Cmix_B4_Iter1K_negprompt_grounding_dino"
    "../results/EDLoRA_Jane_Burnham_Cmix_B4_Iter1K/visualization/iters_EDLoRA_Jane_Burnham_Cmix_B4_Iter1K_negprompt_grounding_dino"
    "../results/EDLoRA_Lester_Burnham_Cmix_B4_Iter1K/visualization/iters_EDLoRA_Lester_Burnham_Cmix_B4_Iter1K_negprompt_grounding_dino"
)

for((i=0;i<${#INPUT_DIR_LIST[@]};i++));
do
    python demo/image_demo_batch.py \
        configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
        hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
        --input_dir ${INPUT_DIR_LIST[i]} \
        --image_type "image_crop.png" \
        --output_dir "${INPUT_DIR_LIST[i]}_pose" \
        --kpt-thr 0.3
done

cd ..
```

See also: `mmpose/run_mmpose_batch.sh`

### Step 6: Layout Generation with LLM

Use an LLM (e.g., GPT-4) to generate bounding box layouts for each story panel. The prompt template:

<details>
<summary>Click to expand LLM prompt template</summary>

```
You are an intelligent bounding box generator. I will provide you with a caption for a
photo, image, or painting. Your task is to generate the bounding boxes for the objects
mentioned in the caption, along with a background prompt describing the scene. The images
are of height 512 and width 1024 and the bounding boxes should not overlap or go beyond
the image boundaries. Each bounding box should be in the format of (object name,
[top-left x coordinate, top-left y coordinate, box width, box height]) and include
exactly one object. Make the boxes larger if possible. Do not put objects that are already
provided in the bounding boxes into the background prompt. If needed, you can make
reasonable guesses. Generate the object descriptions and background prompts in English
even if the caption might not be in English. Do not include non-existing or excluded
objects in the background prompt. Please refer to the example below for the desired format.

Caption: A girl in red dress, a girl wearing a hat, and a boy in white suit are walking near a lake.
Objects: [('a girl in red dress, near a lake', [115, 61, 158, 451]), ('a boy in white suit, near a lake', [292, 19, 220, 493]), ('a girl wearing a hat, near a lake', [519, 48, 187, 464])]
Background prompt: A lake

Caption: A woman and a man, both in hogwarts school uniform, holding hands, facing a strong monster, near the castle.
Objects: [('a man, in hogwarts school uniform, holding hands, near the castle', [3, 2, 258, 510]), ('a woman, in hogwarts school uniform, holding hands, near the castle', [207, 7, 253, 505]), ('a strong monster, near the castle', [651, 1, 345, 511])]
Background prompt: A castle

Caption: <your story panel caption>
Objects:
```

**Example output:**

```
Caption: Lester is asleep in the back seat of a car. Carolyn is driving, and Jane is sitting in the front passenger seat.
Objects: [('Lester, asleep in the back seat', [400, 150, 280, 350]), ('Carolyn, driving the car', [100, 100, 250, 300]), ('Jane, sitting in the front passenger seat', [250, 150, 200, 280]), ('the car', [0, 0, 1024, 512])]
Background prompt: Inside a car
```

</details>

You can also use the built-in story generation script:

```bash
python story_utils/generate_story_1024x512.py \
    --story "Write a short story about 3 characters..." \
    --output_dir output_stories \
    --model gpt-4
```

### Step 7: Compose Keypose Layout

Select suitable per-character pose images and compose them into a full panel layout. Create a YAML config:

```yaml
output_dir: results/movie_keypose
image_width: 1024

layout1:
  output_image: "panel_001.png"
  box_layout:
    - [400, 200, 80, 160]   # Character A position [x, y, w, h]
    - [256, 384, 64, 128]   # Character B position
  keypose_path:
    - "path/to/characterA_pose.png"
    - "path/to/characterB_pose.png"

layout2:
  output_image: "panel_002.png"
  box_layout:
    - [183, 144, 141, 173]  # Character A
    - [141, 181, 115, 153]  # Character B
    - [274, 220, 145, 171]  # Character C
  keypose_path:
    - "path/to/characterA_pose.png"
    - "path/to/characterB_pose.png"
    - "path/to/characterC_pose.png"
```

Then compose:

```bash
python story_utils/compose_keypose.py --config path/to/config_keypose_compose.yaml
```

### Step 8: Final Multi-Character Generation

Run regionally controllable generation with the fused model, keypose conditions, and per-region prompts:

```bash
combined_model_root="experiments/composed_edlora/chilloutmix"
expdir="Jane_Burnham+Lester_Burnham+Carolyn_Burnham+Jim_Olmeyer+Jim_Berkley"
SEED=100

keypose_condition='results/movie_keypose/panel_001.png'
keypose_adaptor_weight=1.0
sketch_condition=''
sketch_adaptor_weight=0.5

context_prompt='Lester is asleep in the back seat of a car. Carolyn is driving, and Jane is sitting in the front passenger seat.'
context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

region1_prompt='[a <Lester_Burnham1> <Lester_Burnham2>, wearing a suit, sleeping in the back seat, 4K, high quality, high resolution, best quality]'
region1_neg_prompt="[${context_neg_prompt}]"
region1='[400, 150, 280, 350]'

region2_prompt='[a <Carolyn_Burnham1> <Carolyn_Burnham2>, wearing a suit, driving the car, 4K, high quality, high resolution, best quality]'
region2_neg_prompt="[${context_neg_prompt}]"
region2='[100, 100, 250, 300]'

region3_prompt='[a <Jane_Burnham1> <Jane_Burnham2>, wearing a dark jacket, sitting in the front passenger seat, 4K, high quality, high resolution, best quality]'
region3_neg_prompt="[${context_neg_prompt}]"
region3='[250, 150, 200, 280]'

prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model="${combined_model_root}/${expdir}/combined_model_.pth" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adaptor/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight} \
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adaptor/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight} \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=${SEED}
```

For batch generation of multiple panels, use the batch script:

```bash
python inference/mix_of_show_sample_batch.py \
    --config "$WORK_DIR/final_gen/final_gen.yaml" \
    --pretrained_model experiments/pretrained_models/chilloutmix \
    --combined_model path/to/combined_model_.pth \
    --keypose_adaptor_model "experiments/pretrained_models/t2i_adaptor/t2iadapter_openpose_sd14v1.pth" \
    --sketch_adaptor_model "experiments/pretrained_models/t2i_adaptor/t2iadapter_sketch_sd14v1.pth" \
    --save_dir "$WORK_DIR/final_gen" \
    --pipeline_type "adaptor_pplus"
```

---

## ⚡ End-to-End Script

For an end-to-end run, see `run_story_1024.sh` which chains all steps together. Toggle each stage with the flag variables at the top of the script:

```bash
bash run_story_1024.sh
```

Key flags in the script:
- `GENERATE_STORY=1` — Generate story panels via LLM
- `PROCESS_BOX=1` — Process bounding box layouts
- `UPDATE_ROLE_YAML_FILE=1` — Create per-character test configs
- `RUN_SINGLE_GEN=1` — Generate single-character images
- `RUN_DET=1` — Run Grounding-DINO detection
- `RUN_KEYPOSE=1` — Run MMPose keypose estimation
- `RUN_KEPOSE_COMPOSE=1` — Compose keypose layouts
- `RUN_FINAL_GEN=1` — Run final multi-character generation

---

## 💡 Tips

- **Larger bounding boxes** produce higher-quality characters. Adjust LLM outputs as needed.
- **Pose selection matters** — manually review and select the best single-character pose for each panel.
- **ED-LoRA quality is critical** — invest effort in training good single-character ED-LoRAs before fusion, as this determines final quality.
- **LLM outputs are starting points** — tune prompts, layouts, and bounding boxes manually for best results.

---

## 📜 License and Acknowledgement

For non-commercial academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).


This codebase builds on the following open-source projects:

- [Mix-of-Show](https://github.com/TencentARC/Mix-of-Show) — Multi-concept customization of diffusion models
- [diffusers](https://github.com/huggingface/diffusers) — Diffusion model library
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) — Grounding-DINO + SAM
- [MMPose](https://github.com/open-mmlab/mmpose) — Pose estimation toolkit
- [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) — Spatial condition adapters
- [PiDiNet](https://github.com/zhuoinoulu/pidinet) — Edge detection for sketch conditions
- [LoRA for Diffusion Models](https://github.com/cloneofsimo/lora)
- [Custom Diffusion](https://github.com/adobe-research/custom-diffusion)


## 🌏 Citation

If you find our work useful, please consider citing:

```BibTeX
@article{AutoStory,
  title={AutoStory: Generating Diverse Storytelling Images with Minimal Human Effort},
  author={Wang, Wen and Zhao, Canyu and Chen, Hao and Chen, Zhekai and Zheng, Kecheng and Shen, Chunhua},
  journal={Int. J. Computer Vision},
  year={2024},
}
```