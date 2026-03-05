import os
import time
import argparse
import openai
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
# openai.api_key = os.getenv("OPENAI_API_KEY")


bg_prompt_text = "Background prompt: "

default_template = """You are an intelligent bounding box generator. I will provide you with a caption for a photo, image, or painting. Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene. The images are of size 512x512, and the bounding boxes should not overlap or go beyond the image boundaries. Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and include exactly one object. Make the boxes larger if possible. Do not put objects that are already provided in the bounding boxes into the background prompt. If needed, you can make reasonable guesses. Generate the object descriptions and background prompts in English even if the caption might not be in English. Do not include non-existing or excluded objects in the background prompt. Please refer to the example below for the desired format.

Caption: A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky.
Objects: [('a green car', [21, 181, 211, 159]), ('a blue truck', [269, 181, 209, 160]), ('a red air balloon', [66, 8, 145, 135]), ('a bird', [296, 42, 143, 100])]
Background prompt: A realistic image of a landscape scene

Caption: A watercolor painting of a wooden table in the living room with an apple on it.
Objects: [('a wooden table', [65, 243, 344, 206]), ('a apple', [206, 306, 81, 69])]
Background prompt: A watercolor painting of a living room

Caption: A watercolor painting of two pandas eating bamboo in a forest.
Objects: [('a panda eating bambooo', [30, 171, 212, 226]), ('a panda eating bambooo', [264, 173, 222, 221])]
Background prompt: A watercolor painting of a forest

Caption: A realistic image of four skiers standing in a line on the snow near a palm tree.
Objects: [('a skier', [5, 152, 139, 168]), ('a skier', [278, 192, 121, 158]), ('a skier', [148, 173, 124, 155]), ('a palm tree', [404, 180, 103, 180])]
Background prompt: A realistic image of an outdoor scene with snow

Caption: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea.
Objects: [('a steam boat', [232, 225, 257, 149]), ('a jumping pink dolphin', [21, 249, 189, 123])]
Background prompt: An oil painting of the sea

Caption: A realistic image of a cat playing with a dog in a park with flowers.
Objects: [('a playful cat', [51, 67, 271, 324]), ('a playful dog', [302, 119, 211, 228])]
Background prompt: A realistic image of a park with flowers"""

# Caption: 一个客厅场景的油画，墙上挂着电视，电视下面是一个柜子，柜子上有一个花瓶。
# Objects: [('a tv', [88, 85, 335, 203]), ('a cabinet', [57, 308, 404, 201]), ('a flower vase', [166, 222, 92, 108])]
# Background prompt: An oil painting of a living room scene

simplified_prompt = """{template}

Caption: {prompt}
Objects: """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--story", type=str, help="user input prompt",
                        default="Write a short story between Tezuka Kunimitsu and Hina Amano.")
    parser.add_argument("--model", type=str, default="gpt-4", help="user input prompt",
                        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"])
    parser.add_argument("--output_dir", type=str, default="output_stories", help="path to output file")
    args = parser.parse_args()
    return args


def generate_response(prompt, url="http://10.15.82.10:8000/v1/chat/completions", model="gpt-4"):
# def generate_response(prompt, url="http://10.15.82.10:8000/v1/chat/completions", model="gpt-3.5-turbo"):
    assert model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]
    # apiKey = "<your-api-key>"
    # headers = {"Authorization": "Bearer "+ apiKey}
    headers = None  # no need for openai-api-key

    data = {
        "model": model,
        "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."  # TODO: more advanced system prompt
        },
        {
            "role": "user",
            "content": prompt,
        }
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    response = response.json()

    # complete text
    completed_text = response['choices'][0]['message']['content']
    print(f"prompt: {prompt}")
    print(f"completed_text: {completed_text}")
    return completed_text, response


if __name__ == "__main__":
    
    args = get_args()
    
    # generate the story
    prompt = args.story + " " + "Never use 'he', 'she', 'it', or 'they' in the story."
    story, response_story = generate_response(prompt)

    # split the story into sentences
    prompt = "Split the above story to several sentences, each sentence corresponds to a single panel in a comic and starts with 'Panel:' "
    panels, response_panels = generate_response(story + '\n' + prompt)

    # split the panels into sentences
    sentence_list = []
    for sentence in panels.split('\n'):
        sentence = sentence.strip()
        if sentence == '':
            continue
        sentence_parts = sentence.split(':')
        if not len(sentence_parts) == 2:
            print("abnormal sentence: ", sentence)
            continue
        assert 'Panel' in sentence_parts[0], sentence_parts[0]
        sentence_list.append(sentence_parts[1].strip())
    sentence_list = sentence_list[:-1]  # remove the last incomplete sentence (due to max_tokens truncation)

    # generate prompts with the sentences
    prompt = "Generate a single prompt starts with 'Prompt:' from the following story for stable diffusion to generate images, depicting the event, character, and scene."
    prompt_to_sd_list = []
    for sentence in sentence_list:
        prompt_to_sd, response_to_sd = generate_response(prompt + '\n' + sentence)
        prompt_to_sd_parts = prompt_to_sd.split('Prompt: ')
        if not len(prompt_to_sd_parts) == 2:
            print("abnormal sentence: ", prompt_to_sd)
            continue
        prompt_to_sd_list.append(prompt_to_sd_parts[1])

    # generate the layout
    # TODO: add previous layout as context !!!
    layout_list = []
    for prompt_to_sd in prompt_to_sd_list:
        prompt  = simplified_prompt.format(template=default_template, prompt=prompt_to_sd)
        layout, response_layout = generate_response(prompt)
        layout = f"Caption: {prompt_to_sd}\nObjects: " + layout
        layout_list.append(layout.strip())

    # save all the data
    replicate = 0
    output_dir = os.path.join(args.output_dir, args.model + "_" + "512x512", args.story.replace(" ", "_"), f"replicate_{replicate}")
    while os.path.exists(output_dir):
        replicate += 1
        output_dir = os.path.join(args.output_dir, args.model, args.story, replicate)
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    
    file_name = os.path.join(output_dir, f"story.txt")
    with open(os.path.join(output_dir, f"story.txt"), 'w') as f:
        f.write(story + '\n')

    with open(os.path.join(output_dir, f"panels.txt"), 'w') as f:
        for sentence in sentence_list:
            f.write(sentence + '\n')

    with open(os.path.join(output_dir, f"prompts.txt"), 'w') as f:
        for prompt_to_sd in prompt_to_sd_list:
            f.write(prompt_to_sd + '\n')

    with open(os.path.join(output_dir, f"layout.txt"), 'w') as f:
        for layout in layout_list:
            f.write(layout + '\n\n')
