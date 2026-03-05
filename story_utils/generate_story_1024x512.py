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

default_template = """You are an intelligent bounding box generator. I will provide you with a caption for a photo, image, or painting. Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene. The images are of hight 512 and width 1024 and the bounding boxes should not overlap or go beyond the image boundaries. Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and include exactly one object. Make the boxes larger if possible. Do not put objects that are already provided in the bounding boxes into the background prompt. If needed, you can make reasonable guesses. Generate the object descriptions and background prompts in English even if the caption might not be in English. Do not include non-existing or excluded objects in the background prompt. Please refer to the example below for the desired format.

Caption: A girl in red dress, a girl wearing a hat, and a boy in white suit are walking near a lake.
Objects: [('a girl in red dress, near a lake', [115, 61, 158, 451]), ('a boy in white suit, near a lake', [292, 19, 220, 493]), ('a girl wearing a hat, near a lake', [519, 48, 187, 464])]
Background prompt: A lake

Caption: A woman and a man, both in hogwarts school uniform, holding hands, facing a strong monster, near the castle.
Objects: [('a man, in hogwarts school uniform, holding hands, near the castle', [3, 2, 258, 510]), ('a woman, in hogwarts school uniform, holding hands, near the castle', [207, 7, 253, 505]), ('a strong monster, near the castle', [651, 1, 345, 511])]
Background prompt: A castle

Caption: A man sit in a chair, a dog and a cat sit on a table, in a living room.
Objects: [('a man sit on a chair, in a living room', [0, 0, 400, 512]), ('A dog, sit, in a living room', [501, 60, 205, 290]), ('A cat, sit, in a living room', [692, 57, 248, 286]), ('a table, in a living room', [423, 280, 560, 228])]
Background prompt: A living room

Caption: Two dogs and a cat, on the grass, under the sunset, animal photography.
Objects: [('a dog, on the grass, under the sunset, animal photography', [76, 160, 274, 345]), ('a cat, on the grass, under the sunset, animal photography', [370, 162, 315, 338]), ('a dog, on the grass, under the sunset, animal photography', [666, 134, 339, 378])]
Background prompt: The grass, under the sunset, animal photography"""

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
    prompt = args.story + ". " + "Never use 'he', 'she', 'it', or 'they' in the story. \
        Do not call the subjects in general like using 'a person', 'they', 'a girl', 'the trio'. \
            Make sure when you describe the subjects, must use their names!"
    story, response_story = generate_response(prompt)

    # split the story into sentences
    prompt = "Split the above story to several sentences, each sentence corresponds to a single panel in a comic and starts with 'Panel:'. \
        And you must clarify the name of characters clearly in each panel. Do not call the subjects in general like using 'a person', 'they', 'a girl', 'the trio'. \
            Make sure in each panel when you describe the subjects, must use their names!"
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
    output_dir = os.path.join(args.output_dir, args.model + "_" + "1024x512", args.story, f"replicate_{replicate}")
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
