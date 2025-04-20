import os
import json
from dotenv import load_dotenv
import random
from vertexai.vision_models import ImageGenerationModel
import base64
import io
from PIL import Image
from vertexai import init

init(project="intro-deep-learning-449516")

load_dotenv(override=True)

ROOT = os.getcwd()
STYLES = ["impressionist", "expressionist"]
PROMPT_BASE = (
    "Create a single {style}-style painting which depicts the following scene: "
)

MODEL = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

def collect_images_prompt(prompt_base, caption_file="ArtCap.json", shuffle=False):
    prompt_base = PROMPT_BASE.format(style=random.choice(STYLES))
    with open(os.path.join(ROOT, caption_file), "r") as file:
        captions = json.load(file)
    prompt_dict = {i: prompt_base + ". ".join(caps) for i, caps in captions.items()}

    if shuffle:
        keys = list(prompt_dict.keys())
        random.shuffle(keys)
        prompt_dict = {key: prompt_dict[key] for key in keys}

    return prompt_dict


def generate_image(prompt):
        # Generate the image
    response = MODEL.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="1:1",
        add_watermark=False,
        # safety_filter_level="block_none"
        # person_generation="allow_all"
    )
    print(response)
    
    # Get the image data
    image_data = response[0]
    
    return image_data


def generate_and_save_images(
    image_prompt_dict,
    generator="titan",
    output_dir="generated_images",
    max_invocations=1,
):
    os.makedirs(os.path.join(ROOT, output_dir), exist_ok=True)
    flagged_images_path = os.path.join(output_dir, "flagged_images.txt")
    try:
        with open(flagged_images_path, "r") as f:
            flagged = set(line.strip() for line in f)
    except FileNotFoundError:
        flagged = set()

    num_invocations = 0
    for image, prompt in image_prompt_dict.items():
        output_file = f"{generator}_{image}"
        output_path = os.path.join(output_dir, output_file)
        try:
            ##COMPLETE THIS FUNCTION
            if os.path.isfile(output_path):
                print(f"{image}\t| Image already exists")
                continue
            elif output_path in flagged:
                print(f"{image}\t| Prompt is in blacklist")
            else:
                num_invocations += 1
                image_data = generate_image(prompt)
                image_data.save(output_path)
                print(f"{image}\t| Successfully created image")
        except Exception as e:
            if "violate" in str(e):
                if output_path not in flagged:
                    with open(flagged_images_path, "a") as f:
                        f.write(f"{output_path}\n")
                    flagged.add(output_path)
                print(f"{image}\t| Prompt was flagged as inappropriate by API")
            else:
                print(f"{image}\t| {e}")

        if num_invocations >= max_invocations:
            break

    # return number of successfuly written images
    return len(os.listdir(os.path.join(ROOT, output_dir)))


image_prompt_dict = collect_images_prompt(PROMPT_BASE, shuffle=True)
print(len(image_prompt_dict))
success_count = generate_and_save_images(
    image_prompt_dict,
    generator="imagen",
    output_dir="images/imagen",
    max_invocations=1,
)
print(f"Generated {success_count} images successfully")
