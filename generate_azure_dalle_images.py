import os
import json
from openai import AzureOpenAI, BadRequestError
from dotenv import load_dotenv
import random
import requests

load_dotenv(override=True)

ROOT = os.getcwd()
STYLES = ["impressionist", "expressionist"]
PROMPT_BASE = "Create an {style}-style painting which depicts the following scene: "

CLIENT = AzureOpenAI(
    api_version="2024-02-01",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)


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
    # Generate image through Azure DALL-E 3
    result = CLIENT.images.generate(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # Your DALL-E 3 deployment name
        prompt=prompt,
        n=1,
        size="1024x1024",
    )

    # Extract image URL from response
    image_url = result.data[0].url

    # Download and return image data
    response = requests.get(image_url)
    return response.content


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
                with open(output_path, "wb") as file:
                    file.write(image_data)
                print(f"{image}\t| Successfully created image")
        except BadRequestError as e:
            if output_path not in flagged:
                with open(flagged_images_path, "a") as f:
                    f.write(f"{output_path}\n")
                flagged.add(output_path)
            print(f"{image}\t| Prompt was flagged as inappropriate by API")
        except Exception as e:
            print(f"{image}\t| {e}")

        if num_invocations >= max_invocations:
            break

    # return number of successfuly written images
    return len(os.listdir(os.path.join(ROOT, output_dir)))


image_prompt_dict = collect_images_prompt(PROMPT_BASE, shuffle=True)
print(len(image_prompt_dict))
success_count = generate_and_save_images(
    image_prompt_dict,
    generator="dalle",
    output_dir="images/dalle",
    max_invocations=250,
)
print(f"Generated {success_count} images successfully")
